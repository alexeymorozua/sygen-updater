"""Sygen updater sidecar.

Runs inside the compose stack on the shared docker network. Responsibilities:

1. Every ``CHECK_INTERVAL`` seconds, query GHCR for the current remote digest
   of the core and admin images and compare it to whatever digest is
   currently running on the local docker daemon. Write the result atomically
   to ``/state/updates.json`` — core reads it back via ``/api/system/updates``.
2. Expose ``GET /health`` (unauthenticated — used by compose healthchecks).
3. Expose ``POST /apply`` (bearer-auth via ``SYGEN_UPDATER_TOKEN``). When
   called, runs ``docker compose pull && docker compose up -d`` scoped to
   the ``core`` and ``admin`` services. The updater itself and watchtower
   are deliberately excluded — the operator redeploys those out-of-band so
   an apply never kills the apply request mid-flight.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse

logger = logging.getLogger("sygen-updater")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", "1800"))
STATE_PATH = Path(os.environ.get("STATE_PATH", "/state/updates.json"))
COMPOSE_FILE = os.environ.get("COMPOSE_FILE", "/srv/sygen/docker-compose.yml")
COMPOSE_ENV_FILE = os.environ.get("COMPOSE_ENV_FILE", "/srv/sygen/.env")
DOCKER_SOCKET = os.environ.get("DOCKER_SOCKET", "/var/run/docker.sock")
UPDATER_TOKEN = os.environ.get("SYGEN_UPDATER_TOKEN", "")

# Image references. These are the *full* image strings used in compose
# (``ghcr.io/owner/repo:tag``). We need tag + repo for manifest lookup, so we
# parse them lazily.
CORE_IMAGE = os.environ.get("CORE_IMAGE", "ghcr.io/alexeymorozua/sygen-core:latest")
ADMIN_IMAGE = os.environ.get("ADMIN_IMAGE", "ghcr.io/alexeymorozua/sygen-admin:latest")

# Container names we query for the currently-running digest. Must match
# the ``container_name:`` fields in the compose file.
CORE_CONTAINER = os.environ.get("CORE_CONTAINER", "sygen-core")
ADMIN_CONTAINER = os.environ.get("ADMIN_CONTAINER", "sygen-admin")

# Compose services that ``apply`` is allowed to restart. The updater
# itself and watchtower are NOT in this list on purpose (see module
# docstring).
APPLY_SERVICES = ("core", "admin")

# Track last successful check for /health diagnostics.
_last_check: Optional[str] = None
_last_error: Optional[str] = None

# Per-user rate limit for /apply. The spec says "1 per minute per user",
# but the updater does not know the user — it only sees the bearer token.
# Enforce a global 1-per-60s instead; core already hops through JWT/admin
# checks before proxying, so this is belt-and-braces.
_APPLY_MIN_INTERVAL = 60.0
_last_apply_at: float = 0.0
_apply_lock = asyncio.Lock()

app = FastAPI(title="sygen-updater", version="1.0.0")


# ---------------------------------------------------------------------------
# GHCR manifest + local digest probing
# ---------------------------------------------------------------------------


def _parse_image(image: str) -> tuple[str, str]:
    """Split ``ghcr.io/owner/repo:tag`` into (``owner/repo``, ``tag``).

    Digest pins (``@sha256:...``) are passed through as the "tag".
    Defaults to ``latest`` if no tag is given.
    """
    if image.startswith("ghcr.io/"):
        image = image[len("ghcr.io/") :]
    if "@" in image:
        repo, digest = image.split("@", 1)
        return repo, digest
    if ":" in image:
        repo, tag = image.rsplit(":", 1)
        return repo, tag
    return image, "latest"


async def _ghcr_token(client: httpx.AsyncClient, repo: str) -> str:
    """Fetch a short-lived pull token from GHCR's anonymous token endpoint.

    Public repos accept this without credentials. Private repos would need a
    PAT — not supported here since the installer only uses public images.
    """
    url = "https://ghcr.io/token"
    params = {"service": "ghcr.io", "scope": f"repository:{repo}:pull"}
    r = await client.get(url, params=params, timeout=15.0)
    r.raise_for_status()
    data = r.json()
    tok = data.get("token") or data.get("access_token") or ""
    if not tok:
        raise RuntimeError(f"no token from ghcr for {repo}")
    return tok


async def fetch_remote_digest(client: httpx.AsyncClient, image: str) -> str:
    """Resolve ``image`` to the current remote digest via GHCR's v2 API.

    The ``Docker-Content-Digest`` response header is the manifest digest,
    which matches what ``docker pull`` would record as ``RepoDigests[0]``.
    """
    repo, reference = _parse_image(image)
    token = await _ghcr_token(client, repo)
    url = f"https://ghcr.io/v2/{repo}/manifests/{reference}"
    headers = {
        "Authorization": f"Bearer {token}",
        # Accept OCI + legacy docker manifest formats. GHCR serves an OCI
        # index for multi-platform images; the digest header is identical
        # across types.
        "Accept": (
            "application/vnd.oci.image.index.v1+json,"
            "application/vnd.oci.image.manifest.v1+json,"
            "application/vnd.docker.distribution.manifest.list.v2+json,"
            "application/vnd.docker.distribution.manifest.v2+json"
        ),
    }
    # HEAD is enough — we only need the digest header.
    r = await client.head(url, headers=headers, timeout=15.0)
    # Some registries return 405 on HEAD for certain media types; fall back
    # to GET.
    if r.status_code == 405:
        r = await client.get(url, headers=headers, timeout=30.0)
    r.raise_for_status()
    digest = r.headers.get("Docker-Content-Digest") or r.headers.get("docker-content-digest")
    if not digest:
        raise RuntimeError(f"no Docker-Content-Digest header for {image}")
    return digest


def _docker_transport() -> httpx.AsyncHTTPTransport:
    """httpx transport that speaks HTTP over the docker UNIX socket."""
    return httpx.AsyncHTTPTransport(uds=DOCKER_SOCKET)


async def fetch_local_digest(client: httpx.AsyncClient, container: str) -> Optional[str]:
    """Return the ``RepoDigests[0]`` sha256 of the image currently running in
    ``container``, or ``None`` if the container or image cannot be inspected.

    Uses the docker daemon HTTP API over the mounted UNIX socket.
    """
    try:
        r = await client.get(f"http://localhost/containers/{container}/json", timeout=10.0)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        image_id = r.json().get("Image")
        if not image_id:
            return None
        ri = await client.get(f"http://localhost/images/{image_id}/json", timeout=10.0)
        if ri.status_code == 404:
            return None
        ri.raise_for_status()
        repo_digests = ri.json().get("RepoDigests") or []
        if not repo_digests:
            return None
        # RepoDigests entries look like "ghcr.io/foo/bar@sha256:...".
        first = repo_digests[0]
        if "@" in first:
            return first.split("@", 1)[1]
        return first
    except httpx.HTTPError as exc:
        logger.warning("local digest lookup failed for %s: %s", container, exc)
        return None


# ---------------------------------------------------------------------------
# State file
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to ``path`` via a temp-file-and-rename on the same dir."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".updates.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        # Best-effort cleanup; swallow secondary errors.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


async def run_check() -> dict[str, Any]:
    """One-shot: probe remote + local digests for core and admin, write state."""
    global _last_check, _last_error

    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    result: dict[str, Any] = {"checked_at": now}

    targets = [("core", CORE_IMAGE, CORE_CONTAINER), ("admin", ADMIN_IMAGE, ADMIN_CONTAINER)]

    async with httpx.AsyncClient() as registry_client, httpx.AsyncClient(
        transport=_docker_transport()
    ) as docker_client:
        for key, image, container in targets:
            entry: dict[str, Any] = {
                "image": image,
                "current": None,
                "latest": None,
                "update_available": False,
                "checked_at": now,
                "error": None,
            }
            try:
                entry["latest"] = await fetch_remote_digest(registry_client, image)
            except Exception as exc:
                entry["error"] = f"remote: {exc}"
                logger.warning("%s: remote digest failed: %s", key, exc)

            entry["current"] = await fetch_local_digest(docker_client, container)

            if entry["current"] and entry["latest"]:
                entry["update_available"] = entry["current"] != entry["latest"]

            result[key] = entry

    try:
        _atomic_write_json(STATE_PATH, result)
        _last_check = now
        _last_error = None
        logger.info(
            "check complete core=%s admin=%s",
            result.get("core", {}).get("update_available"),
            result.get("admin", {}).get("update_available"),
        )
    except Exception as exc:  # pragma: no cover — disk failure
        _last_error = str(exc)
        logger.error("failed to write state: %s", exc)

    return result


async def periodic_checker() -> None:
    """Background task loop. Runs once at startup, then every CHECK_INTERVAL."""
    # Slight initial delay so compose finishes wiring before the first check.
    await asyncio.sleep(5)
    while True:
        try:
            await run_check()
        except Exception as exc:  # pragma: no cover — keep the loop alive
            logger.exception("periodic check failed: %s", exc)
        await asyncio.sleep(CHECK_INTERVAL)


# ---------------------------------------------------------------------------
# HTTP handlers
# ---------------------------------------------------------------------------


def _auth(authorization: Optional[str] = Header(default=None)) -> None:
    """Bearer-token guard for /apply."""
    if not UPDATER_TOKEN:
        # Fail closed when the sidecar is deployed without a token — this
        # avoids a silent "anyone on the network can restart the stack"
        # footgun if install.sh skipped the token generation step.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="updater token not configured",
        )
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
        )
    provided = authorization.split(" ", 1)[1].strip()
    # Constant-time compare.
    import hmac

    if not hmac.compare_digest(provided, UPDATER_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid token",
        )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "last_check": _last_check,
        "last_error": _last_error,
        "check_interval": CHECK_INTERVAL,
    }


@app.post("/check")
async def manual_check(_: None = Depends(_auth)) -> dict[str, Any]:
    """Force a digest-check now without restarting anything."""
    data = await run_check()
    return {"ok": True, "data": data}


async def _run_compose(args: list[str]) -> tuple[int, str, str]:
    """Run ``docker compose <args>`` and capture stdout/stderr."""
    cmd = [
        "docker",
        "compose",
        "-f",
        COMPOSE_FILE,
        "--env-file",
        COMPOSE_ENV_FILE,
        *args,
    ]
    logger.info("run: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await proc.communicate()
    return proc.returncode or 0, stdout_b.decode(errors="replace"), stderr_b.decode(errors="replace")


@app.post("/apply")
async def apply_updates(_: None = Depends(_auth)) -> JSONResponse:
    """Pull + recreate core and admin containers.

    Runs ``docker compose pull`` then ``docker compose up -d`` scoped to the
    ``APPLY_SERVICES`` list. Returns the stdout/stderr of each step and
    re-runs the digest check so the state file reflects the new versions.
    """
    global _last_apply_at

    async with _apply_lock:
        now = asyncio.get_event_loop().time()
        if now - _last_apply_at < _APPLY_MIN_INTERVAL:
            retry_after = int(_APPLY_MIN_INTERVAL - (now - _last_apply_at)) + 1
            return JSONResponse(
                {"ok": False, "error": "rate limited", "retry_after": retry_after},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
        _last_apply_at = now

        services = list(APPLY_SERVICES)

        pull_rc, pull_out, pull_err = await _run_compose(["pull", *services])
        if pull_rc != 0:
            return JSONResponse(
                {
                    "ok": False,
                    "error": "docker compose pull failed",
                    "rc": pull_rc,
                    "stdout": pull_out,
                    "stderr": pull_err,
                },
                status_code=500,
            )

        up_rc, up_out, up_err = await _run_compose(["up", "-d", *services])
        if up_rc != 0:
            return JSONResponse(
                {
                    "ok": False,
                    "error": "docker compose up failed",
                    "rc": up_rc,
                    "stdout": up_out,
                    "stderr": up_err,
                },
                status_code=500,
            )

        # Refresh state so the next /api/system/updates read reflects the
        # new running digests.
        state_after: dict[str, Any] = {}
        try:
            state_after = await run_check()
        except Exception as exc:  # pragma: no cover
            logger.warning("post-apply check failed: %s", exc)

        return JSONResponse(
            {
                "ok": True,
                "restarted": services,
                "pull": {"stdout": pull_out, "stderr": pull_err},
                "up": {"stdout": up_out, "stderr": up_err},
                "state": state_after,
            }
        )


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def _startup() -> None:
    logger.info(
        "sygen-updater starting — interval=%ss core=%s admin=%s state=%s",
        CHECK_INTERVAL,
        CORE_IMAGE,
        ADMIN_IMAGE,
        STATE_PATH,
    )
    asyncio.create_task(periodic_checker())
