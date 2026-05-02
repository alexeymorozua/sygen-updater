"""Sygen updater (native install).

Polls GitHub Releases for new sygen / sygen-admin tags and exposes a
local HTTP endpoint that core's /apply flow calls to swap in a new
venv + admin tarball atomically.

Architecture:

1. Every ``CHECK_INTERVAL`` seconds, GET the GitHub Releases API for the
   ``alexeymorozua/sygen`` and ``alexeymorozua/sygen-admin`` repos, take
   the latest release's ``tag_name`` (e.g. ``v1.6.75``), strip the
   leading ``v`` to get the version. Compare to the version pinned in
   ``$SYGEN_HOME/.env`` (``SYGEN_CORE_VERSION`` /
   ``SYGEN_ADMIN_VERSION``). Write the result atomically to
   ``$SYGEN_HOME/host_updates/_updates.json`` — core reads it back via
   ``GET /api/system/updates``.

2. Expose ``GET /health`` (unauthenticated, used by smoke tests).

3. Expose ``POST /apply`` (bearer-auth via ``SYGEN_UPDATER_TOKEN``).
   When called:
     - download the new wheel from the GitHub Release
     - create a new venv at ``$VENV_NEW_DIR`` and ``pip install`` the wheel
     - mv-swap: ``venv → venv-prev``, ``venv-new → venv``
     - download the admin tarball, extract to ``$ADMIN_NEW_DIR``
     - mv-swap: ``admin → admin-prev``, ``admin-new → admin``
     - update ``.env`` with the new version pins
     - kick the core + admin services (launchctl on macOS, systemctl on
       Linux) so the new code is running

The updater itself is NOT swapped during an apply — that would kill the
in-flight request. ``sygen-updater`` ships from the same wheel as
``sygen-core``, so a manual ``pip install --upgrade sygen-updater`` is
the path to update the updater. (Or: a fresh re-run of install.sh.)
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hmac
import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("sygen-updater")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Config (resolved from .env at startup)
# ---------------------------------------------------------------------------


def _resolve_sygen_home() -> Path:
    h = os.environ.get("SYGEN_HOME") or os.environ.get("SYGEN_ROOT")
    if h:
        return Path(h)
    # Fallback heuristics for ad-hoc invocations.
    if platform.system() == "Darwin":
        return Path.home() / ".sygen-local"
    return Path("/srv/sygen")


SYGEN_HOME = _resolve_sygen_home()
ENV_FILE = SYGEN_HOME / ".env"
STATE_PATH = Path(
    os.environ.get("STATE_PATH", str(SYGEN_HOME / "host_updates" / "_updates.json"))
)
VENV_DIR = Path(os.environ.get("SYGEN_VENV_DIR", str(SYGEN_HOME / "venv")))
ADMIN_DIR = Path(os.environ.get("SYGEN_ADMIN_DIR", str(SYGEN_HOME / "admin")))
LISTEN_HOST = os.environ.get("SYGEN_UPDATER_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("SYGEN_UPDATER_PORT", "8082"))
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", "1800"))
APPLY_TIMEOUT = float(os.environ.get("SYGEN_APPLY_TIMEOUT", "600"))


def _load_env_file(path: Path) -> dict[str, str]:
    """Cheap .env reader. KEY=value lines, ignoring blanks + comments."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError as exc:
        logger.warning("read %s: %s", path, exc)
    return out


def _config() -> dict[str, str]:
    """Combine .env file values with process env (process env wins)."""
    cfg = _load_env_file(ENV_FILE)
    for k in (
        "SYGEN_CORE_VERSION",
        "SYGEN_ADMIN_VERSION",
        "SYGEN_UPDATER_TOKEN",
        "SYGEN_CORE_GITHUB_REPO",
        "SYGEN_ADMIN_GITHUB_REPO",
    ):
        v = os.environ.get(k)
        if v:
            cfg[k] = v
    return cfg


# Per-process apply lock + rate limit. The /apply call is destructive and
# expensive, so reject overlapping requests outright.
_APPLY_MIN_INTERVAL = 60.0
_apply_lock = asyncio.Lock()
_last_apply_at = 0.0
_last_check_at: Optional[str] = None
_last_check_error: Optional[str] = None


# ---------------------------------------------------------------------------
# GitHub Releases API
# ---------------------------------------------------------------------------


def _http_get_json(url: str, timeout: float = 15.0) -> Any:
    """Plain stdlib HTTP GET → JSON. No external deps in the venv."""
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "sygen-updater",
            # Inherit GH_TOKEN if set (rate-limit relief on private testing).
            **(
                {"Authorization": f"Bearer {os.environ['GH_TOKEN']}"}
                if os.environ.get("GH_TOKEN")
                else {}
            ),
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def fetch_latest_version(repo: str) -> Optional[str]:
    """Return the latest released tag of ``repo`` (e.g. "1.6.74"), or None.

    Strips the leading ``v`` so the returned string is comparable to the
    semver values pinned in .env.
    """
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    try:
        data = _http_get_json(url)
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError, OSError) as exc:
        logger.warning("GitHub Releases API error for %s: %s", repo, exc)
        return None
    tag = data.get("tag_name") if isinstance(data, dict) else None
    if not isinstance(tag, str):
        return None
    return tag.lstrip("v") or None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".updates.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


async def run_check() -> dict[str, Any]:
    """Probe latest GitHub Releases tags, write state file, return payload."""
    global _last_check_at, _last_check_error

    cfg = _config()
    core_repo = cfg.get("SYGEN_CORE_GITHUB_REPO", "alexeymorozua/sygen")
    admin_repo = cfg.get("SYGEN_ADMIN_GITHUB_REPO", "alexeymorozua/sygen-admin")
    core_current = cfg.get("SYGEN_CORE_VERSION", "")
    admin_current = cfg.get("SYGEN_ADMIN_VERSION", "")

    loop = asyncio.get_event_loop()
    core_latest, admin_latest = await asyncio.gather(
        loop.run_in_executor(None, fetch_latest_version, core_repo),
        loop.run_in_executor(None, fetch_latest_version, admin_repo),
    )

    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    payload = {
        "checked_at": now,
        "core": {
            "repo": core_repo,
            "current": core_current,
            "latest": core_latest,
            "version": core_current,
            "latest_version": core_latest,
            "update_available": bool(
                core_latest and core_current and core_latest != core_current
            ),
            "error": None if core_latest else "github releases api error",
        },
        "admin": {
            "repo": admin_repo,
            "current": admin_current,
            "latest": admin_latest,
            "version": admin_current,
            "latest_version": admin_latest,
            "update_available": bool(
                admin_latest and admin_current and admin_latest != admin_current
            ),
            "error": None if admin_latest else "github releases api error",
        },
    }
    try:
        _atomic_write_json(STATE_PATH, payload)
        _last_check_at = now
        _last_check_error = None
    except OSError as exc:
        _last_check_error = str(exc)
        logger.error("write %s: %s", STATE_PATH, exc)
    return payload


async def periodic_checker() -> None:
    await asyncio.sleep(5)
    while True:
        try:
            await run_check()
        except Exception as exc:  # pragma: no cover
            logger.exception("periodic check failed: %s", exc)
        await asyncio.sleep(CHECK_INTERVAL)


# ---------------------------------------------------------------------------
# Apply: atomic venv + admin swap
# ---------------------------------------------------------------------------


def _release_asset_url(repo: str, version: str, filename: str) -> str:
    return f"https://github.com/{repo}/releases/download/v{version}/{filename}"


def _download(url: str, dest: Path, timeout: float = 600.0) -> None:
    """Stream a release asset to ``dest`` via stdlib urllib (handles 302)."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "sygen-updater",
            **(
                {"Authorization": f"Bearer {os.environ['GH_TOKEN']}"}
                if os.environ.get("GH_TOKEN")
                else {}
            ),
        },
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)


def _python_for_new_venv() -> str:
    """Pick the Python to seed a new venv. Default: same interpreter we are
    running inside (so the new venv matches the runtime)."""
    return os.environ.get("SYGEN_PYTHON_BIN", shutil.which("python3") or "python3")


def _restart_service(label: str) -> tuple[int, str]:
    """Trigger a service restart via the appropriate platform tool.

    macOS: ``launchctl kickstart -k gui/<uid>/<label>`` (where label is
    e.g. ``pro.sygen.core``).
    Linux: ``systemctl restart <unit>`` (where unit is e.g. ``sygen-core``).
    """
    if platform.system() == "Darwin":
        target = f"gui/{os.getuid()}/{label}"
        cmd = ["launchctl", "kickstart", "-k", target]
    else:
        cmd = ["systemctl", "restart", label]
    logger.info("restart: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return proc.returncode, (proc.stderr or proc.stdout).strip()


def _update_env_pin(key: str, value: str) -> None:
    """Replace KEY= line in $SYGEN_HOME/.env (or append if missing).

    Atomic write via tmp + os.replace. .env stays 0600.
    """
    if not ENV_FILE.exists():
        return
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    found = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")
    tmp = ENV_FILE.with_suffix(".env.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, ENV_FILE)


def _apply_core(version: str, repo: str) -> None:
    """Download wheel, install in fresh venv, mv-swap."""
    wheel_name = f"sygen-{version}-py3-none-any.whl"
    wheel_url = _release_asset_url(repo, version, wheel_name)
    wheel_path = Path(tempfile.mkdtemp(prefix="sygen-wheel-")) / wheel_name
    logger.info("downloading core wheel %s", wheel_url)
    _download(wheel_url, wheel_path)

    # Build the new venv next to the live one. --clear ensures a clean
    # site-packages — we never want to inherit stale packages from a
    # prior aborted apply.
    venv_new = VENV_DIR.with_name("venv-new")
    venv_prev = VENV_DIR.with_name("venv-prev")
    if venv_new.exists():
        shutil.rmtree(venv_new)
    logger.info("creating new venv at %s", venv_new)
    subprocess.run(
        [_python_for_new_venv(), "-m", "venv", "--clear", str(venv_new)],
        check=True,
        timeout=120,
    )
    pip_new = venv_new / "bin" / "pip"
    subprocess.run(
        [str(pip_new), "install", "--quiet", "--upgrade", "pip", "wheel"],
        check=True,
        timeout=120,
    )
    logger.info("pip install %s", wheel_path)
    subprocess.run(
        [str(pip_new), "install", "--quiet", str(wheel_path)], check=True, timeout=300
    )
    # The updater itself runs from the LIVE venv; swapping it out from
    # under itself would kill us mid-flight. Install sygen-updater into
    # the new venv anyway so the next service restart picks it up.
    updater_wheel_url = _release_asset_url(
        repo, version, f"sygen_updater-{version}-py3-none-any.whl"
    )
    updater_wheel_path = wheel_path.with_name(f"sygen_updater-{version}-py3-none-any.whl")
    try:
        _download(updater_wheel_url, updater_wheel_path)
        subprocess.run(
            [str(pip_new), "install", "--quiet", str(updater_wheel_path)],
            check=True,
            timeout=300,
        )
    except Exception as exc:
        logger.warning("could not install sygen-updater into new venv: %s", exc)

    # Atomic mv-swap.
    if venv_prev.exists():
        shutil.rmtree(venv_prev)
    if VENV_DIR.exists():
        VENV_DIR.rename(venv_prev)
    venv_new.rename(VENV_DIR)
    shutil.rmtree(wheel_path.parent, ignore_errors=True)


def _apply_admin(version: str, repo: str) -> None:
    """Download admin tarball, extract, mv-swap."""
    tarball_name = f"sygen-admin-{version}.tar.gz"
    tarball_url = _release_asset_url(repo, version, tarball_name)
    tmp_root = Path(tempfile.mkdtemp(prefix="sygen-admin-"))
    tarball_path = tmp_root / tarball_name
    logger.info("downloading admin tarball %s", tarball_url)
    _download(tarball_url, tarball_path)

    staging = ADMIN_DIR.with_name("admin-new")
    admin_prev = ADMIN_DIR.with_name("admin-prev")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["tar", "-xzf", str(tarball_path), "-C", str(staging)],
        check=True,
        timeout=300,
    )
    if not (staging / "server.js").exists():
        raise RuntimeError("admin tarball missing server.js — refusing to swap in")

    if admin_prev.exists():
        shutil.rmtree(admin_prev)
    if ADMIN_DIR.exists():
        ADMIN_DIR.rename(admin_prev)
    staging.rename(ADMIN_DIR)
    shutil.rmtree(tmp_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# HTTP server (stdlib aiohttp-free — keeps the venv tiny)
# ---------------------------------------------------------------------------


def _auth_ok(headers: dict[str, str]) -> bool:
    cfg = _config()
    token = cfg.get("SYGEN_UPDATER_TOKEN", "")
    if not token:
        return False
    auth = headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return False
    provided = auth[7:].strip()
    return hmac.compare_digest(provided, token)


async def _read_request(reader: asyncio.StreamReader) -> tuple[str, str, dict[str, str], bytes]:
    """Tiny HTTP/1.1 request parser. Reads request line + headers + body."""
    request_line = (await reader.readline()).decode("ascii", errors="replace").rstrip("\r\n")
    parts = request_line.split(" ")
    if len(parts) < 3:
        raise ValueError("bad request line")
    method, path, _ = parts[0], parts[1], parts[2]
    headers: dict[str, str] = {}
    while True:
        line = (await reader.readline()).decode("ascii", errors="replace").rstrip("\r\n")
        if not line:
            break
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    length = int(headers.get("content-length", "0") or "0")
    body = await reader.readexactly(length) if length > 0 else b""
    return method, path, headers, body


def _http_response(
    status_code: int,
    body: dict[str, Any] | str,
    *,
    headers: Optional[dict[str, str]] = None,
) -> bytes:
    if isinstance(body, dict):
        payload = json.dumps(body).encode("utf-8")
        ctype = "application/json"
    else:
        payload = body.encode("utf-8")
        ctype = "text/plain"
    reason = {200: "OK", 401: "Unauthorized", 429: "Too Many Requests", 500: "Internal Server Error", 503: "Service Unavailable"}.get(
        status_code, "OK"
    )
    hdrs = {
        "Content-Type": ctype,
        "Content-Length": str(len(payload)),
        "Connection": "close",
    }
    if headers:
        hdrs.update(headers)
    head = f"HTTP/1.1 {status_code} {reason}\r\n"
    head += "".join(f"{k}: {v}\r\n" for k, v in hdrs.items())
    head += "\r\n"
    return head.encode("ascii") + payload


async def _handle_health() -> bytes:
    return _http_response(
        200,
        {
            "ok": True,
            "last_check": _last_check_at,
            "last_error": _last_check_error,
            "check_interval": CHECK_INTERVAL,
            "venv_dir": str(VENV_DIR),
            "admin_dir": str(ADMIN_DIR),
        },
    )


async def _handle_check(headers: dict[str, str]) -> bytes:
    if not _auth_ok(headers):
        return _http_response(401, {"ok": False, "error": "missing or invalid bearer token"})
    data = await run_check()
    return _http_response(200, {"ok": True, "data": data})


async def _handle_apply(headers: dict[str, str]) -> bytes:
    global _last_apply_at

    if not _auth_ok(headers):
        return _http_response(401, {"ok": False, "error": "missing or invalid bearer token"})
    cfg = _config()
    if not cfg.get("SYGEN_UPDATER_TOKEN"):
        return _http_response(503, {"ok": False, "error": "updater token not configured"})

    async with _apply_lock:
        loop = asyncio.get_event_loop()
        now = loop.time()
        if now - _last_apply_at < _APPLY_MIN_INTERVAL:
            retry_after = int(_APPLY_MIN_INTERVAL - (now - _last_apply_at)) + 1
            return _http_response(
                429,
                {"ok": False, "error": "rate limited", "retry_after": retry_after},
                headers={"Retry-After": str(retry_after)},
            )
        _last_apply_at = now

        # Fetch latest release versions for both core and admin.
        core_repo = cfg.get("SYGEN_CORE_GITHUB_REPO", "alexeymorozua/sygen")
        admin_repo = cfg.get("SYGEN_ADMIN_GITHUB_REPO", "alexeymorozua/sygen-admin")
        core_latest = await loop.run_in_executor(None, fetch_latest_version, core_repo)
        admin_latest = await loop.run_in_executor(None, fetch_latest_version, admin_repo)
        if not core_latest or not admin_latest:
            return _http_response(
                500,
                {
                    "ok": False,
                    "error": "could not resolve latest versions from GitHub Releases",
                    "core_latest": core_latest,
                    "admin_latest": admin_latest,
                },
            )

        core_current = cfg.get("SYGEN_CORE_VERSION", "")
        admin_current = cfg.get("SYGEN_ADMIN_VERSION", "")
        applied: list[str] = []

        try:
            if core_latest != core_current:
                logger.info("applying core %s → %s", core_current, core_latest)
                await asyncio.wait_for(
                    loop.run_in_executor(None, _apply_core, core_latest, core_repo),
                    timeout=APPLY_TIMEOUT,
                )
                _update_env_pin("SYGEN_CORE_VERSION", core_latest)
                applied.append(f"core {core_current} → {core_latest}")

            if admin_latest != admin_current:
                logger.info("applying admin %s → %s", admin_current, admin_latest)
                await asyncio.wait_for(
                    loop.run_in_executor(None, _apply_admin, admin_latest, admin_repo),
                    timeout=APPLY_TIMEOUT,
                )
                _update_env_pin("SYGEN_ADMIN_VERSION", admin_latest)
                applied.append(f"admin {admin_current} → {admin_latest}")
        except asyncio.TimeoutError:
            return _http_response(
                504,
                {"ok": False, "error": "apply timed out", "timeout_seconds": APPLY_TIMEOUT, "applied": applied},
            )
        except (subprocess.CalledProcessError, urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            logger.exception("apply failed")
            return _http_response(
                500,
                {"ok": False, "error": f"apply failed: {exc}", "applied": applied},
            )

        # Restart core + admin so the swap takes effect. Updater stays
        # alive (we're inside it).
        restarted: list[dict[str, Any]] = []
        for label, unit in (
            ("pro.sygen.core", "sygen-core"),
            ("pro.sygen.admin", "sygen-admin"),
        ):
            try:
                rc, msg = _restart_service(
                    label if platform.system() == "Darwin" else unit
                )
                restarted.append({"target": label if platform.system() == "Darwin" else unit, "rc": rc, "msg": msg})
            except (subprocess.SubprocessError, OSError) as exc:
                restarted.append({"target": label, "rc": -1, "msg": str(exc)})

        # Refresh state so the next /api/system/updates read reflects new pins.
        state_after = await run_check()
        return _http_response(
            200,
            {"ok": True, "applied": applied, "restarted": restarted, "state": state_after},
        )


async def _serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        method, path, headers, _body = await _read_request(reader)
    except (ValueError, asyncio.IncompleteReadError, ConnectionError):
        writer.close()
        return
    try:
        path = path.split("?", 1)[0]
        if method == "GET" and path == "/health":
            resp = await _handle_health()
        elif method == "POST" and path == "/check":
            resp = await _handle_check(headers)
        elif method == "POST" and path == "/apply":
            resp = await _handle_apply(headers)
        else:
            resp = _http_response(404, {"ok": False, "error": "not found"})
        writer.write(resp)
        await writer.drain()
    except Exception as exc:  # pragma: no cover
        logger.exception("handler error: %s", exc)
        try:
            writer.write(_http_response(500, {"ok": False, "error": "internal error"}))
            await writer.drain()
        except ConnectionError:
            pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def main() -> None:
    logger.info(
        "sygen-updater starting — listen=%s:%s home=%s state=%s interval=%ss",
        LISTEN_HOST,
        LISTEN_PORT,
        SYGEN_HOME,
        STATE_PATH,
        CHECK_INTERVAL,
    )
    asyncio.create_task(periodic_checker())
    server = await asyncio.start_server(_serve, LISTEN_HOST, LISTEN_PORT)
    async with server:
        await server.serve_forever()


def entrypoint() -> None:
    """Console script entry point — referenced by pyproject.toml."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    entrypoint()
