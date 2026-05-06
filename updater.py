"""Sygen updater (native install).

Polls GitHub Releases for new sygen / sygen-admin tags and exposes a
local HTTP endpoint that core's /apply flow calls to swap in a new
venv + admin tarball atomically.

Architecture:

1. Every ``CHECK_INTERVAL`` seconds, query the GitHub Releases API and
   resolve the latest available core and admin versions:

   * **Mirror mode (default)** — query ``repos/<releases_repo>/releases``
     once, filter by ``tag_name`` prefix (``core-`` for sygen+sygen-updater
     wheels, ``admin-`` for the sygen-admin tarball), pick the highest
     numeric semver. ``releases_repo`` defaults to
     ``alexeymorozua/sygen-releases``: a public repo that auto-mirrors
     release artefacts from the private source repos so anonymous curl
     can reach them. Override via ``SYGEN_RELEASES_GITHUB_REPO``.
   * **Legacy mode** — set ``SYGEN_RELEASES_GITHUB_REPO=`` (empty) to
     fall back to ``repos/<source>/releases/latest`` against the private
     source repos with ``v<version>`` tags. Only useful when running
     against a fork that hasn't adopted the public mirror layout.

   The result is compared against the versions pinned in
   ``$SYGEN_HOME/.env`` (``SYGEN_CORE_VERSION`` / ``SYGEN_ADMIN_VERSION``)
   and written atomically to ``$SYGEN_HOME/host_updates/_updates.json``
   — core reads it back via ``GET /api/system/updates``.

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
import hashlib
import hmac
import json
import logging
import os
import platform
import re
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
# install.sh writes .install_manifest.json into SYGEN_ROOT (one level above
# SYGEN_HOME on real installs: /srv/sygen/.install_manifest.json with
# SYGEN_HOME=/srv/sygen/data). Fall back to SYGEN_HOME.parent when SYGEN_ROOT
# is unset — wrong on macOS dev (~/.sygen-local), but the manifest doesn't
# exist there so _update_install_manifest silent no-ops.
SYGEN_ROOT = Path(os.environ.get("SYGEN_ROOT") or str(SYGEN_HOME.parent))
# install.sh writes .env into SYGEN_ROOT (one level above SYGEN_HOME on real
# installs: /srv/sygen/.env with SYGEN_HOME=/srv/sygen/data). Same parent as
# the install manifest — keeping them co-located.
ENV_FILE = SYGEN_ROOT / ".env"
INSTALL_MANIFEST = SYGEN_ROOT / ".install_manifest.json"
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
        "SYGEN_RELEASES_GITHUB_REPO",
        # P0-3: pinned by install.sh so the updater seeds new venvs with
        # the same brew/apt python the live venv was built with.
        "SYGEN_PYTHON_BIN",
    ):
        v = os.environ.get(k)
        if v is not None:
            # Empty string is a meaningful override (opt-in to legacy mode
            # for SYGEN_RELEASES_GITHUB_REPO), so we copy it verbatim.
            cfg[k] = v
    return cfg


_DEFAULT_RELEASES_REPO = "alexeymorozua/sygen-releases"
_DEFAULT_CORE_REPO = "alexeymorozua/sygen"
_DEFAULT_ADMIN_REPO = "alexeymorozua/sygen-admin"


def _releases_repo(cfg: dict[str, str]) -> str:
    """Return the public mirror repo to poll, or empty string for legacy mode.

    The key may be absent (→ default mirror) or explicitly empty (→ legacy
    v-tag mode against the private source repos).
    """
    if "SYGEN_RELEASES_GITHUB_REPO" not in cfg:
        return _DEFAULT_RELEASES_REPO
    return cfg["SYGEN_RELEASES_GITHUB_REPO"].strip()


def _legacy_source_repo(cfg: dict[str, str], component: str) -> str:
    if component == "core":
        return cfg.get("SYGEN_CORE_GITHUB_REPO", _DEFAULT_CORE_REPO)
    return cfg.get("SYGEN_ADMIN_GITHUB_REPO", _DEFAULT_ADMIN_REPO)


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
    """Legacy mode: latest released tag of ``repo`` (e.g. "1.6.74") or None.

    Strips the leading ``v`` so the returned string is comparable to the
    semver values pinned in .env. Used only when
    ``SYGEN_RELEASES_GITHUB_REPO`` is explicitly set to an empty string.
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


def _parse_semver(s: str) -> Optional[tuple[int, ...]]:
    """Strict numeric semver: ``X.Y.Z`` (or longer) of integers only.

    Returns None for pre-release tags like ``1.6.75-rc1`` so we never auto-
    apply them — the install.sh / mirror tagging convention treats those as
    out of band, not the floor of "latest".
    """
    if not s:
        return None
    parts = s.split(".")
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return None


def fetch_latest_from_mirror(releases_repo: str, prefix: str) -> Optional[str]:
    """Mirror mode: list releases of ``releases_repo``, filter by tag prefix
    (``core-`` or ``admin-``), return the highest numeric semver.

    Skips draft and pre-release entries. Pulls a single page of 100 — the
    mirror auto-prunes old artefacts so the latest is always near the top.
    """
    url = f"https://api.github.com/repos/{releases_repo}/releases?per_page=100"
    try:
        data = _http_get_json(url)
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError, OSError) as exc:
        logger.warning("GitHub Releases API error for %s: %s", releases_repo, exc)
        return None
    if not isinstance(data, list):
        return None
    best: Optional[tuple[tuple[int, ...], str]] = None
    for rel in data:
        if not isinstance(rel, dict):
            continue
        if rel.get("draft") or rel.get("prerelease"):
            continue
        tag = rel.get("tag_name")
        if not isinstance(tag, str) or not tag.startswith(prefix):
            continue
        version_str = tag[len(prefix):]
        version = _parse_semver(version_str)
        if version is None:
            continue
        if best is None or version > best[0]:
            best = (version, version_str)
    return best[1] if best else None


def fetch_latest_for(cfg: dict[str, str], component: str) -> Optional[str]:
    """Resolve the latest available version of ``component`` (``core`` or
    ``admin``) honoring mirror vs legacy mode.
    """
    releases_repo = _releases_repo(cfg)
    if releases_repo:
        return fetch_latest_from_mirror(releases_repo, f"{component}-")
    return fetch_latest_version(_legacy_source_repo(cfg, component))


def _payload_repo(cfg: dict[str, str], component: str) -> str:
    """Repo string surfaced to the admin UI for the "where did this version
    come from" link. Mirror in mirror mode, source in legacy mode.
    """
    releases_repo = _releases_repo(cfg)
    return releases_repo if releases_repo else _legacy_source_repo(cfg, component)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".updates.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        # P1-7: tempfile.mkstemp creates 0600; loosen to 0644 so other
        # tooling under host_updates/ can read the state file.
        os.chmod(tmp, 0o644)
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
    core_current = cfg.get("SYGEN_CORE_VERSION", "")
    admin_current = cfg.get("SYGEN_ADMIN_VERSION", "")

    loop = asyncio.get_event_loop()
    core_latest, admin_latest = await asyncio.gather(
        loop.run_in_executor(None, fetch_latest_for, cfg, "core"),
        loop.run_in_executor(None, fetch_latest_for, cfg, "admin"),
    )

    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    payload = {
        "checked_at": now,
        "core": {
            "repo": _payload_repo(cfg, "core"),
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
            "repo": _payload_repo(cfg, "admin"),
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

    existing: dict[str, Any] = {}
    if STATE_PATH.exists():
        try:
            loaded = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except (json.JSONDecodeError, OSError):
            existing = {}
    merged = {**existing, **payload}

    try:
        _atomic_write_json(STATE_PATH, merged)
        _last_check_at = now
        _last_check_error = None
    except OSError as exc:
        _last_check_error = str(exc)
        logger.error("write %s: %s", STATE_PATH, exc)
    return merged


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


def _release_asset_url(component: str, version: str, filename: str, cfg: Optional[dict[str, str]] = None) -> str:
    """Resolve the release asset URL.

    Mirror mode (default): ``<releases_repo>/releases/download/<component>-<version>/<filename>``
    Legacy mode: ``<source_repo>/releases/download/v<version>/<filename>``
    """
    cfg = cfg if cfg is not None else _config()
    releases_repo = _releases_repo(cfg)
    if releases_repo:
        return f"https://github.com/{releases_repo}/releases/download/{component}-{version}/{filename}"
    legacy_repo = _legacy_source_repo(cfg, component)
    return f"https://github.com/{legacy_repo}/releases/download/v{version}/{filename}"


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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_verified(asset_url: str, dest: Path, timeout: float = 600.0) -> None:
    """P0-2: download a release asset *and* verify its SHA256 against the
    matching ``<asset>.sha256`` sidecar.

    Sidecar format (matches sha256sum -c): ``<hex>  <basename>``. Missing
    sidecar = fail closed (raises RuntimeError). The release workflows in
    sygen + sygen-admin emit these sidecars; absence means the release
    was published incorrectly and we refuse to install.
    """
    _download(asset_url, dest, timeout=timeout)
    sha_url = asset_url + ".sha256"
    sha_dest = dest.with_suffix(dest.suffix + ".sha256")
    try:
        _download(sha_url, sha_dest, timeout=60.0)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        try:
            dest.unlink(missing_ok=True)
            sha_dest.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(
            f"missing checksum sidecar at {sha_url} — refusing to install unverified asset ({exc})"
        ) from exc
    expected = sha_dest.read_text(encoding="utf-8").split()[0].strip().lower()
    sha_dest.unlink(missing_ok=True)
    actual = _sha256_file(dest).lower()
    if not expected or expected != actual:
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(
            f"SHA256 mismatch for {asset_url} — expected {expected or '<unreadable>'}, got {actual}"
        )


def _python_for_new_venv() -> str:
    """Pick the Python to seed a new venv.

    Priority: ``SYGEN_PYTHON_BIN`` from .env / process env (pinned to the
    brew/apt python that install.sh built the live venv with) →
    ``shutil.which("python3")`` (fallback only). The pin is load-bearing
    on macOS, where ``shutil.which("python3")`` resolves to
    ``/usr/bin/python3`` (Apple stub, often 3.9) instead of the brew
    python@3.14 we want.
    """
    pinned = _config().get("SYGEN_PYTHON_BIN")
    if pinned:
        return pinned
    return shutil.which("python3") or "python3"


def _macos_plist_path(label: str) -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"


def _stop_service(label: str) -> None:
    """P1-6: fully stop the service so the mv-swap window can't be
    raced by launchd's KeepAlive (or systemd's Restart=). Best-effort
    — a missing/inactive service is fine.
    """
    if platform.system() == "Darwin":
        plist = _macos_plist_path(label)
        if plist.exists():
            subprocess.run(
                ["launchctl", "unload", str(plist)],
                capture_output=True,
                text=True,
                timeout=30,
            )
    else:
        subprocess.run(
            ["systemctl", "stop", label],
            capture_output=True,
            text=True,
            timeout=30,
        )


def _start_service(label: str) -> tuple[int, str]:
    """P1-6: start (or load) the service after the mv-swap. Pairs with
    ``_stop_service``. On macOS this is ``launchctl load -w`` (which
    also re-arms KeepAlive); on Linux it's ``systemctl start``.
    """
    if platform.system() == "Darwin":
        plist = _macos_plist_path(label)
        if not plist.exists():
            return 1, f"plist not found: {plist}"
        cmd = ["launchctl", "load", "-w", str(plist)]
    else:
        cmd = ["systemctl", "start", label]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return proc.returncode, (proc.stderr or proc.stdout).strip()


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


def _poll_health(url: str, timeout_seconds: float = 30.0) -> tuple[bool, Optional[int], Optional[str]]:
    """P1-9: poll an HTTP endpoint until it answers (any non-5xx) or
    ``timeout_seconds`` elapses. Returns ``(healthy, last_code, last_err)``.

    "Any non-5xx" matches the install.sh smoke-test rule: 200/301/302/401/
    403/404 all prove the server is alive and routing — only connect
    failures and 5xx are treated as unhealthy.
    """
    import socket
    import time

    deadline = time.monotonic() + timeout_seconds
    last_code: Optional[int] = None
    last_err: Optional[str] = None
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sygen-updater/health"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                last_code = resp.status
                if 200 <= resp.status < 500:
                    return True, last_code, None
        except urllib.error.HTTPError as exc:
            last_code = exc.code
            if 200 <= exc.code < 500:
                return True, last_code, None
            last_err = f"HTTP {exc.code}"
        except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as exc:
            last_err = str(exc)
        time.sleep(1.0)
    return False, last_code, last_err


def _update_env_pin(key: str, value: str) -> None:
    """Replace KEY= line in $SYGEN_ROOT/.env (or append if missing).

    Atomic write via tmp + os.replace. .env stays 0600.

    Also updates ``os.environ[key]``: ``_config()`` prefers process env over
    .env file values, so without this the long-running updater process
    keeps returning the pre-apply version pin to ``run_check()``, leaving
    state.json stuck on the old version even after a successful swap.
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
    os.environ[key] = value


def _update_install_manifest(key: str, value: str) -> None:
    """Update a single field in $SYGEN_ROOT/.install_manifest.json.

    Atomic write via tmp + os.replace. Silent no-op if the manifest is
    missing (legacy installs predating the manifest), unreadable, or
    not a JSON object — we never create it from scratch since install.sh
    is the source of truth for the rest of its fields.
    """
    if not INSTALL_MANIFEST.exists():
        return
    try:
        data = json.loads(INSTALL_MANIFEST.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(data, dict):
        return
    if data.get(key) == value:
        return
    data[key] = value
    tmp = INSTALL_MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, INSTALL_MANIFEST)


def _record_apply_npm_results(results: dict[str, Any]) -> None:
    """Merge ``last_apply_npm_results`` into the existing state file.

    State file already carries the periodic-check payload; we tack the
    apply-time npm summary on as a sibling field so /api/system/updates
    can surface it. Silent no-op if the file is missing or unreadable.
    """
    if not STATE_PATH.exists():
        return
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(data, dict):
        return
    data["last_apply_npm_results"] = results
    try:
        _atomic_write_json(STATE_PATH, data)
    except OSError as exc:
        logger.warning("could not record apply npm results: %s", exc)


# Map npm package name → CLI binary name. `version.py` in core uses
# this binary name to key the cli_tools cache. After a successful
# `npm update -g <pkg>`, we run `<binary> --version` and surface the
# new version under `binary_versions` so the core API handler can
# override its (stale) cli_tools cache without waiting for the next
# weekly probe.
_NPM_PACKAGE_TO_BINARY: dict[str, str] = {
    "@anthropic-ai/claude-code": "claude",
}


def _read_binary_version(binary: str) -> str | None:
    """Return ``<binary> --version`` first whitespace-split token, or None."""
    bin_path = shutil.which(binary)
    if not bin_path:
        return None
    try:
        proc = subprocess.run(
            [bin_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    raw = (proc.stdout or "").strip()
    if not raw:
        return None
    return raw.split()[0] or None


def _post_apply_update_npm_packages(force_npm_preexisting: bool = False) -> dict[str, Any]:
    """Refresh globally-installed npm CLIs sygen tracks.

    Apply Update only swaps the venv + admin tarball. CLI tools that
    install.sh dropped via ``npm install -g`` (today: only
    ``@anthropic-ai/claude-code``) get frozen at install time unless we
    actively refresh them here.

    1.6.116+: every package the manifest records — both
    ``installed_npm`` (sygen-provisioned) and ``preexisting_npm``
    (user-owned, recorded by install.sh) — gets refreshed on every
    Apply. Sygen treats every CLI it knows about the same way; the
    earlier installed-vs-preexisting distinction is gone.

    ``force_npm_preexisting`` is a deprecated noop kept for backwards
    compatibility with 1.6.115 callers. Both ``True`` and ``False``
    produce identical behaviour now (both buckets refreshed). The
    parameter is preserved so nothing in the call chain has to grow a
    version-aware adapter.

    Errors per-package are logged + recorded; they never abort the apply
    flow — we already have the new venv mv-swapped in by the time this
    runs.
    """
    del force_npm_preexisting  # 1.6.116: deprecated noop, kept for compat.
    results: dict[str, Any] = {
        "updated": [],
        "skipped": [],
        "errors": [],
        "binary_versions": {},
        "checked_at": dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    }
    if not INSTALL_MANIFEST.exists():
        return results
    try:
        manifest = json.loads(INSTALL_MANIFEST.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("post-apply npm: cannot read manifest: %s", exc)
        return results
    if not isinstance(manifest, dict):
        return results

    installed = manifest.get("installed_npm") or []
    installed = [p for p in installed if isinstance(p, str) and p]
    raw_pre = manifest.get("preexisting_npm") or []
    preexisting = [p for p in raw_pre if isinstance(p, str) and p]
    # Avoid touching the same package twice if it ended up in both
    # buckets (shouldn't happen, but be defensive — duplicates would
    # also double-count in updated[]/binary_versions).
    preexisting = [p for p in preexisting if p not in installed]

    pkgs: list[tuple[str, bool]] = [(p, False) for p in installed] + [
        (p, True) for p in preexisting
    ]
    if not pkgs:
        return results

    npm_bin = shutil.which("npm")
    if not npm_bin:
        for pkg, _is_pre in pkgs:
            results["errors"].append({"pkg": pkg, "error": "npm not found in PATH"})
        return results

    for pkg, is_preexisting in pkgs:
        try:
            proc = subprocess.run(
                [npm_bin, "update", "-g", pkg],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            logger.warning("post-apply npm update -g %s timed out", pkg)
            err: dict[str, Any] = {"pkg": pkg, "error": f"timeout: {exc}"}
            if is_preexisting:
                err["preexisting"] = True
            results["errors"].append(err)
            continue
        except OSError as exc:
            logger.warning("post-apply npm update -g %s failed: %s", pkg, exc)
            err = {"pkg": pkg, "error": str(exc)}
            if is_preexisting:
                err["preexisting"] = True
            results["errors"].append(err)
            continue
        if proc.returncode == 0:
            logger.info(
                "post-apply npm: refreshed %s%s",
                pkg,
                " (preexisting opt-in)" if is_preexisting else "",
            )
            results["updated"].append(pkg)
            binary = _NPM_PACKAGE_TO_BINARY.get(pkg)
            if binary:
                bin_version = _read_binary_version(binary)
                if bin_version:
                    results["binary_versions"][binary] = bin_version
        else:
            stderr = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:]
            msg = stderr[0] if stderr else f"exit {proc.returncode}"
            logger.warning("post-apply npm update -g %s failed: %s", pkg, msg)
            err = {"pkg": pkg, "error": msg}
            if is_preexisting:
                err["preexisting"] = True
            results["errors"].append(err)

    return results


def _post_apply_warmup_embeddings() -> None:
    """Warm the fastembed model cache in the new venv.

    After a venv swap the new ``site-packages`` may have a freshly-installed
    fastembed/onnxruntime that has never imported the model in this process
    image. Loading it eagerly here means the first user message after Apply
    doesn't pay the ~5–30 s ONNX cold load (the model cache itself, ~220 MB
    on disk, persists across venv swaps so the network roundtrip stays one-
    time at install). Failure is non-fatal — the bot's own ``_warmup_embeddings``
    background task will retry on the next service restart.
    """
    py = VENV_DIR / "bin" / "python"
    if not py.exists():
        return
    cache_dir = SYGEN_HOME / "embeddings" / "model_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("post-apply warmup: cannot mkdir %s: %s", cache_dir, exc)
        return
    code = (
        "import sys\n"
        "from fastembed import TextEmbedding\n"
        "list(TextEmbedding(model_name=sys.argv[1], cache_dir=sys.argv[2])"
        ".embed(['warmup']))\n"
    )
    try:
        proc = subprocess.run(
            [
                str(py), "-c", code,
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                str(cache_dir),
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("post-apply embedding warmup: subprocess failed: %s", exc)
        return
    if proc.returncode == 0:
        logger.info("post-apply embedding warmup: cache ready at %s", cache_dir)
    else:
        stderr = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:]
        msg = stderr[0] if stderr else f"exit {proc.returncode}"
        logger.warning("post-apply embedding warmup failed: %s", msg)


_PYTHON_SHEBANG_RE = re.compile(rb"^#!(\S+/python\d*(?:\.\d+)?)")


def _rewrite_venv_shebangs(venv_dir: Path) -> None:
    """Rewrite stale build-path shebangs in ``venv/bin/*`` and ``pyvenv.cfg``.

    Python's stdlib ``venv`` module bakes the absolute interpreter path into
    every generated entry-point script (``#!.../bin/python3``) and into
    ``pyvenv.cfg``'s ``command =`` line. After we rename ``venv-new`` →
    ``venv`` to atomically swap in the new release, those paths still point
    at the build-time location (``.../venv-new/bin/python3``) which no
    longer exists — so ``ExecStart=/srv/sygen/venv/bin/sygen`` fails with
    ``status=203/EXEC`` and the service flaps. This walks the new ``bin/``
    directory, rewrites any python shebang to use ``venv_dir/bin/<basename>``,
    and patches sibling ``venv-*`` paths in ``pyvenv.cfg`` the same way.
    """
    # Resolve to an absolute path: shebangs must be absolute to be portable
    # across the cwd of whichever process exec()s the script (systemd /
    # launchd both spawn with cwd=/).
    venv_dir = venv_dir.resolve() if not venv_dir.is_absolute() else venv_dir
    bin_dir = venv_dir / "bin"
    if bin_dir.is_dir():
        for f in bin_dir.iterdir():
            if not f.is_file():
                continue
            try:
                content = f.read_bytes()
            except OSError:
                continue
            if not content.startswith(b"#!"):
                continue  # binary (e.g. python interpreter itself) or non-script
            first_line_end = content.find(b"\n")
            shebang = content if first_line_end == -1 else content[:first_line_end]
            m = _PYTHON_SHEBANG_RE.match(shebang)
            if not m:
                continue  # non-python shebang (e.g. #!/bin/sh activator) — leave alone
            old_path = m.group(1).decode()
            new_path = str(venv_dir / "bin" / Path(old_path).name)
            if old_path == new_path:
                continue
            # Preserve any trailing flags after the interpreter path
            # (e.g. ``#!/path/python3 -sE``).
            new_shebang = b"#!" + new_path.encode() + shebang[m.end():]
            rest = content[len(shebang):] if first_line_end != -1 else b""
            mode = f.stat().st_mode
            f.write_bytes(new_shebang + rest)
            f.chmod(mode)  # write_bytes preserves mode, but be explicit

    cfg = venv_dir / "pyvenv.cfg"
    if cfg.is_file():
        try:
            text = cfg.read_text(encoding="utf-8")
        except OSError:
            return
        # Replace any sibling ``<parent>/venv-<suffix>`` reference (the
        # build location pip recorded in command=) with the final venv_dir.
        # System interpreter paths like /opt/homebrew/opt/python@3.14/bin
        # don't match this anchor and stay untouched.
        parent_str = str(venv_dir.parent)
        new_text = re.sub(
            rf"{re.escape(parent_str)}/venv-[A-Za-z0-9_.-]+",
            str(venv_dir),
            text,
        )
        if new_text != text:
            cfg.write_text(new_text, encoding="utf-8")


_CORE_LABEL_DARWIN = "pro.sygen.core"
_CORE_LABEL_LINUX = "sygen-core"
_ADMIN_LABEL_DARWIN = "pro.sygen.admin"
_ADMIN_LABEL_LINUX = "sygen-admin"


def _service_label(role: str) -> str:
    """``role`` is ``core`` or ``admin``. Returns the platform-specific label."""
    if role == "core":
        return _CORE_LABEL_DARWIN if platform.system() == "Darwin" else _CORE_LABEL_LINUX
    return _ADMIN_LABEL_DARWIN if platform.system() == "Darwin" else _ADMIN_LABEL_LINUX


def _apply_core(version: str) -> None:
    """Download wheel (SHA256-verified), install in fresh venv, mv-swap.

    The mv-swap is bracketed by stop/start of the core service so launchd's
    KeepAlive (or systemd Restart=always) can't respawn into the empty
    $VENV_DIR window between the two ``rename()`` calls (P1-6).
    """
    cfg = _config()
    wheel_name = f"sygen-{version}-py3-none-any.whl"
    wheel_url = _release_asset_url("core", version, wheel_name, cfg)
    wheel_path = Path(tempfile.mkdtemp(prefix="sygen-wheel-")) / wheel_name
    logger.info("downloading + verifying core wheel %s", wheel_url)
    _download_verified(wheel_url, wheel_path)

    # Build the new venv next to the live one. --clear ensures a clean
    # site-packages — we never want to inherit stale packages from a
    # prior aborted apply.
    venv_new = VENV_DIR.with_name("venv-new")
    venv_prev = VENV_DIR.with_name("venv-prev")
    if venv_new.exists():
        shutil.rmtree(venv_new)
    logger.info("creating new venv at %s with %s", venv_new, _python_for_new_venv())
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
        "core", version, f"sygen_updater-{version}-py3-none-any.whl", cfg
    )
    updater_wheel_path = wheel_path.with_name(f"sygen_updater-{version}-py3-none-any.whl")
    try:
        _download_verified(updater_wheel_url, updater_wheel_path)
        subprocess.run(
            [str(pip_new), "install", "--quiet", str(updater_wheel_path)],
            check=True,
            timeout=300,
        )
    except Exception as exc:
        logger.warning("could not install sygen-updater into new venv: %s", exc)

    # P1-6: stop core *before* the swap, restart after, so KeepAlive can't
    # respawn into the empty-VENV_DIR window. If the second rename fails,
    # restore venv_prev → VENV_DIR before re-starting so we never start
    # the service against a missing path.
    core_label = _service_label("core")
    _stop_service(core_label)
    try:
        if venv_prev.exists():
            shutil.rmtree(venv_prev)
        renamed_away = False
        if VENV_DIR.exists():
            VENV_DIR.rename(venv_prev)
            renamed_away = True
        try:
            venv_new.rename(VENV_DIR)
        except OSError:
            if renamed_away and venv_prev.exists():
                venv_prev.rename(VENV_DIR)
            raise
        # P0 (1.6.78): pip baked /srv/sygen/venv-new/bin/python3 into every
        # script's shebang and pyvenv.cfg. After the rename above those
        # paths point at a directory that no longer exists, so systemd
        # ExecStart=/srv/sygen/venv/bin/sygen fails 203/EXEC and the unit
        # flaps. Rewrite shebangs in place to the final venv path. If the
        # rewrite itself fails mid-way we'd leave VENV_DIR half-patched, so
        # roll back to venv_prev and let the caller surface the error.
        try:
            _rewrite_venv_shebangs(VENV_DIR)
        except OSError as exc:
            logger.error("shebang rewrite failed: %s — rolling back venv swap", exc)
            try:
                VENV_DIR.rename(venv_new)
            except OSError:
                pass
            if renamed_away and venv_prev.exists():
                venv_prev.rename(VENV_DIR)
            raise
        # P0 (1.6.79): persist the new version BEFORE start_service so /check
        # reflects the swap on the very next poll. Done inside the try-block
        # so a rolled-back swap (rename or shebang-rewrite failure) leaves
        # the pins on the previous version.
        _update_env_pin("SYGEN_CORE_VERSION", version)
        _update_install_manifest("core_version", version)
    finally:
        _start_service(core_label)
    shutil.rmtree(wheel_path.parent, ignore_errors=True)


def _apply_admin(version: str) -> None:
    """Download admin tarball (SHA256-verified), extract, mv-swap.

    Same stop/start bracketing as ``_apply_core`` — the empty-$ADMIN_DIR
    window between the two ``rename()`` calls would otherwise let
    KeepAlive respawn ``node $ADMIN_DIR/server.js`` against ENOENT.
    """
    cfg = _config()
    tarball_name = f"sygen-admin-{version}.tar.gz"
    tarball_url = _release_asset_url("admin", version, tarball_name, cfg)
    tmp_root = Path(tempfile.mkdtemp(prefix="sygen-admin-"))
    tarball_path = tmp_root / tarball_name
    logger.info("downloading + verifying admin tarball %s", tarball_url)
    _download_verified(tarball_url, tarball_path)

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

    admin_label = _service_label("admin")
    _stop_service(admin_label)
    try:
        if admin_prev.exists():
            shutil.rmtree(admin_prev)
        renamed_away = False
        if ADMIN_DIR.exists():
            ADMIN_DIR.rename(admin_prev)
            renamed_away = True
        try:
            staging.rename(ADMIN_DIR)
        except OSError:
            if renamed_away and admin_prev.exists():
                admin_prev.rename(ADMIN_DIR)
            raise
        # P0 (1.6.79): persist the new version BEFORE start_service so /check
        # reflects the swap on the very next poll. Done inside the try-block
        # so a failed rename leaves the pins on the previous version.
        _update_env_pin("SYGEN_ADMIN_VERSION", version)
        _update_install_manifest("admin_version", version)
    finally:
        _start_service(admin_label)
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


# P1-11: cap accepted request body. Local-only attack surface
# (LISTEN_HOST=127.0.0.1) but on a shared dev mac it's nonzero; 1 MiB
# is comfortably above any legitimate /apply request (zero body in
# practice — auth-only POST).
_MAX_REQUEST_BODY = 1 << 20


class _RequestTooLarge(ValueError):
    """Raised when Content-Length exceeds ``_MAX_REQUEST_BODY``."""


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
    try:
        length = int(headers.get("content-length", "0") or "0")
    except ValueError:
        length = 0
    if length < 0:
        length = 0
    if length > _MAX_REQUEST_BODY:
        raise _RequestTooLarge(f"content-length {length} > limit {_MAX_REQUEST_BODY}")
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
    # P1-12: include 404 + 413 + 504 so strict HTTP parsers don't reject
    # "OK" as the reason for non-200 codes.
    reason = {
        200: "OK",
        401: "Unauthorized",
        404: "Not Found",
        413: "Payload Too Large",
        429: "Too Many Requests",
        500: "Internal Server Error",
        503: "Service Unavailable",
        504: "Gateway Timeout",
    }.get(status_code, "OK")
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


def _parse_apply_body(body: bytes) -> dict[str, Any]:
    """Parse the optional /apply request body.

    1.6.115: callers may pass ``{"force_npm_preexisting": true}`` to
    opt the post-apply npm refresh into also bumping packages from
    ``preexisting_npm`` (user-owned CLIs install.sh detected but did
    not install). 1.6.116+: that flag is a deprecated noop — sygen now
    always refreshes both buckets. The body is still parsed verbatim so
    1.6.115 clients keep working unchanged. An empty / missing /
    invalid body is treated as the default ``{}``.
    """
    if not body:
        return {}
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


async def _handle_apply(headers: dict[str, str], body: bytes = b"") -> bytes:
    global _last_apply_at

    if not _auth_ok(headers):
        return _http_response(401, {"ok": False, "error": "missing or invalid bearer token"})
    cfg = _config()
    if not cfg.get("SYGEN_UPDATER_TOKEN"):
        return _http_response(503, {"ok": False, "error": "updater token not configured"})

    parsed_body = _parse_apply_body(body)
    force_npm_preexisting = bool(parsed_body.get("force_npm_preexisting", False))

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

        # Fetch latest release versions for both core and admin (mirror-aware).
        core_latest = await loop.run_in_executor(None, fetch_latest_for, cfg, "core")
        admin_latest = await loop.run_in_executor(None, fetch_latest_for, cfg, "admin")
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
                # _apply_core writes the new version pin to .env +
                # .install_manifest.json itself (1.6.79) once the swap is
                # past the rollback window — no second pin update here.
                await asyncio.wait_for(
                    loop.run_in_executor(None, _apply_core, core_latest),
                    timeout=APPLY_TIMEOUT,
                )
                applied.append(f"core {core_current} → {core_latest}")

            if admin_latest != admin_current:
                logger.info("applying admin %s → %s", admin_current, admin_latest)
                await asyncio.wait_for(
                    loop.run_in_executor(None, _apply_admin, admin_latest),
                    timeout=APPLY_TIMEOUT,
                )
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
        # alive (we're inside it). _apply_* already stop+start the affected
        # service to bracket the mv-swap (P1-6); the kickstart here is
        # belt-and-suspenders so a no-op _apply (e.g. version unchanged
        # but operator forced /apply) still picks up new env or .plist.
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

        # P1-9: health poll. After kicking the services, give them up to
        # 30s to answer. We don't auto-rollback yet (architectural follow-
        # up for v1.7.1) — just report ``healthy=false`` so the admin UI
        # can surface the failure and the operator can decide.
        admin_port = cfg.get("SYGEN_ADMIN_PORT", os.environ.get("SYGEN_ADMIN_PORT", "8080"))
        core_url = "http://127.0.0.1:8081/api/system/status"
        admin_url = f"http://127.0.0.1:{admin_port}/"
        core_healthy, core_code, core_err = await loop.run_in_executor(
            None, _poll_health, core_url, 30.0
        )
        admin_healthy, admin_code, admin_err = await loop.run_in_executor(
            None, _poll_health, admin_url, 30.0
        )
        healthy = core_healthy and admin_healthy

        # P0 (1.6.83): refresh globally-installed npm CLIs (claude) regardless
        # of whether core/admin needed swapping. Banner can be triggered by a
        # cli_tools-only delta — `_apply_core` is then never called and a
        # post-hook nested inside it would never fire. The npm refresh has to
        # live at the top of `_handle_apply` so it always runs.
        try:
            npm_results = _post_apply_update_npm_packages(
                force_npm_preexisting=force_npm_preexisting,
            )
            _record_apply_npm_results(npm_results)
        except Exception as exc:  # pragma: no cover — defensive belt
            logger.warning("post-apply npm refresh raised: %s", exc)
            npm_results = {"updated": [], "skipped": [], "errors": [str(exc)]}

        # P0 (1.6.88): warm the fastembed model cache in the new venv so the
        # first user message after Apply doesn't stall on a cold ONNX load.
        # Backgrounded so a slow warmup can't hold the /apply response —
        # core's own startup warmup will catch up regardless.
        try:
            await loop.run_in_executor(None, _post_apply_warmup_embeddings)
        except Exception as exc:  # pragma: no cover — defensive belt
            logger.warning("post-apply embedding warmup raised: %s", exc)

        # Refresh state so the next /api/system/updates read reflects new pins.
        # run_check() merges with existing state.json (1.6.81), preserving
        # last_apply_npm_results we just wrote.
        state_after = await run_check()
        return _http_response(
            200,
            {
                "ok": True,
                "applied": applied,
                "restarted": restarted,
                "state": state_after,
                "npm": npm_results,
                "healthy": healthy,
                "rolled_back": False,
                "health": {
                    "core": {"healthy": core_healthy, "last_code": core_code, "last_error": core_err},
                    "admin": {"healthy": admin_healthy, "last_code": admin_code, "last_error": admin_err},
                },
            },
        )


async def _serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        method, path, headers, body = await _read_request(reader)
    except _RequestTooLarge as exc:
        try:
            writer.write(
                _http_response(
                    413,
                    {"ok": False, "error": str(exc), "max_bytes": _MAX_REQUEST_BODY},
                )
            )
            await writer.drain()
        except ConnectionError:
            pass
        writer.close()
        return
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
            resp = await _handle_apply(headers, body)
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


def _enforce_loopback_bind() -> None:
    """Refuse to expose /apply beyond loopback unless the operator opts in.

    /apply is bearer-authed but executes a privileged venv/admin swap that
    restarts core+admin. A misconfigured ``SYGEN_UPDATER_HOST=0.0.0.0`` (or
    a LAN address) silently turns it into a remote-controllable kill switch
    on the local network. Default 127.0.0.1 is fine; anything else needs
    explicit ``SYGEN_UPDATER_ALLOW_REMOTE=1`` to acknowledge the risk.
    """
    if LISTEN_HOST in ("127.0.0.1", "::1", "localhost"):
        return
    if os.environ.get("SYGEN_UPDATER_ALLOW_REMOTE", "0") != "1":
        raise RuntimeError(
            f"Refusing to bind on {LISTEN_HOST}. /apply must stay on loopback. "
            "Set SYGEN_UPDATER_ALLOW_REMOTE=1 only if you know what you're doing."
        )
    logger.warning(
        "Updater binding on non-loopback %s — /apply exposed beyond localhost",
        LISTEN_HOST,
    )


def _warn_on_non_default_repos() -> None:
    """Warn loudly when GH release source is overridden.

    An attacker with .env (or process env) write access could redirect the
    updater to a fork they control and roll the host. Default is silent —
    this turns the override into a paper trail in the journal.
    """
    cfg = _config()
    releases_repo = _releases_repo(cfg)
    if not releases_repo:
        # Legacy mode opted in explicitly.
        core_repo = _legacy_source_repo(cfg, "core")
        admin_repo = _legacy_source_repo(cfg, "admin")
        logger.warning(
            "Updater in LEGACY mode (SYGEN_RELEASES_GITHUB_REPO=''): polling "
            "private source repos directly via /releases/latest with v-tags. "
            "core=%s admin=%s. Anonymous curl can't reach private repos — "
            "set GH_TOKEN or switch back to mirror mode.",
            core_repo,
            admin_repo,
        )
        return
    if releases_repo != _DEFAULT_RELEASES_REPO:
        logger.warning(
            "Updater fetching from non-default mirror repo: %s. "
            "Proceed only if this was intentional — an attacker with .env "
            "write access could redirect updates here.",
            releases_repo,
        )


async def main() -> None:
    _enforce_loopback_bind()
    _warn_on_non_default_repos()
    logger.info(
        "sygen-updater starting — listen=%s:%s home=%s state=%s interval=%ss",
        LISTEN_HOST,
        LISTEN_PORT,
        SYGEN_HOME,
        STATE_PATH,
        CHECK_INTERVAL,
    )
    # P0-3: surface the resolved Python interpreter so the operator can
    # verify after each restart that the updater will seed new venvs
    # with the same brew/apt python the live venv was built with.
    logger.info("Updater: using python at %s", _python_for_new_venv())
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
