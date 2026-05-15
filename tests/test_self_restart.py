"""Tests for the post-apply self-restart hook (1.6.125).

Background: pre-1.6.125 the running updater process kept executing
whatever ``updater.py`` image it loaded at startup. ``_apply_core``
installed the freshly downloaded ``sygen_updater`` wheel into the new
venv, but the in-memory module never got swapped in — any updater-side
fix stayed dormant on the host until the operator manually ran
``launchctl kickstart -k gui/$UID/pro.sygen.updater`` (or rebooted).
1.6.125 closes that gap: at the end of every successful ``/apply`` the
updater spawns a detached helper that asks launchd / systemd to restart
it. ``KeepAlive`` (Darwin) / ``Restart=always`` (Linux) bring the
service back on the freshly installed wheel.

These tests pin the new behaviour:

* ``_schedule_self_restart`` spawns a detached ``Popen`` with the right
  argv on Darwin and Linux, returns ``True``, and never raises.
* Unsupported platforms return ``False`` without spawning anything.
* ``Popen`` failures are caught + logged + return ``False`` (the apply
  itself must not break because of a self-restart hiccup).
* ``_handle_apply`` calls ``_schedule_self_restart`` exactly once on the
  success path and never on the early-error paths (auth / token /
  rate-limit / version-resolve / timeout / apply-failure).
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import updater  # noqa: E402


class ScheduleSelfRestartTests(unittest.TestCase):
    """Direct unit tests on ``_schedule_self_restart``."""

    def test_darwin_spawns_launchctl_kickstart(self) -> None:
        with (
            mock.patch.object(updater.platform, "system", return_value="Darwin"),
            mock.patch.object(updater.os, "getuid", return_value=501, create=True),
            mock.patch.object(updater.subprocess, "Popen") as popen,
        ):
            ok = updater._schedule_self_restart()

        self.assertTrue(ok)
        self.assertEqual(popen.call_count, 1)
        argv = popen.call_args.args[0]
        # Detached helper: shell wrapper with sleep + launchctl kickstart.
        self.assertEqual(argv[0], "sh")
        self.assertEqual(argv[1], "-c")
        self.assertIn("sleep", argv[2])
        self.assertIn("launchctl kickstart -k", argv[2])
        self.assertIn("gui/501/pro.sygen.updater", argv[2])
        # Detached: new session so launchd's SIGTERM of us doesn't take
        # the helper down before it can fire the kickstart.
        self.assertTrue(popen.call_args.kwargs.get("start_new_session"))
        # No fd inheritance — helper is fire-and-forget.
        self.assertEqual(popen.call_args.kwargs.get("stdin"), subprocess.DEVNULL)
        self.assertEqual(popen.call_args.kwargs.get("stdout"), subprocess.DEVNULL)
        self.assertEqual(popen.call_args.kwargs.get("stderr"), subprocess.DEVNULL)

    def test_linux_spawns_systemctl_restart(self) -> None:
        with (
            mock.patch.object(updater.platform, "system", return_value="Linux"),
            mock.patch.object(updater.subprocess, "Popen") as popen,
        ):
            ok = updater._schedule_self_restart()

        self.assertTrue(ok)
        self.assertEqual(popen.call_count, 1)
        argv = popen.call_args.args[0]
        self.assertEqual(argv[0], "sh")
        self.assertEqual(argv[1], "-c")
        self.assertIn("sleep", argv[2])
        # Linux uses system-level systemctl — install.sh seeds
        # sygen-updater.service as a system unit (matches sygen-core),
        # not a --user unit.
        self.assertIn("systemctl restart sygen-updater", argv[2])
        self.assertNotIn("--user", argv[2])
        self.assertTrue(popen.call_args.kwargs.get("start_new_session"))

    def test_unsupported_platform_no_op(self) -> None:
        with (
            mock.patch.object(updater.platform, "system", return_value="FreeBSD"),
            mock.patch.object(updater.subprocess, "Popen") as popen,
        ):
            ok = updater._schedule_self_restart()
        self.assertFalse(ok)
        self.assertEqual(popen.call_count, 0)

    def test_popen_oserror_swallowed_and_logged(self) -> None:
        with (
            mock.patch.object(updater.platform, "system", return_value="Darwin"),
            mock.patch.object(updater.os, "getuid", return_value=501, create=True),
            mock.patch.object(
                updater.subprocess, "Popen", side_effect=OSError("nope")
            ),
        ):
            ok = updater._schedule_self_restart()
        # Apply flow must not break because the kickstart helper failed.
        self.assertFalse(ok)

    def test_returns_immediately_does_not_block(self) -> None:
        """``Popen`` is fire-and-forget — we must NOT ``communicate()`` /
        ``wait()`` it, otherwise the in-flight HTTP response gets stuck
        behind the 1-second sleep in the helper.
        """
        proc = mock.Mock()
        with (
            mock.patch.object(updater.platform, "system", return_value="Darwin"),
            mock.patch.object(updater.os, "getuid", return_value=501, create=True),
            mock.patch.object(updater.subprocess, "Popen", return_value=proc),
        ):
            updater._schedule_self_restart()
        proc.communicate.assert_not_called()
        proc.wait.assert_not_called()


class _ApplyHarness(unittest.TestCase):
    """Drive ``_handle_apply`` end-to-end with the heavy parts stubbed.

    Apply is a coroutine that calls ``run_check`` + ``_apply_core`` +
    ``_post_apply_update_npm_packages`` + a health poll. We stub every
    one of those so the test focuses on *whether*
    ``_schedule_self_restart`` was called and on what code path.
    """

    BEARER = "test-token"

    def setUp(self) -> None:
        self._patches = [
            mock.patch.object(
                updater,
                "_config",
                return_value={
                    "SYGEN_UPDATER_TOKEN": self.BEARER,
                    "SYGEN_CORE_VERSION": "1.6.124",
                },
            ),
            mock.patch.object(updater, "_apply_core"),
            mock.patch.object(
                updater,
                "_restart_service",
                return_value=(0, "ok"),
            ),
            mock.patch.object(
                updater,
                "_poll_health",
                return_value=(True, 200, None),
            ),
            mock.patch.object(
                updater,
                "_post_apply_update_npm_packages",
                return_value={
                    "updated": [],
                    "skipped": [],
                    "errors": [],
                    "binary_versions": {},
                    "checked_at": "2026-05-07T00:00:00Z",
                },
            ),
            mock.patch.object(updater, "_record_apply_npm_results"),
            mock.patch.object(updater, "_post_apply_warmup_embeddings"),
            mock.patch.object(
                updater,
                "run_check",
                new=mock.AsyncMock(return_value={"checked_at": "2026-05-07T00:00:00Z"}),
            ),
        ]
        for p in self._patches:
            p.start()
        # Reset rate-limit so consecutive tests don't see 429.
        updater._last_apply_at = 0.0

    def tearDown(self) -> None:
        for p in reversed(self._patches):
            p.stop()

    def _headers(self, *, auth: bool = True) -> dict[str, str]:
        return (
            {"authorization": f"Bearer {self.BEARER}"}
            if auth
            else {}
        )

    def _resolve_versions(
        self,
        core: str | None = "1.6.125",
    ) -> mock.MagicMock:
        return mock.patch.object(
            updater,
            "fetch_latest_for",
            side_effect=lambda _cfg, _component: core,
        )


class HandleApplySelfRestartTests(_ApplyHarness):
    def test_success_path_schedules_self_restart(self) -> None:
        with (
            self._resolve_versions(),
            mock.patch.object(
                updater,
                "_schedule_self_restart",
                return_value=True,
            ) as sched,
        ):
            resp = asyncio.run(updater._handle_apply(self._headers(), b""))
        sched.assert_called_once()
        # Response surfaces the self-restart flag for observability.
        self.assertIn(b'"self_restart_scheduled": true', resp)
        # Smoke: status line is 200.
        self.assertTrue(resp.startswith(b"HTTP/1.1 200 "))

    def test_no_op_apply_still_schedules(self) -> None:
        """Even when nothing was applied (versions match), we still
        schedule. Matches the existing belt-and-suspenders core
        kickstart for forced no-op applies.
        """
        with (
            self._resolve_versions(core="1.6.124"),
            mock.patch.object(
                updater, "_schedule_self_restart", return_value=True,
            ) as sched,
        ):
            resp = asyncio.run(updater._handle_apply(self._headers(), b""))
        sched.assert_called_once()
        self.assertTrue(resp.startswith(b"HTTP/1.1 200 "))

    def test_auth_failure_does_not_schedule(self) -> None:
        with mock.patch.object(
            updater, "_schedule_self_restart", return_value=True,
        ) as sched:
            resp = asyncio.run(updater._handle_apply(self._headers(auth=False), b""))
        sched.assert_not_called()
        self.assertTrue(resp.startswith(b"HTTP/1.1 401 "))

    def test_rate_limit_does_not_schedule(self) -> None:
        # Push _last_apply_at into the future so the next call hits 429.
        loop = asyncio.new_event_loop()
        try:
            updater._last_apply_at = loop.time() + 1000
        finally:
            loop.close()
        with mock.patch.object(
            updater, "_schedule_self_restart", return_value=True,
        ) as sched:
            resp = asyncio.run(updater._handle_apply(self._headers(), b""))
        sched.assert_not_called()
        self.assertTrue(resp.startswith(b"HTTP/1.1 429 "))

    def test_version_resolve_failure_does_not_schedule(self) -> None:
        with (
            self._resolve_versions(core=None),
            mock.patch.object(
                updater, "_schedule_self_restart", return_value=True,
            ) as sched,
        ):
            resp = asyncio.run(updater._handle_apply(self._headers(), b""))
        sched.assert_not_called()
        self.assertTrue(resp.startswith(b"HTTP/1.1 500 "))

    def test_apply_subprocess_failure_does_not_schedule(self) -> None:
        with (
            self._resolve_versions(),
            mock.patch.object(
                updater,
                "_apply_core",
                side_effect=subprocess.CalledProcessError(returncode=1, cmd="pip"),
            ),
            mock.patch.object(
                updater, "_schedule_self_restart", return_value=True,
            ) as sched,
        ):
            resp = asyncio.run(updater._handle_apply(self._headers(), b""))
        sched.assert_not_called()
        self.assertTrue(resp.startswith(b"HTTP/1.1 500 "))

    def test_apply_timeout_does_not_schedule(self) -> None:
        async def _slow(*_a, **_kw):
            await asyncio.sleep(10)

        async def _drive() -> bytes:
            with (
                mock.patch.object(updater, "APPLY_TIMEOUT", 0.05),
                mock.patch.object(
                    updater, "_apply_core", side_effect=lambda *a, **kw: None,
                ),
                # Force the wait_for branch into TimeoutError.
                mock.patch.object(
                    asyncio,
                    "wait_for",
                    side_effect=asyncio.TimeoutError,
                ),
                self._resolve_versions(),
                mock.patch.object(
                    updater, "_schedule_self_restart", return_value=True,
                ) as sched,
            ):
                resp = await updater._handle_apply(self._headers(), b"")
                sched.assert_not_called()
                return resp

        resp = asyncio.run(_drive())
        self.assertTrue(resp.startswith(b"HTTP/1.1 504 "))


if __name__ == "__main__":
    unittest.main()
