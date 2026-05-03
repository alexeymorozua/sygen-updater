"""Tests for the post-apply npm refresh hook (P0 in v1.6.81).

Background: Apply Update mv-swaps the venv + admin tarball but leaves
globally-installed CLI tools (claude) frozen at install time. install.sh
records each ``npm install -g`` it ran in
``$SYGEN_ROOT/.install_manifest.json`` under ``installed_npm`` (the
``preexisting_npm`` bucket holds CLIs the user already had — those must
NOT be touched). This module asserts:

* Every package in ``installed_npm`` gets one ``npm update -g <pkg>`` call.
* Packages in ``preexisting_npm`` are never touched.
* npm errors per-package are recorded but do not raise (the apply already
  succeeded by the time the hook runs).
* Manifest absent / empty / corrupt → silent no-op.
* ``last_apply_npm_results`` is merged into the existing state file
  without clobbering the periodic-check payload.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import updater  # noqa: E402


def _ok(*_a, **_kw) -> mock.Mock:
    return mock.Mock(returncode=0, stdout="", stderr="")


def _fail(stderr: str = "EACCES", code: int = 1) -> mock.Mock:
    return mock.Mock(returncode=code, stdout="", stderr=stderr)


class _NpmHookHarness(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.sygen_root = self.root / "sygen"
        self.sygen_root.mkdir()
        self.manifest = self.sygen_root / ".install_manifest.json"
        self.state = self.sygen_root / "host_updates" / "_updates.json"
        self.state.parent.mkdir(parents=True, exist_ok=True)
        # Default: npm is on PATH, but the per-package binary lookup
        # used by binary_versions returns None (no claude in fixture).
        # Tests that exercise the version-readback override `which`.
        def _default_which(name: str) -> str | None:
            return "/usr/bin/npm" if name == "npm" else None

        self._patches = [
            mock.patch.object(updater, "INSTALL_MANIFEST", self.manifest),
            mock.patch.object(updater, "STATE_PATH", self.state),
            mock.patch.object(updater.shutil, "which", _default_which),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self) -> None:
        for p in reversed(self._patches):
            p.stop()
        self._tmp.cleanup()

    def _write_manifest(self, **fields: object) -> None:
        self.manifest.write_text(
            json.dumps(fields, indent=2) + "\n", encoding="utf-8"
        )

    def _write_state(self, payload: dict[str, object]) -> None:
        self.state.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class PostApplyNpmTests(_NpmHookHarness):
    def test_updates_each_installed_npm_package(self) -> None:
        self._write_manifest(
            installed_npm=["@anthropic-ai/claude-code"],
            preexisting_npm=[],
        )
        with mock.patch.object(subprocess, "run", side_effect=_ok) as run:
            results = updater._post_apply_update_npm_packages()
        commands = [tuple(c.args[0]) for c in run.call_args_list]
        self.assertEqual(
            commands,
            [("/usr/bin/npm", "update", "-g", "@anthropic-ai/claude-code")],
        )
        self.assertEqual(results["updated"], ["@anthropic-ai/claude-code"])
        self.assertEqual(results["errors"], [])

    def test_preexisting_npm_is_not_touched(self) -> None:
        """A CLI the user owned before sygen ran (preexisting_npm) must
        not get an npm update — that would silently mutate state outside
        sygen's control. Only the installed_npm bucket is in scope."""
        self._write_manifest(
            installed_npm=["@anthropic-ai/claude-code"],
            preexisting_npm=["typescript"],
        )
        with mock.patch.object(subprocess, "run", side_effect=_ok) as run:
            updater._post_apply_update_npm_packages()
        commands = [tuple(c.args[0]) for c in run.call_args_list]
        self.assertEqual(
            commands,
            [("/usr/bin/npm", "update", "-g", "@anthropic-ai/claude-code")],
        )
        self.assertNotIn(
            ("/usr/bin/npm", "update", "-g", "typescript"),
            commands,
        )

    def test_binary_versions_recorded_after_successful_update(self) -> None:
        """After ``npm update -g`` succeeds, the hook should also run
        ``<binary> --version`` and surface the new version under
        ``binary_versions`` so the core API can override its stale
        cli_tools cache without waiting for the weekly probe."""
        self._write_manifest(
            installed_npm=["@anthropic-ai/claude-code"],
            preexisting_npm=[],
        )

        def _which(name: str) -> str | None:
            if name == "npm":
                return "/usr/bin/npm"
            if name == "claude":
                return "/usr/local/bin/claude"
            return None

        def _run(*args, **_kw):
            argv = args[0]
            if argv[1:2] == ["update"]:
                return mock.Mock(returncode=0, stdout="", stderr="")
            if argv[-1] == "--version":
                return mock.Mock(returncode=0, stdout="2.1.126 (Claude Code)\n", stderr="")
            return mock.Mock(returncode=0, stdout="", stderr="")

        with (
            mock.patch.object(updater.shutil, "which", _which),
            mock.patch.object(subprocess, "run", side_effect=_run),
        ):
            results = updater._post_apply_update_npm_packages()

        self.assertEqual(results["updated"], ["@anthropic-ai/claude-code"])
        self.assertEqual(results["binary_versions"], {"claude": "2.1.126"})

    def test_binary_versions_omits_when_binary_missing(self) -> None:
        """If the binary isn't on PATH after npm update (very rare),
        binary_versions stays empty rather than blocking the apply.
        """
        self._write_manifest(
            installed_npm=["@anthropic-ai/claude-code"],
            preexisting_npm=[],
        )
        with mock.patch.object(subprocess, "run", side_effect=_ok):
            results = updater._post_apply_update_npm_packages()
        self.assertEqual(results["updated"], ["@anthropic-ai/claude-code"])
        self.assertEqual(results["binary_versions"], {})

    def test_npm_failure_recorded_but_does_not_raise(self) -> None:
        self._write_manifest(
            installed_npm=["@anthropic-ai/claude-code"],
            preexisting_npm=[],
        )
        with mock.patch.object(
            subprocess,
            "run",
            side_effect=lambda *a, **kw: _fail("EACCES: permission denied"),
        ):
            results = updater._post_apply_update_npm_packages()
        self.assertEqual(results["updated"], [])
        self.assertEqual(len(results["errors"]), 1)
        self.assertEqual(results["errors"][0]["pkg"], "@anthropic-ai/claude-code")
        self.assertIn("EACCES", results["errors"][0]["error"])

    def test_npm_timeout_recorded_but_does_not_raise(self) -> None:
        self._write_manifest(
            installed_npm=["@anthropic-ai/claude-code"],
            preexisting_npm=[],
        )

        def boom(*_a, **_kw):
            raise subprocess.TimeoutExpired(cmd="npm", timeout=120)

        with mock.patch.object(subprocess, "run", side_effect=boom):
            results = updater._post_apply_update_npm_packages()
        self.assertEqual(results["updated"], [])
        self.assertEqual(len(results["errors"]), 1)
        self.assertIn("timeout", results["errors"][0]["error"].lower())

    def test_missing_manifest_silent_no_op(self) -> None:
        self.assertFalse(self.manifest.exists())
        with mock.patch.object(subprocess, "run", side_effect=_ok) as run:
            results = updater._post_apply_update_npm_packages()
        self.assertEqual(run.call_count, 0)
        self.assertEqual(results["updated"], [])
        self.assertEqual(results["errors"], [])

    def test_empty_installed_npm_silent_no_op(self) -> None:
        self._write_manifest(installed_npm=[], preexisting_npm=[])
        with mock.patch.object(subprocess, "run", side_effect=_ok) as run:
            results = updater._post_apply_update_npm_packages()
        self.assertEqual(run.call_count, 0)
        self.assertEqual(results["updated"], [])

    def test_corrupt_manifest_silent_no_op(self) -> None:
        self.manifest.write_text("{not valid json", encoding="utf-8")
        with mock.patch.object(subprocess, "run", side_effect=_ok) as run:
            results = updater._post_apply_update_npm_packages()
        self.assertEqual(run.call_count, 0)
        self.assertEqual(results["updated"], [])

    def test_npm_not_in_path_records_errors(self) -> None:
        self._write_manifest(
            installed_npm=["@anthropic-ai/claude-code"],
            preexisting_npm=[],
        )
        with (
            mock.patch.object(updater.shutil, "which", lambda _b: None),
            mock.patch.object(subprocess, "run", side_effect=_ok) as run,
        ):
            results = updater._post_apply_update_npm_packages()
        self.assertEqual(run.call_count, 0)
        self.assertEqual(results["updated"], [])
        self.assertEqual(len(results["errors"]), 1)
        self.assertIn("npm not found", results["errors"][0]["error"])


class RecordApplyNpmResultsTests(_NpmHookHarness):
    def test_merges_into_existing_state_without_clobber(self) -> None:
        self._write_state(
            {
                "checked_at": "2026-05-03T00:00:00Z",
                "core": {"current": "1.6.80", "latest": "1.6.81"},
                "admin": {"current": "0.5.57", "latest": "0.5.57"},
            }
        )
        updater._record_apply_npm_results(
            {"updated": ["@anthropic-ai/claude-code"], "skipped": [], "errors": []}
        )
        data = json.loads(self.state.read_text(encoding="utf-8"))
        # Periodic-check fields preserved.
        self.assertEqual(data["checked_at"], "2026-05-03T00:00:00Z")
        self.assertEqual(data["core"]["current"], "1.6.80")
        # New field appended.
        self.assertEqual(
            data["last_apply_npm_results"]["updated"],
            ["@anthropic-ai/claude-code"],
        )

    def test_missing_state_silent_no_op(self) -> None:
        self.assertFalse(self.state.exists())
        updater._record_apply_npm_results({"updated": [], "skipped": [], "errors": []})
        self.assertFalse(self.state.exists())


class RunCheckPreservesApplyResultsTests(_NpmHookHarness):
    """Regression: run_check() must merge with existing state, not clobber.

    Before the merge fix, run_check() wrote a fresh dict and erased
    last_apply_npm_results that _record_apply_npm_results had just merged in.
    The periodic checker would then keep clobbering it forever.
    """

    def test_run_check_preserves_last_apply_npm_results(self) -> None:
        import asyncio

        self._write_state(
            {
                "checked_at": "2026-05-03T00:00:00Z",
                "core": {"current": "1.6.80"},
                "admin": {"current": "0.5.57"},
                "last_apply_npm_results": {
                    "updated": ["@anthropic-ai/claude-code"],
                    "skipped": [],
                    "errors": [],
                    "checked_at": "2026-05-03T00:00:30Z",
                },
            }
        )
        with (
            mock.patch.object(updater, "_config", return_value={
                "SYGEN_CORE_VERSION": "1.6.81",
                "SYGEN_ADMIN_VERSION": "0.5.58",
            }),
            mock.patch.object(updater, "fetch_latest_for", return_value="1.6.81"),
        ):
            asyncio.run(updater.run_check())

        data = json.loads(self.state.read_text(encoding="utf-8"))
        # New periodic fields written.
        self.assertEqual(data["core"]["current"], "1.6.81")
        # Apply results preserved through merge.
        self.assertEqual(
            data["last_apply_npm_results"]["updated"],
            ["@anthropic-ai/claude-code"],
        )

    def test_run_check_returns_merged_state(self) -> None:
        import asyncio

        self._write_state(
            {
                "last_apply_npm_results": {"updated": ["foo"], "skipped": [], "errors": []},
            }
        )
        with (
            mock.patch.object(updater, "_config", return_value={
                "SYGEN_CORE_VERSION": "1.6.81",
                "SYGEN_ADMIN_VERSION": "0.5.58",
            }),
            mock.patch.object(updater, "fetch_latest_for", return_value="1.6.81"),
        ):
            result = asyncio.run(updater.run_check())

        self.assertIn("last_apply_npm_results", result)
        self.assertEqual(result["last_apply_npm_results"]["updated"], ["foo"])


if __name__ == "__main__":
    unittest.main()
