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


class ForceNpmPreexistingTests(_NpmHookHarness):
    """1.6.115: opt-in ``force_npm_preexisting=True`` extends the
    post-apply npm refresh loop to cover ``preexisting_npm`` packages
    in install_manifest.json — used to drive Claude CLI updates from
    the Apply Update button when the user explicitly asked for it.
    """

    def test_default_false_leaves_preexisting_untouched(self) -> None:
        """Regression guard: the new flag must default to False so
        callers without the 1.6.115 update path see exactly the
        pre-1.6.115 behaviour (preexisting CLIs never bumped).
        """
        self._write_manifest(
            installed_npm=["@anthropic-ai/claude-code"],
            preexisting_npm=["typescript"],
        )
        with mock.patch.object(subprocess, "run", side_effect=_ok) as run:
            results = updater._post_apply_update_npm_packages()
        commands = [tuple(c.args[0]) for c in run.call_args_list]
        self.assertEqual(
            commands,
            [("/usr/bin/npm", "update", "-g", "@anthropic-ai/claude-code")],
        )
        self.assertNotIn(
            ("/usr/bin/npm", "update", "-g", "typescript"),
            commands,
        )
        self.assertNotIn("typescript", results["updated"])

    def test_flag_true_runs_npm_update_for_preexisting(self) -> None:
        """When the flag is set, every package in ``preexisting_npm``
        gets one ``npm update -g`` after the installed_npm loop.
        """
        self._write_manifest(
            installed_npm=["@anthropic-ai/claude-code"],
            preexisting_npm=["typescript", "@vercel/some-cli"],
        )
        with mock.patch.object(subprocess, "run", side_effect=_ok) as run:
            results = updater._post_apply_update_npm_packages(
                force_npm_preexisting=True,
            )
        commands = [tuple(c.args[0]) for c in run.call_args_list]
        self.assertIn(
            ("/usr/bin/npm", "update", "-g", "@anthropic-ai/claude-code"),
            commands,
        )
        self.assertIn(
            ("/usr/bin/npm", "update", "-g", "typescript"),
            commands,
        )
        self.assertIn(
            ("/usr/bin/npm", "update", "-g", "@vercel/some-cli"),
            commands,
        )
        # Both preexisting packages also land in updated[].
        self.assertEqual(
            sorted(results["updated"]),
            sorted([
                "@anthropic-ai/claude-code",
                "typescript",
                "@vercel/some-cli",
            ]),
        )

    def test_preexisting_failure_recorded_but_does_not_skip_others(self) -> None:
        """A single preexisting npm failure must be recorded with
        ``preexisting=True`` and NOT abort the loop — remaining
        packages still get refreshed (graceful per-package).
        """
        self._write_manifest(
            installed_npm=["@anthropic-ai/claude-code"],
            preexisting_npm=["broken-pkg", "ok-pkg"],
        )

        def _run(*args, **_kw):
            argv = args[0]
            pkg = argv[-1]
            if pkg == "broken-pkg":
                return mock.Mock(returncode=1, stdout="",
                                  stderr="EACCES: permission denied")
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch.object(subprocess, "run", side_effect=_run):
            results = updater._post_apply_update_npm_packages(
                force_npm_preexisting=True,
            )

        # ok-pkg + installed pkg succeeded.
        self.assertIn("ok-pkg", results["updated"])
        self.assertIn("@anthropic-ai/claude-code", results["updated"])
        # broken-pkg recorded as an error with preexisting marker.
        broken_errors = [e for e in results["errors"] if e["pkg"] == "broken-pkg"]
        self.assertEqual(len(broken_errors), 1)
        self.assertIn("EACCES", broken_errors[0]["error"])
        self.assertTrue(broken_errors[0].get("preexisting"))

    def test_flag_true_records_binary_versions_for_preexisting(self) -> None:
        """Successful preexisting refresh records ``binary_versions``
        the same way installed_npm does — so the core override
        mechanism clears the banner regardless of bucket.
        """
        self._write_manifest(
            installed_npm=[],
            preexisting_npm=["@anthropic-ai/claude-code"],
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
                return mock.Mock(returncode=0, stdout="2.1.129 (Claude Code)\n", stderr="")
            return mock.Mock(returncode=0, stdout="", stderr="")

        with (
            mock.patch.object(updater.shutil, "which", _which),
            mock.patch.object(subprocess, "run", side_effect=_run),
        ):
            results = updater._post_apply_update_npm_packages(
                force_npm_preexisting=True,
            )

        self.assertEqual(results["updated"], ["@anthropic-ai/claude-code"])
        self.assertEqual(results["binary_versions"], {"claude": "2.1.129"})

    def test_duplicate_in_both_buckets_runs_only_once(self) -> None:
        """Defensive: if a package somehow ends up in both
        installed_npm and preexisting_npm, we should not double-run
        npm update on it (would double-count in updated[]).
        """
        self._write_manifest(
            installed_npm=["@anthropic-ai/claude-code"],
            preexisting_npm=["@anthropic-ai/claude-code"],
        )
        with mock.patch.object(subprocess, "run", side_effect=_ok) as run:
            results = updater._post_apply_update_npm_packages(
                force_npm_preexisting=True,
            )
        commands = [tuple(c.args[0]) for c in run.call_args_list]
        # Single npm update call, not two.
        npm_update_calls = [
            c for c in commands
            if c[:3] == ("/usr/bin/npm", "update", "-g")
        ]
        self.assertEqual(len(npm_update_calls), 1)
        self.assertEqual(results["updated"], ["@anthropic-ai/claude-code"])

    def test_empty_preexisting_with_flag_true_no_op(self) -> None:
        """Empty preexisting_npm with the flag set is a silent no-op."""
        self._write_manifest(
            installed_npm=["@anthropic-ai/claude-code"],
            preexisting_npm=[],
        )
        with mock.patch.object(subprocess, "run", side_effect=_ok) as run:
            results = updater._post_apply_update_npm_packages(
                force_npm_preexisting=True,
            )
        commands = [tuple(c.args[0]) for c in run.call_args_list]
        # Only the one installed_npm package was touched.
        self.assertEqual(
            commands,
            [("/usr/bin/npm", "update", "-g", "@anthropic-ai/claude-code")],
        )
        self.assertEqual(results["updated"], ["@anthropic-ai/claude-code"])


class ParseApplyBodyTests(unittest.TestCase):
    """1.6.115: /apply now accepts an optional JSON body. Empty / invalid
    body → empty dict (default behaviour preserved)."""

    def test_empty_body_returns_empty_dict(self) -> None:
        self.assertEqual(updater._parse_apply_body(b""), {})

    def test_well_formed_dict_returned_verbatim(self) -> None:
        body = b'{"force_npm_preexisting": true}'
        self.assertEqual(
            updater._parse_apply_body(body),
            {"force_npm_preexisting": True},
        )

    def test_malformed_json_returns_empty_dict(self) -> None:
        self.assertEqual(updater._parse_apply_body(b"{not json"), {})

    def test_non_dict_returns_empty_dict(self) -> None:
        # Top-level array → not a dict → ignored.
        self.assertEqual(updater._parse_apply_body(b"[1,2,3]"), {})
        # Top-level scalar → not a dict → ignored.
        self.assertEqual(updater._parse_apply_body(b'"hello"'), {})

    def test_non_utf8_returns_empty_dict(self) -> None:
        # Latin-1 byte that isn't valid UTF-8 → graceful empty dict.
        self.assertEqual(updater._parse_apply_body(b"\xff\xfe"), {})


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
