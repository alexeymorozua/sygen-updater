"""Tests for on-disk version tracking after Apply Update (P0/P1 in v1.6.79).

Background: pre-1.6.79 the ``_handle_apply`` handler called
``_update_env_pin`` *after* ``_apply_core`` / ``_apply_admin`` returned,
and ``$SYGEN_ROOT/.install_manifest.json`` was never refreshed at all.
Symptom: after a successful Apply Update the admin banner kept showing
"update available" because the next ``/check`` read the stale pin back
from ``$SYGEN_HOME/.env``. Fix moves the pin write inside
``_apply_core`` / ``_apply_admin``, just after the rollback window
closes (post-shebang-rewrite / post-rename), and adds a new
``_update_install_manifest`` helper that touches the manifest install.sh
seeded.

These tests stub out the heavy parts of an apply (download, venv build,
service start/stop) and assert just the version-tracking behaviour:

* On a successful core/admin swap, the .env pin AND manifest are updated.
* If the swap rolls back mid-apply (subprocess fail, shebang-rewrite
  fail), the pin stays on the previous version.
* ``_update_install_manifest`` is idempotent and silent-no-ops on a
  missing or unreadable manifest.
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


def _fake_venv_create(cmd, *args, **kwargs):
    """Stand-in for subprocess.run that materialises the venv directory.

    ``_apply_core`` shells out to ``python -m venv --clear <venv_new>``
    expecting the directory to exist after the call. The pip install
    calls that follow are no-ops in tests — we don't need real packages,
    just the directory tree so the subsequent rename succeeds.
    """
    if isinstance(cmd, (list, tuple)) and "-m" in cmd and "venv" in cmd:
        venv_dir = Path(cmd[-1])
        (venv_dir / "bin").mkdir(parents=True, exist_ok=True)
    elif isinstance(cmd, (list, tuple)) and cmd and cmd[0] == "tar":
        # ``tar -xzf <tarball> -C <staging>`` — drop the server.js
        # marker into the staging dir so _apply_admin's existence check
        # passes.
        try:
            staging = Path(cmd[cmd.index("-C") + 1])
        except (ValueError, IndexError):
            staging = None
        if staging is not None:
            (staging / "server.js").write_text("// fake admin\n", encoding="utf-8")
    return mock.Mock(returncode=0, stdout="", stderr="")


class _ApplyHarness(unittest.TestCase):
    """Shared scaffolding: temp SYGEN_HOME with .env + manifest, mocked
    download / subprocess / service hooks. Subclasses run ``_apply_core``
    or ``_apply_admin`` against the patched module constants.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        # Mirror the VPS layout: SYGEN_ROOT/data/ holds .env, manifest is at
        # SYGEN_ROOT/.install_manifest.json (one level above).
        self.sygen_root = self.root / "sygen"
        self.sygen_home = self.sygen_root / "data"
        self.sygen_home.mkdir(parents=True)
        self.env_file = self.sygen_home / ".env"
        self.env_file.write_text(
            "SYGEN_CORE_VERSION=1.6.78\n"
            "SYGEN_ADMIN_VERSION=0.5.55\n"
            "SYGEN_UPDATER_TOKEN=test-token\n",
            encoding="utf-8",
        )
        os.chmod(self.env_file, 0o600)
        self.manifest = self.sygen_root / ".install_manifest.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "core_version": "1.6.78",
                    "admin_version": "0.5.55",
                    "install_root": str(self.sygen_root),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.venv_dir = self.sygen_home / "venv"
        self.venv_dir.mkdir()
        (self.venv_dir / "bin").mkdir()
        self.admin_dir = self.sygen_home / "admin"
        self.admin_dir.mkdir()

        # Patch module-level constants so the helper functions read/write
        # against our temp tree.
        self._patches = [
            mock.patch.object(updater, "ENV_FILE", self.env_file),
            mock.patch.object(updater, "INSTALL_MANIFEST", self.manifest),
            mock.patch.object(updater, "SYGEN_HOME", self.sygen_home),
            mock.patch.object(updater, "SYGEN_ROOT", self.sygen_root),
            mock.patch.object(updater, "VENV_DIR", self.venv_dir),
            mock.patch.object(updater, "ADMIN_DIR", self.admin_dir),
            mock.patch.object(updater, "_download_verified", lambda *a, **kw: None),
            mock.patch.object(updater, "_stop_service", lambda *a, **kw: None),
            mock.patch.object(updater, "_start_service", lambda *a, **kw: (0, "")),
            mock.patch.object(updater, "_rewrite_venv_shebangs", lambda *a, **kw: None),
            mock.patch.object(updater, "_python_for_new_venv", lambda: "/usr/bin/python3"),
            mock.patch.object(subprocess, "run", side_effect=_fake_venv_create),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self) -> None:
        for p in reversed(self._patches):
            p.stop()
        self._tmp.cleanup()

    def _read_env_pin(self, key: str) -> str:
        for line in self.env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1]
        return ""

    def _read_manifest_field(self, key: str) -> str:
        return json.loads(self.manifest.read_text(encoding="utf-8")).get(key, "")


class ApplyCoreVersionTrackingTests(_ApplyHarness):
    def test_env_pin_updated_after_successful_swap(self) -> None:
        updater._apply_core("1.6.79")
        self.assertEqual(self._read_env_pin("SYGEN_CORE_VERSION"), "1.6.79")
        # Other keys must survive untouched.
        self.assertEqual(self._read_env_pin("SYGEN_ADMIN_VERSION"), "0.5.55")
        self.assertEqual(self._read_env_pin("SYGEN_UPDATER_TOKEN"), "test-token")

    def test_manifest_core_version_updated_after_successful_swap(self) -> None:
        updater._apply_core("1.6.79")
        self.assertEqual(self._read_manifest_field("core_version"), "1.6.79")
        # Sibling fields (e.g. admin_version, install_root) survive.
        self.assertEqual(self._read_manifest_field("admin_version"), "0.5.55")
        self.assertEqual(self._read_manifest_field("install_root"), str(self.sygen_root))

    def test_pins_unchanged_when_subprocess_fails_mid_apply(self) -> None:
        """Mock the venv-create subprocess to fail. Pins must stay on the
        previous version — no half-applied state."""

        def boom(cmd, *args, **kwargs):
            if isinstance(cmd, (list, tuple)) and "-m" in cmd and "venv" in cmd:
                raise subprocess.CalledProcessError(returncode=1, cmd=cmd)
            return _fake_venv_create(cmd, *args, **kwargs)

        with mock.patch.object(subprocess, "run", side_effect=boom):
            with self.assertRaises(subprocess.CalledProcessError):
                updater._apply_core("1.6.79")
        self.assertEqual(self._read_env_pin("SYGEN_CORE_VERSION"), "1.6.78")
        self.assertEqual(self._read_manifest_field("core_version"), "1.6.78")

    def test_pins_unchanged_when_shebang_rewrite_fails(self) -> None:
        """Rewrite-shebangs failure rolls back the venv swap — the pin
        must reflect the rollback, not the attempted version."""

        def shebang_boom(*args, **kwargs):
            raise OSError("rewrite blew up")

        with mock.patch.object(updater, "_rewrite_venv_shebangs", side_effect=shebang_boom):
            with self.assertRaises(OSError):
                updater._apply_core("1.6.79")
        self.assertEqual(self._read_env_pin("SYGEN_CORE_VERSION"), "1.6.78")
        self.assertEqual(self._read_manifest_field("core_version"), "1.6.78")


class ApplyAdminVersionTrackingTests(_ApplyHarness):
    def test_env_pin_updated_after_successful_swap(self) -> None:
        updater._apply_admin("0.5.56")
        self.assertEqual(self._read_env_pin("SYGEN_ADMIN_VERSION"), "0.5.56")
        self.assertEqual(self._read_env_pin("SYGEN_CORE_VERSION"), "1.6.78")

    def test_manifest_admin_version_updated_after_successful_swap(self) -> None:
        updater._apply_admin("0.5.56")
        self.assertEqual(self._read_manifest_field("admin_version"), "0.5.56")
        self.assertEqual(self._read_manifest_field("core_version"), "1.6.78")

    def test_pins_unchanged_when_tar_extract_fails(self) -> None:
        def tar_boom(cmd, *args, **kwargs):
            if isinstance(cmd, (list, tuple)) and cmd and cmd[0] == "tar":
                raise subprocess.CalledProcessError(returncode=1, cmd=cmd)
            return _fake_venv_create(cmd, *args, **kwargs)

        with mock.patch.object(subprocess, "run", side_effect=tar_boom):
            with self.assertRaises(subprocess.CalledProcessError):
                updater._apply_admin("0.5.56")
        self.assertEqual(self._read_env_pin("SYGEN_ADMIN_VERSION"), "0.5.55")
        self.assertEqual(self._read_manifest_field("admin_version"), "0.5.55")


class UpdateInstallManifestTests(unittest.TestCase):
    """Direct unit tests for the helper — independent of the apply flow."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.manifest = self.root / ".install_manifest.json"
        self._patch = mock.patch.object(updater, "INSTALL_MANIFEST", self.manifest)
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        self._tmp.cleanup()

    def test_missing_manifest_silent_no_op(self) -> None:
        # Legacy installs without the manifest must not raise.
        self.assertFalse(self.manifest.exists())
        updater._update_install_manifest("core_version", "1.6.79")
        self.assertFalse(self.manifest.exists())

    def test_idempotent_no_rewrite_for_same_value(self) -> None:
        self.manifest.write_text(
            json.dumps({"core_version": "1.6.79"}, indent=2) + "\n",
            encoding="utf-8",
        )
        mtime_before = self.manifest.stat().st_mtime_ns
        updater._update_install_manifest("core_version", "1.6.79")
        mtime_after = self.manifest.stat().st_mtime_ns
        # Same value → no replace → mtime untouched.
        self.assertEqual(mtime_before, mtime_after)

    def test_overwrites_existing_field(self) -> None:
        self.manifest.write_text(
            json.dumps({"core_version": "1.6.78", "other": "keep"}, indent=2) + "\n",
            encoding="utf-8",
        )
        updater._update_install_manifest("core_version", "1.6.79")
        data = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(data["core_version"], "1.6.79")
        self.assertEqual(data["other"], "keep")

    def test_appends_new_field(self) -> None:
        self.manifest.write_text(
            json.dumps({"core_version": "1.6.78"}, indent=2) + "\n",
            encoding="utf-8",
        )
        updater._update_install_manifest("admin_version", "0.5.56")
        data = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(data["core_version"], "1.6.78")
        self.assertEqual(data["admin_version"], "0.5.56")

    def test_corrupt_manifest_silent_no_op(self) -> None:
        # A half-written manifest must not crash the apply flow — leave it
        # alone and let the next install.sh run / operator surface it.
        self.manifest.write_text("{not valid json", encoding="utf-8")
        updater._update_install_manifest("core_version", "1.6.79")
        self.assertEqual(self.manifest.read_text(encoding="utf-8"), "{not valid json")

    def test_non_object_manifest_silent_no_op(self) -> None:
        self.manifest.write_text("[1, 2, 3]", encoding="utf-8")
        updater._update_install_manifest("core_version", "1.6.79")
        self.assertEqual(self.manifest.read_text(encoding="utf-8"), "[1, 2, 3]")

    def test_atomic_write_leaves_no_tmp_file(self) -> None:
        self.manifest.write_text(
            json.dumps({"core_version": "1.6.78"}, indent=2) + "\n",
            encoding="utf-8",
        )
        updater._update_install_manifest("core_version", "1.6.79")
        siblings = list(self.manifest.parent.iterdir())
        self.assertEqual([p.name for p in siblings], [".install_manifest.json"])


if __name__ == "__main__":
    unittest.main()
