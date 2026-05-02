"""Tests for stale-shebang rewrite after the venv mv-swap (P0 in v1.6.78).

Python's stdlib ``venv`` and pip bake the absolute interpreter path into
every generated entry-point script (``#!.../bin/python3``) and into
``pyvenv.cfg``'s ``command =`` line. The updater builds the venv at
``$SYGEN_HOME/venv-new`` and then renames it to ``$SYGEN_HOME/venv`` —
after that rename, every shebang points at a directory that no longer
exists and ``ExecStart=/srv/sygen/venv/bin/sygen`` fails with
``status=203/EXEC``. Reproduced on prod VPS today (sygen-core 47×
restart). ``_rewrite_venv_shebangs`` patches the new venv in place after
the rename so the entry points actually execute.

These tests build a fake venv-new tree, rename it to venv (mirroring the
real flow), call the rewrite, and assert: python shebangs point to the
final dir, non-python shebangs and binaries are untouched, exec bits are
preserved, and pyvenv.cfg sibling paths are rewritten while system paths
stay intact.
"""

from __future__ import annotations

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import updater  # noqa: E402


class RewriteVenvShebangsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.build_dir = self.root / "venv-new"
        self.final_dir = self.root / "venv"
        (self.build_dir / "bin").mkdir(parents=True)

        # Pip-style entry-point script with hard-coded build-time interpreter.
        self.script = self.build_dir / "bin" / "sygen"
        self.script.write_text(
            f"#!{self.build_dir}/bin/python3\n"
            "# -*- coding: utf-8 -*-\n"
            "import sys\n"
            "from updater import entrypoint\n"
            "sys.exit(entrypoint())\n"
        )
        self.script.chmod(0o755)

        # Versioned shebang (e.g. ``python3.14``) — also a real pip output.
        self.versioned = self.build_dir / "bin" / "pip3.14"
        self.versioned.write_text(
            f"#!{self.build_dir}/bin/python3.14\nimport sys\n"
        )
        self.versioned.chmod(0o755)

        # Non-python shell shebang — must be left alone.
        self.shell = self.build_dir / "bin" / "activate-helper"
        self.shell.write_text("#!/bin/sh\nexport PATH=...\n")
        self.shell.chmod(0o755)

        # Binary masquerading as an executable file (no shebang). Stand-in
        # for the python interpreter symlink/binary in real venvs.
        self.bin_blob = self.build_dir / "bin" / "python3"
        self.bin_blob.write_bytes(b"\x7fELFnotreallyabinary\x00\x01\x02")
        self.bin_blob.chmod(0o755)

        # Realistic pyvenv.cfg as Python emits during ``python -m venv``.
        self.cfg = self.build_dir / "pyvenv.cfg"
        self.cfg.write_text(
            "home = /opt/homebrew/opt/python@3.14/bin\n"
            "include-system-site-packages = false\n"
            "version = 3.14.1\n"
            "executable = /opt/homebrew/opt/python@3.14/bin/python3.14\n"
            "command = /opt/homebrew/opt/python@3.14/bin/python3.14 "
            f"-m venv --clear {self.build_dir}\n"
        )

        # Simulate the atomic mv-swap the updater performs.
        self.build_dir.rename(self.final_dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # --- python shebangs --------------------------------------------------

    def test_python_shebang_rewritten_to_final_path(self) -> None:
        updater._rewrite_venv_shebangs(self.final_dir)
        first_line = (self.final_dir / "bin" / "sygen").read_text().splitlines()[0]
        self.assertEqual(first_line, f"#!{self.final_dir}/bin/python3")

    def test_versioned_python_shebang_rewritten(self) -> None:
        updater._rewrite_venv_shebangs(self.final_dir)
        first_line = (self.final_dir / "bin" / "pip3.14").read_text().splitlines()[0]
        self.assertEqual(first_line, f"#!{self.final_dir}/bin/python3.14")

    def test_shebang_with_trailing_flags_preserved(self) -> None:
        # Some pip-installed scripts emit ``#!.../python3 -sE``; we must keep
        # the flag suffix when rewriting the path.
        flagged = self.final_dir / "bin" / "with-flags"
        flagged.write_text(
            f"#!{self.build_dir}/bin/python3 -sE\nprint('hi')\n"
        )
        flagged.chmod(0o755)
        updater._rewrite_venv_shebangs(self.final_dir)
        first_line = flagged.read_text().splitlines()[0]
        self.assertEqual(first_line, f"#!{self.final_dir}/bin/python3 -sE")

    def test_script_body_unchanged(self) -> None:
        before = (self.final_dir / "bin" / "sygen").read_text().splitlines()[1:]
        updater._rewrite_venv_shebangs(self.final_dir)
        after = (self.final_dir / "bin" / "sygen").read_text().splitlines()[1:]
        self.assertEqual(before, after)

    # --- left-alone cases -------------------------------------------------

    def test_non_python_shebang_left_alone(self) -> None:
        updater._rewrite_venv_shebangs(self.final_dir)
        first_line = (self.final_dir / "bin" / "activate-helper").read_text().splitlines()[0]
        self.assertEqual(first_line, "#!/bin/sh")

    def test_binary_file_left_alone(self) -> None:
        updater._rewrite_venv_shebangs(self.final_dir)
        self.assertEqual(
            (self.final_dir / "bin" / "python3").read_bytes(),
            b"\x7fELFnotreallyabinary\x00\x01\x02",
        )

    # --- mode preservation ------------------------------------------------

    def test_exec_bit_preserved(self) -> None:
        updater._rewrite_venv_shebangs(self.final_dir)
        mode = (self.final_dir / "bin" / "sygen").stat().st_mode
        self.assertTrue(mode & stat.S_IXUSR)
        self.assertTrue(mode & stat.S_IXGRP)
        self.assertTrue(mode & stat.S_IXOTH)

    # --- pyvenv.cfg -------------------------------------------------------

    def test_pyvenv_cfg_rewritten(self) -> None:
        updater._rewrite_venv_shebangs(self.final_dir)
        text = (self.final_dir / "pyvenv.cfg").read_text()
        self.assertNotIn(str(self.build_dir), text)
        self.assertIn(str(self.final_dir), text)

    def test_pyvenv_cfg_system_paths_untouched(self) -> None:
        updater._rewrite_venv_shebangs(self.final_dir)
        text = (self.final_dir / "pyvenv.cfg").read_text()
        # System interpreter paths must survive the rewrite verbatim.
        self.assertIn("home = /opt/homebrew/opt/python@3.14/bin", text)
        self.assertIn("executable = /opt/homebrew/opt/python@3.14/bin/python3.14", text)

    def test_missing_pyvenv_cfg_does_not_raise(self) -> None:
        (self.final_dir / "pyvenv.cfg").unlink()
        # No exception, no other side effects.
        updater._rewrite_venv_shebangs(self.final_dir)

    def test_missing_bin_dir_does_not_raise(self) -> None:
        import shutil as _sh
        _sh.rmtree(self.final_dir / "bin")
        updater._rewrite_venv_shebangs(self.final_dir)

    # --- idempotence ------------------------------------------------------

    def test_idempotent(self) -> None:
        updater._rewrite_venv_shebangs(self.final_dir)
        first = (self.final_dir / "bin" / "sygen").read_bytes()
        first_cfg = (self.final_dir / "pyvenv.cfg").read_text()
        updater._rewrite_venv_shebangs(self.final_dir)
        second = (self.final_dir / "bin" / "sygen").read_bytes()
        second_cfg = (self.final_dir / "pyvenv.cfg").read_text()
        self.assertEqual(first, second)
        self.assertEqual(first_cfg, second_cfg)


if __name__ == "__main__":
    unittest.main()
