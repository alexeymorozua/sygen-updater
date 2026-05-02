"""Unit tests for the GitHub Releases resolution layer.

These cover the v1.6.76 rewrite that switched the updater from polling the
private source repos (with ``v<version>`` tags) to polling the public
``alexeymorozua/sygen-releases`` mirror with prefixed tags
(``core-<version>``, ``admin-<version>``). Legacy v-tag mode is still
reachable via ``SYGEN_RELEASES_GITHUB_REPO=''`` and is tested too.

No network access — every GitHub API call is monkey-patched.
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import Any
from unittest import mock

# Make ``updater`` importable when tests run from the repo root.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import updater  # noqa: E402


class ParseSemverTests(unittest.TestCase):
    def test_strict_numeric_triple(self):
        self.assertEqual(updater._parse_semver("1.6.75"), (1, 6, 75))

    def test_arbitrary_length(self):
        self.assertEqual(updater._parse_semver("1.2.3.4"), (1, 2, 3, 4))

    def test_prerelease_rejected(self):
        # We do not auto-apply rc/beta tags — the install convention treats
        # them as out-of-band. ``_parse_semver`` returns None so the
        # mirror filter skips them.
        self.assertIsNone(updater._parse_semver("1.6.75-rc1"))
        self.assertIsNone(updater._parse_semver("1.6.75rc1"))

    def test_garbage_rejected(self):
        self.assertIsNone(updater._parse_semver("not-a-version"))
        self.assertIsNone(updater._parse_semver(""))


class FetchLatestFromMirrorTests(unittest.TestCase):
    """Mirror-mode polling: filter by prefix, pick highest semver."""

    def _fake_releases(self) -> list[dict[str, Any]]:
        # Mixed payload: both components, draft + prerelease entries that
        # must be skipped, and a non-prefixed tag that's none of our
        # business. Ordered randomly to confirm we sort by semver, not by
        # GitHub's created_at.
        return [
            {"tag_name": "core-1.6.74", "draft": False, "prerelease": False},
            {"tag_name": "admin-0.5.55", "draft": False, "prerelease": False},
            {"tag_name": "core-1.10.0", "draft": False, "prerelease": False},
            {"tag_name": "core-1.6.75", "draft": False, "prerelease": False},
            {"tag_name": "admin-0.5.54", "draft": False, "prerelease": False},
            # Skipped:
            {"tag_name": "core-1.99.0", "draft": True, "prerelease": False},
            {"tag_name": "core-2.0.0-rc1", "draft": False, "prerelease": True},
            {"tag_name": "v1.0.0", "draft": False, "prerelease": False},
            {"tag_name": "core-broken", "draft": False, "prerelease": False},
        ]

    def test_picks_highest_core_semver(self):
        releases = self._fake_releases()
        with mock.patch.object(updater, "_http_get_json", return_value=releases):
            self.assertEqual(
                updater.fetch_latest_from_mirror("alexeymorozua/sygen-releases", "core-"),
                "1.10.0",  # numeric, not lex
            )

    def test_picks_highest_admin_semver(self):
        releases = self._fake_releases()
        with mock.patch.object(updater, "_http_get_json", return_value=releases):
            self.assertEqual(
                updater.fetch_latest_from_mirror("alexeymorozua/sygen-releases", "admin-"),
                "0.5.55",
            )

    def test_skips_drafts_and_prereleases(self):
        # If we didn't skip these, core would resolve to 1.99.0 or 2.0.0-rc1.
        releases = self._fake_releases()
        with mock.patch.object(updater, "_http_get_json", return_value=releases):
            result = updater.fetch_latest_from_mirror("alexeymorozua/sygen-releases", "core-")
            self.assertNotEqual(result, "1.99.0")
            self.assertNotEqual(result, "2.0.0-rc1")

    def test_no_matching_prefix_returns_none(self):
        releases = self._fake_releases()
        with mock.patch.object(updater, "_http_get_json", return_value=releases):
            self.assertIsNone(
                updater.fetch_latest_from_mirror("alexeymorozua/sygen-releases", "updater-")
            )

    def test_api_error_returns_none(self):
        import urllib.error

        def boom(_url, timeout=15.0):
            raise urllib.error.URLError("network down")

        with mock.patch.object(updater, "_http_get_json", side_effect=boom):
            self.assertIsNone(
                updater.fetch_latest_from_mirror("alexeymorozua/sygen-releases", "core-")
            )

    def test_non_list_response_returns_none(self):
        # GitHub sometimes returns ``{"message": "Not Found"}`` for a missing
        # repo even with HTTP 200; defend against it.
        with mock.patch.object(updater, "_http_get_json", return_value={"message": "Not Found"}):
            self.assertIsNone(
                updater.fetch_latest_from_mirror("alexeymorozua/sygen-releases", "core-")
            )


class ReleasesRepoResolutionTests(unittest.TestCase):
    """``_releases_repo`` decides mirror vs legacy based on cfg presence."""

    def test_default_when_unset(self):
        self.assertEqual(updater._releases_repo({}), "alexeymorozua/sygen-releases")

    def test_explicit_default(self):
        self.assertEqual(
            updater._releases_repo({"SYGEN_RELEASES_GITHUB_REPO": "alexeymorozua/sygen-releases"}),
            "alexeymorozua/sygen-releases",
        )

    def test_explicit_empty_means_legacy(self):
        self.assertEqual(updater._releases_repo({"SYGEN_RELEASES_GITHUB_REPO": ""}), "")

    def test_custom_mirror(self):
        self.assertEqual(
            updater._releases_repo({"SYGEN_RELEASES_GITHUB_REPO": "fork/mirror"}),
            "fork/mirror",
        )


class FetchLatestForTests(unittest.TestCase):
    """``fetch_latest_for`` routes to mirror or legacy depending on cfg."""

    def test_mirror_mode_uses_prefix_filter(self):
        cfg = {}  # default → mirror mode
        with mock.patch.object(updater, "fetch_latest_from_mirror", return_value="1.6.75") as m_mirror, \
             mock.patch.object(updater, "fetch_latest_version") as m_legacy:
            self.assertEqual(updater.fetch_latest_for(cfg, "core"), "1.6.75")
            m_mirror.assert_called_once_with("alexeymorozua/sygen-releases", "core-")
            m_legacy.assert_not_called()

    def test_legacy_mode_uses_v_tag_per_source_repo(self):
        cfg = {"SYGEN_RELEASES_GITHUB_REPO": ""}
        with mock.patch.object(updater, "fetch_latest_version", return_value="1.6.75") as m_legacy, \
             mock.patch.object(updater, "fetch_latest_from_mirror") as m_mirror:
            self.assertEqual(updater.fetch_latest_for(cfg, "core"), "1.6.75")
            m_legacy.assert_called_once_with("alexeymorozua/sygen")
            m_mirror.assert_not_called()

    def test_legacy_mode_admin_uses_admin_source_repo(self):
        cfg = {"SYGEN_RELEASES_GITHUB_REPO": ""}
        with mock.patch.object(updater, "fetch_latest_version", return_value="0.5.55") as m_legacy:
            self.assertEqual(updater.fetch_latest_for(cfg, "admin"), "0.5.55")
            m_legacy.assert_called_once_with("alexeymorozua/sygen-admin")

    def test_legacy_mode_honors_custom_source_repo_override(self):
        cfg = {
            "SYGEN_RELEASES_GITHUB_REPO": "",
            "SYGEN_CORE_GITHUB_REPO": "fork/sygen",
        }
        with mock.patch.object(updater, "fetch_latest_version", return_value="9.9.9") as m_legacy:
            updater.fetch_latest_for(cfg, "core")
            m_legacy.assert_called_once_with("fork/sygen")


class ReleaseAssetUrlTests(unittest.TestCase):
    """``_release_asset_url`` builder for both modes."""

    def test_mirror_core_wheel(self):
        cfg = {}  # default mirror
        url = updater._release_asset_url("core", "1.6.75", "sygen-1.6.75-py3-none-any.whl", cfg)
        self.assertEqual(
            url,
            "https://github.com/alexeymorozua/sygen-releases/releases/download/"
            "core-1.6.75/sygen-1.6.75-py3-none-any.whl",
        )

    def test_mirror_admin_tarball(self):
        cfg = {}
        url = updater._release_asset_url("admin", "0.5.55", "sygen-admin-0.5.55.tar.gz", cfg)
        self.assertEqual(
            url,
            "https://github.com/alexeymorozua/sygen-releases/releases/download/"
            "admin-0.5.55/sygen-admin-0.5.55.tar.gz",
        )

    def test_mirror_updater_wheel(self):
        # The updater itself is a "core" component: it ships from the same
        # core-X.Y.Z mirror tag as the sygen wheel.
        cfg = {}
        url = updater._release_asset_url(
            "core", "1.6.75", "sygen_updater-1.6.75-py3-none-any.whl", cfg
        )
        self.assertEqual(
            url,
            "https://github.com/alexeymorozua/sygen-releases/releases/download/"
            "core-1.6.75/sygen_updater-1.6.75-py3-none-any.whl",
        )

    def test_legacy_core_wheel(self):
        cfg = {"SYGEN_RELEASES_GITHUB_REPO": ""}
        url = updater._release_asset_url("core", "1.6.75", "sygen-1.6.75-py3-none-any.whl", cfg)
        self.assertEqual(
            url,
            "https://github.com/alexeymorozua/sygen/releases/download/"
            "v1.6.75/sygen-1.6.75-py3-none-any.whl",
        )

    def test_legacy_admin_tarball(self):
        cfg = {"SYGEN_RELEASES_GITHUB_REPO": ""}
        url = updater._release_asset_url("admin", "0.5.55", "sygen-admin-0.5.55.tar.gz", cfg)
        self.assertEqual(
            url,
            "https://github.com/alexeymorozua/sygen-admin/releases/download/"
            "v0.5.55/sygen-admin-0.5.55.tar.gz",
        )

    def test_custom_mirror_repo(self):
        cfg = {"SYGEN_RELEASES_GITHUB_REPO": "fork/mirror"}
        url = updater._release_asset_url("core", "1.6.75", "sygen-1.6.75-py3-none-any.whl", cfg)
        self.assertEqual(
            url,
            "https://github.com/fork/mirror/releases/download/"
            "core-1.6.75/sygen-1.6.75-py3-none-any.whl",
        )


class PayloadRepoTests(unittest.TestCase):
    """``_payload_repo`` decides which repo string the admin UI sees."""

    def test_mirror_mode_surfaces_mirror_repo(self):
        cfg = {}
        self.assertEqual(updater._payload_repo(cfg, "core"), "alexeymorozua/sygen-releases")
        self.assertEqual(updater._payload_repo(cfg, "admin"), "alexeymorozua/sygen-releases")

    def test_legacy_mode_surfaces_source_repo(self):
        cfg = {"SYGEN_RELEASES_GITHUB_REPO": ""}
        self.assertEqual(updater._payload_repo(cfg, "core"), "alexeymorozua/sygen")
        self.assertEqual(updater._payload_repo(cfg, "admin"), "alexeymorozua/sygen-admin")


if __name__ == "__main__":
    unittest.main()
