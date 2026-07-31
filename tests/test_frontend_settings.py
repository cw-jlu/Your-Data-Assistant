from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
JAVASCRIPT = (ROOT / "static" / "app.js").read_text(encoding="utf-8")


class FrontendSettingsTests(unittest.TestCase):
    def test_llm_settings_are_visible_and_have_no_built_in_credentials(self) -> None:
        for element_id in (
            "global-settings-trigger",
            "llm-config-state",
            "api-base",
            "model-name",
            "api-key",
            "settings-save",
            "settings-clear",
        ):
            self.assertIn(f'id="{element_id}"', HTML)

        for element_id in ("api-base", "model-name", "api-key"):
            tag = re.search(rf"<input\b[^>]*\bid=\"{element_id}\"[^>]*>", HTML)
            self.assertIsNotNone(tag, element_id)
            self.assertNotRegex(tag.group(0), r"\bvalue\s*=", element_id)

    def test_api_key_is_excluded_from_persistent_browser_settings(self) -> None:
        save_function = JAVASCRIPT.split("function saveLlmSettings()", 1)[1].split(
            "function clearLlmSettings()", 1
        )[0]
        self.assertNotIn('$("#api-key")', save_function)
        self.assertNotIn("apiKey", save_function)
        self.assertIn("localStorage.setItem", save_function)

    def test_example_config_contains_no_api_key(self) -> None:
        config = (ROOT / "configs" / "react_baseline.local.yaml").read_text(
            encoding="utf-8"
        )
        self.assertRegex(config, r"(?m)^\s*api_key:\s*\"\"\s*$")

    def test_tracked_source_contains_no_sk_style_secret(self) -> None:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        secret_pattern = re.compile(rb"(?:^|[^A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}")
        offenders: list[str] = []
        for raw_path in result.stdout.split(b"\0"):
            if not raw_path:
                continue
            relative = raw_path.decode("utf-8")
            path = ROOT / relative
            try:
                content = path.read_bytes()
            except OSError:
                continue
            if b"\0" not in content and secret_pattern.search(content):
                offenders.append(relative)
        self.assertEqual([], offenders, f"possible API keys found in: {offenders}")


if __name__ == "__main__":
    unittest.main()
