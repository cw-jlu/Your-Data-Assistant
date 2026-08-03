import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DesktopRuntimeTests(unittest.TestCase):
    def test_desktop_shell_uses_bundled_qt_without_pythonnet(self) -> None:
        import app.desktop  # noqa: F401 - import is the assertion

        self.assertIn("PySide6", sys.modules)
        self.assertNotIn("webview", sys.modules)
        self.assertNotIn("pythonnet", sys.modules)
        self.assertNotIn("clr", sys.modules)

    def test_desktop_recovers_renderer_crashes(self) -> None:
        source = (ROOT / "app" / "desktop.py").read_text(encoding="utf-8")
        self.assertIn("renderProcessTerminated.connect", source)
        self.assertIn("NormalTerminationStatus", source)
        self.assertIn("renderer-error.log", source)


if __name__ == "__main__":
    unittest.main()
