import sys
import unittest


class DesktopRuntimeTests(unittest.TestCase):
    def test_desktop_shell_uses_bundled_qt_without_pythonnet(self) -> None:
        import app.desktop  # noqa: F401 - import is the assertion

        self.assertIn("PySide6", sys.modules)
        self.assertNotIn("webview", sys.modules)
        self.assertNotIn("pythonnet", sys.modules)
        self.assertNotIn("clr", sys.modules)


if __name__ == "__main__":
    unittest.main()
