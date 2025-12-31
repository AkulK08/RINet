from __future__ import annotations
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from allograph_desktop.ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("RINet")
    try:
        app.setApplicationDisplayName("RINet")
    except Exception:
        pass

    # Prefer user-provided icon in Downloads, else fallback to repo icon
    user_icon = Path.home() / "Downloads" / "logo.png"
    repo_icon = Path(__file__).parent / "ui" / "icons" / "rinet.png"
    icon_path = user_icon if user_icon.exists() else repo_icon
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
