"""Application entry point for the insertion-device Tkinter rewrite."""

from __future__ import annotations

import tkinter as tk

from heatflux.config.app_logger import setup_logging
from heatflux.gui.app_window import MainWindow


def main() -> None:
    setup_logging()
    root = tk.Tk()
    MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
