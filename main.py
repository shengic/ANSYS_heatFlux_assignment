"""Application entry point for the insertion-device Tkinter rewrite."""

from __future__ import annotations

import tkinter as tk

from heatflux.gui.app_window import MainWindow


def main() -> None:
    root = tk.Tk()
    MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
