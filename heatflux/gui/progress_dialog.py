from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class ProgressDialog(tk.Toplevel):
    def __init__(self, parent: tk.Misc, title: str):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.status_var = tk.StringVar(value="Starting...")
        self.percent_var = tk.StringVar(value="0%")
        self.progress = ttk.Progressbar(self, orient="horizontal", mode="determinate", maximum=100)

        ttk.Label(self, textvariable=self.status_var).grid(row=0, column=0, padx=12, pady=(12, 4), sticky="w")
        ttk.Label(self, textvariable=self.percent_var).grid(row=0, column=1, padx=12, pady=(12, 4), sticky="e")
        self.progress.grid(row=1, column=0, columnspan=2, padx=12, pady=(0, 12), sticky="ew")
        self.columnconfigure(0, weight=1)

    def update_progress(self, current: int, total: int, stage: str) -> None:
        if total <= 0:
            percent = 0
        else:
            percent = max(0, min(100, int(round((current / total) * 100))))
        self.status_var.set(stage)
        self.percent_var.set(f"{percent}%")
        self.progress["value"] = percent
        self.update_idletasks()

