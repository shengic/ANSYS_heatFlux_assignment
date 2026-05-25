from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np


class GeometryFrame(ttk.LabelFrame):
    def __init__(self, parent: tk.Misc):
        super().__init__(parent, text="SOURCE GEOMETRY (MM)")
        self._entries: dict[str, dict[str, ttk.Entry]] = {}
        self._build()

    def _build(self) -> None:
        headers = ("LABEL", "X", "Y", "Z")
        for col, label in enumerate(headers):
            ttk.Label(self, text=label, style="SectionTitle.TLabel", anchor="center").grid(
                row=0, column=col, padx=6, pady=(4, 2), sticky="n"
            )

        rows = (("Source (S)", "source"), ("Target (T)", "target"), ("Horizontal point (H)", "horizontal"))
        for row_idx, (label, key) in enumerate(rows, start=1):
            ttk.Label(self, text=label, style="Body.TLabel").grid(row=row_idx, column=0, padx=6, pady=4, sticky="w")
            self._entries[key] = {}
            for col_idx, axis in enumerate(("x", "y", "z"), start=1):
                entry = ttk.Entry(self, width=9, style="Card.TEntry", justify="center")
                entry.insert(0, "0.0")
                entry.grid(row=row_idx, column=col_idx, padx=6, pady=4)
                self._entries[key][axis] = entry

        self.columnconfigure(1, weight=0)
        self.columnconfigure(2, weight=0)
        self.columnconfigure(3, weight=0)

        self.set_target_z_default(15000.0)
        self.set_horizontal_x_default(100.0)

    def set_target_z_default(self, value: float) -> None:
        self._set_entry("target", "z", value)

    def set_horizontal_x_default(self, value: float) -> None:
        self._set_entry("horizontal", "x", value)

    def _set_entry(self, point_key: str, axis: str, value: float) -> None:
        entry = self._entries[point_key][axis]
        entry.delete(0, tk.END)
        entry.insert(0, f"{value:.1f}")

    def get_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self._read_point("source"),
            self._read_point("target"),
            self._read_point("horizontal"),
        )

    def set_points(self, source: np.ndarray, target: np.ndarray, horizontal: np.ndarray) -> None:
        for key, point in (("source", source), ("target", target), ("horizontal", horizontal)):
            p = np.asarray(point, dtype=np.float64)
            if p.shape != (3,):
                raise ValueError("point must be shape (3,)")
            self._set_entry(key, "x", float(p[0]))
            self._set_entry(key, "y", float(p[1]))
            self._set_entry(key, "z", float(p[2]))

    def set_inputs_state(self, state: str) -> None:
        for axis_entries in self._entries.values():
            for entry in axis_entries.values():
                entry.configure(state=state)

    def _read_point(self, key: str) -> np.ndarray:
        x = float(self._entries[key]["x"].get())
        y = float(self._entries[key]["y"].get())
        z = float(self._entries[key]["z"].get())
        return np.array([x, y, z], dtype=np.float64)
