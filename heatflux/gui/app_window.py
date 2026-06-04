from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

_log = logging.getLogger(__name__)

import numpy as np

from heatflux.config.ansys_cache import (
    delete_ansys_cache_entry_file,
    clear_all_ansys_parse_cache,
    delete_ansys_parse_cache,
    has_valid_ansys_parse_cache,
    list_ansys_parse_cache_entries,
    load_ansys_parse_cache,
    save_ansys_parse_cache,
)
from heatflux.config.session_backup import load_session_backup, save_session_backup
from heatflux.config.spectra_cache import (
    clear_all_spectra_parse_cache,
    delete_spectra_cache_entry_file,
    delete_spectra_parse_cache,
    list_spectra_parse_cache_entries,
    load_spectra_parse_cache,
    save_spectra_parse_cache,
)
from heatflux.gui.geometry_frame import GeometryFrame
from heatflux.io.ansys_reader import AnsysParseResult, read_ansys_file
from heatflux.io.output_writer import write_output_from_elements
from heatflux.io.spectra_reader import SpectraParseResult, read_spectra_file
from heatflux.math_core.geometry import build_source_geometry
from heatflux.pipeline.mapping_pipeline import run_mapping
from heatflux.tools.sample_validation import run_default_sample_validation


class MainWindow:
    _COUNT_PATTERN = re.compile(r"\b(nodes|elements|flux)\s*=\s*([0-9,]+)", re.IGNORECASE)

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SPECTRA -> ANSYS Heat Flux Node Assignment")
        self.root.geometry("1000x900")
        self.root.minsize(1080, 820)

        self.ansys_loaded = False
        self.spectra_loaded = False
        self.geometry_valid = False
        self._is_loading_ansys = False
        self._is_loading_spectra = False
        self._is_mapping = False
        self.ansys_result: AnsysParseResult | None = None
        self.spectra_result: SpectraParseResult | None = None
        self.last_output_path: Path | None = None

        self.ansys_path_var = tk.StringVar()
        self.spectra_path_var = tk.StringVar()
        self.output_folder_var = tk.StringVar()
        self.output_filename_var = tk.StringVar(value="EPU66-27A power for ansys.inp")
        self.power_ratio_var = tk.StringVar(value="1.0")

        self.ansys_status_var = tk.StringVar(value="No file loaded")
        self.ansys_percent_var = tk.StringVar(value="0%")
        self.ansys_cache_var = tk.StringVar(value="Cache source: none")
        self.spectra_cache_var = tk.StringVar(value="Cache source: none")
        self.ansys_total_nodes_var = tk.StringVar(value="0")
        self.ansys_total_elements_var = tk.StringVar(value="0")
        self.ansys_flux_elements_var = tk.StringVar(value="0")
        self.spectra_status_var = tk.StringVar(value="No file loaded")
        self.spectra_percent_var = tk.StringVar(value="0%")
        self.spectra_cols_var = tk.StringVar(value="-")
        self.spectra_rows_var = tk.StringVar(value="-")
        self.spectra_x_range_var = tk.StringVar(value="-")
        self.spectra_y_range_var = tk.StringVar(value="-")
        self.spectra_peak_var = tk.StringVar(value="-")
        self.spectra_total_power_var = tk.StringVar(value="-")
        self.mapping_percent_var = tk.StringVar(value="0%")
        self.output_elements_var = tk.StringVar(value="-")
        self.mapped_elements_var = tk.StringVar(value="-")
        self.out_of_grid_var = tk.StringVar(value="-")
        self.total_power_out_var = tk.StringVar(value="-")
        self.total_power_spectra_var = tk.StringVar(value="-")
        self.footer_status_var = tk.StringVar(value="SYSTEM OPERATIONAL")
        self.warn_strip_var = tk.StringVar(value="")
        self._warn_after_id: str | None = None
        self.cache_browser_window: tk.Toplevel | None = None
        self.cache_tree: ttk.Treeview | None = None
        self.loaded_backup_path: Path | None = None

        self._build_layout()
        self._attach_var_traces()
        self.root.protocol("WM_DELETE_WINDOW", self._on_exit)
        self._restore_session_from_backup()
        self._reset_startup_ansys_counters_if_unloaded()
        self._reset_startup_mapping_summary()
        self._set_state_idle()

    def _build_layout(self) -> None:
        self._configure_styles()

        container = ttk.Frame(self.root, style="App.TFrame", padding=(12, 12, 12, 6))
        container.pack(fill=tk.BOTH, expand=True)
        container.rowconfigure(1, weight=1)
        container.columnconfigure(0, weight=1)

        header = ttk.Frame(container, style="Header.TFrame", padding=(14, 8))
        header.grid(row=0, column=0, sticky="ew")
        header_left = ttk.Frame(header, style="Header.TFrame")
        header_left.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(
            header_left,
            text="By Albert Sheng, version 1.0",
            style="HeaderSub.TLabel",
        ).pack(anchor="w", pady=(2, 0))
        ttk.Label(
            header_left,
            text="SPECTRA -> ANSYS Heat Flux Node Assignment",
            style="HeaderTitle.TLabel",
        ).pack(anchor="w")

        footer = ttk.Frame(container, style="Header.TFrame", padding=(14, 4))
        footer.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(footer, textvariable=self.footer_status_var, style="FooterAccent.TLabel").pack(side=tk.LEFT)
        ttk.Label(footer, textvariable=self.warn_strip_var, style="FooterWarn.TLabel").pack(side=tk.RIGHT, padx=(8, 0))

        body = ttk.Frame(container, style="App.TFrame")
        body.grid(row=1, column=0, sticky="nsew", pady=(8, 6))
        body.columnconfigure(0, weight=6)
        body.columnconfigure(1, weight=5)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body, style="App.TFrame")
        right = ttk.Frame(body, style="App.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        right.grid(row=0, column=1, sticky="nsew")

        self._build_ansys_card(left)
        self._build_spectra_card(left)

        geometry_card = self._create_card(left)
        ttk.Label(geometry_card, text="SOURCE GEOMETRY (MM)", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.update_geometry_btn = ttk.Button(geometry_card, text="Update geometry", style="Primary.TButton", command=self._on_update_geometry)
        self.update_geometry_btn.grid(row=0, column=1, sticky="e", pady=(0, 6))
        self.geometry_frame = GeometryFrame(geometry_card)
        self.geometry_frame.configure(text="", style="Card.TLabelframe", padding=8)
        self.geometry_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        geometry_card.columnconfigure(0, weight=1)
        geometry_card.columnconfigure(1, weight=0)

        self._build_output_card(right)
        self._build_mapping_card(right)
        self._build_action_card(right)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self.root.configure(bg="#e6e6e8")

        base_font = "Georgia"
        mono_font = "Georgia"
        self.root.option_add("*Font", "Georgia 10")

        style.configure("App.TFrame", background="#e6e6e8")
        style.configure("Header.TFrame", background="#d9d9db", relief=tk.SOLID, borderwidth=1)
        style.configure("Card.TFrame", background="#f3f3f5", relief=tk.SOLID, borderwidth=1)
        style.configure("FlatCard.TFrame", background="#f3f3f5", relief=tk.FLAT, borderwidth=0)
        style.configure("Inset.TFrame", background="#ececef", relief=tk.FLAT)
        style.configure("Chip.TFrame", background="#e8e8ec", relief=tk.SOLID, borderwidth=1)
        style.configure("FlatChip.TFrame", background="#e8e8ec", relief=tk.FLAT, borderwidth=0)

        style.configure(
            "HeaderTitle.TLabel",
            background="#d9d9db",
            foreground="#1a1c1d",
            font=(base_font, 15, "bold"),
        )
        style.configure("HeaderSub.TLabel", background="#d9d9db", foreground="#4b4d52", font=(base_font, 10, "normal"))
        style.configure("CardTitle.TLabel", background="#f3f3f5", foreground="#5b5b62", font=(base_font, 12, "bold"))
        style.configure("SectionTitle.TLabel", background="#f3f3f5", foreground="#5b5b62", font=(base_font, 10, "bold"))
        style.configure("Body.TLabel", background="#f3f3f5", foreground="#5e5e63", font=(base_font, 10, "normal"))
        style.configure("ChipLabel.TLabel", background="#e8e8ec", foreground="#5e5e63", font=(base_font, 10, "normal"))
        style.configure("ChipValue.TLabel", background="#e8e8ec", foreground="#101114", font=(mono_font, 10, "bold"))
        style.configure("ChipOrangeValue.TLabel", background="#e8e8ec", foreground="#ff6b35", font=(mono_font, 10, "bold"))
        style.configure("Mono.TLabel", background="#f3f3f5", foreground="#101114", font=(mono_font, 10, "normal"))
        style.configure("Value.TLabel", background="#f3f3f5", foreground="#101114", font=(mono_font, 10, "bold"))
        style.configure("MutedValue.TLabel", background="#f3f3f5", foreground="#5e5e63", font=(mono_font, 10, "normal"))
        style.configure("BlueValue.TLabel", background="#f3f3f5", foreground="#00b6eb", font=(mono_font, 10, "bold"))
        style.configure("OrangeValue.TLabel", background="#f3f3f5", foreground="#ff6b35", font=(mono_font, 10, "bold"))
        style.configure("Footer.TLabel", background="#d9d9db", foreground="#3a3c41", font=(base_font, 9, "normal"))
        style.configure("FooterAccent.TLabel", background="#d9d9db", foreground="#00aee0", font=(base_font, 10, "bold"))
        style.configure("FooterWarn.TLabel", background="#d9d9db", foreground="#cc5500", font=(base_font, 9, "normal"))

        style.configure("Card.TEntry", fieldbackground="#ececef", borderwidth=1, relief=tk.SOLID, padding=6, font=(mono_font, 10, "normal"))

        style.configure("Card.TLabelframe", background="#f3f3f5", relief=tk.SOLID, borderwidth=1)
        style.configure("Card.TLabelframe.Label", background="#f3f3f5", foreground="#5b5b62", font=(base_font, 12, "bold"))

        style.configure("Primary.TButton", background="#030304", foreground="#ffffff", borderwidth=0, relief=tk.FLAT, focusthickness=0, focuscolor="#030304", padding=(14, 8), font=(base_font, 10, "bold"))
        style.map("Primary.TButton", background=[("disabled", "#7e7f84"), ("active", "#1b1b1f")], foreground=[("disabled", "#d6d7da"), ("active", "#ffffff")])

        style.configure("Secondary.TButton", background="#ececef", foreground="#1a1c1d", borderwidth=0, relief=tk.FLAT, focusthickness=0, focuscolor="#ececef", padding=(12, 7), font=(base_font, 10, "normal"))
        style.map("Secondary.TButton", background=[("disabled", "#ececef"), ("active", "#e2e2e5")], foreground=[("disabled", "#9a9ba0"), ("active", "#111214")])

        style.configure("Danger.TButton", background="#f7e7e7", foreground="#ba1a1a", borderwidth=0, relief=tk.FLAT, focusthickness=0, focuscolor="#f7e7e7", padding=(12, 7), font=(base_font, 10, "bold"))
        style.map("Danger.TButton", background=[("disabled", "#f1ecec"), ("active", "#eedcdc")], foreground=[("disabled", "#c59a9a"), ("active", "#93000a")])
        style.configure("Disabled.TButton", background="#ececef", foreground="#a8a9ad", borderwidth=0, relief=tk.FLAT, focusthickness=0, focuscolor="#ececef", padding=(12, 7), font=(base_font, 10, "bold"))

        style.configure("Ghost.TButton", background="#f3f3f5", foreground="#1a1c1d", borderwidth=0, relief=tk.FLAT, focusthickness=0, focuscolor="#f3f3f5", padding=(8, 5), font=(base_font, 9, "normal"))
        style.map("Ghost.TButton", background=[("disabled", "#f3f3f5"), ("active", "#ebebee")], foreground=[("disabled", "#9a9ba0"), ("active", "#1a1c1d")])
        style.configure("BoxGhost.TButton", background="#f3f3f5", foreground="#1a1c1d", borderwidth=1, relief=tk.SOLID, focusthickness=0, focuscolor="#f3f3f5", padding=(8, 5), font=(base_font, 9, "normal"))
        style.map("BoxGhost.TButton", background=[("disabled", "#f3f3f5"), ("active", "#ebebee")], foreground=[("disabled", "#9a9ba0"), ("active", "#1a1c1d")])
        style.configure("BoxSecondary.TButton", background="#ececef", foreground="#1a1c1d", borderwidth=1, relief=tk.SOLID, focusthickness=0, focuscolor="#ececef", padding=(12, 7), font=(base_font, 10, "normal"))
        style.map("BoxSecondary.TButton", background=[("disabled", "#ececef"), ("active", "#e2e2e5")], foreground=[("disabled", "#9a9ba0"), ("active", "#111214")])
        style.configure("BoxDanger.TButton", background="#f7e7e7", foreground="#ba1a1a", borderwidth=1, relief=tk.SOLID, focusthickness=0, focuscolor="#f7e7e7", padding=(12, 7), font=(base_font, 10, "bold"))
        style.map("BoxDanger.TButton", background=[("disabled", "#f1ecec"), ("active", "#eedcdc")], foreground=[("disabled", "#c59a9a"), ("active", "#93000a")])

        style.configure("Ans.Horizontal.TProgressbar", troughcolor="#d2d3d7", background="#00b6eb", bordercolor="#d2d3d7", lightcolor="#00b6eb", darkcolor="#00b6eb")
        style.configure("Spectra.Horizontal.TProgressbar", troughcolor="#d2d3d7", background="#ff6b35", bordercolor="#d2d3d7", lightcolor="#ff6b35", darkcolor="#ff6b35")
        style.configure("Mapping.Horizontal.TProgressbar", troughcolor="#d2d3d7", background="#2ca24f", bordercolor="#d2d3d7", lightcolor="#2ca24f", darkcolor="#2ca24f")

    def _create_card(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent, style="Card.TFrame", padding=12)
        frame.pack(fill=tk.X, pady=(0, 4))
        return frame

    def _build_ansys_card(self, parent: ttk.Frame) -> None:
        pair_padx = (8, 0)
        pair_pady = (2, 0)

        frame = self._create_card(parent)
        ttk.Label(frame, text="ANSYS APDL INPUT FILE", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Entry(frame, style="Card.TEntry", textvariable=self.ansys_path_var).grid(row=1, column=0, sticky="ew", padx=(0, 10))
        self.upload_ansys_btn = ttk.Button(frame, text="Upload ANSYS file", width=19, style="Primary.TButton", command=self._on_upload_ansys)
        self.upload_ansys_btn.grid(row=1, column=1, sticky="ew")

        parsing = ttk.Frame(frame, style="Inset.TFrame", padding=8)
        parsing.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 6))
        ttk.Label(parsing, textvariable=self.ansys_status_var, style="Body.TLabel", wraplength=760).grid(row=0, column=0, sticky="w")
        ttk.Label(parsing, textvariable=self.ansys_percent_var, style="BlueValue.TLabel").grid(row=0, column=1, sticky="e")
        self.ansys_progress = ttk.Progressbar(parsing, orient="horizontal", mode="determinate", maximum=100, style="Ans.Horizontal.TProgressbar")
        self.ansys_progress.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        parsing.columnconfigure(0, weight=1)

        ttk.Separator(frame, orient="horizontal").grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 8))

        info_area = ttk.Frame(frame, style="FlatCard.TFrame")
        info_area.grid(row=4, column=0, columnspan=2, sticky="ew", padx=(2, 0))

        stats_left = ttk.Frame(info_area, style="FlatCard.TFrame")
        stats_left.grid(row=0, column=0, sticky="nw")
        ttk.Label(stats_left, text="Total nodes:", style="Body.TLabel").grid(row=0, column=0, sticky="w", pady=pair_pady)
        ttk.Label(stats_left, textvariable=self.ansys_total_nodes_var, style="Value.TLabel").grid(row=0, column=1, sticky="w", padx=pair_padx, pady=pair_pady)
        ttk.Label(stats_left, text="Total elements:", style="Body.TLabel").grid(row=1, column=0, sticky="w", pady=pair_pady)
        ttk.Label(stats_left, textvariable=self.ansys_total_elements_var, style="Value.TLabel").grid(row=1, column=1, sticky="w", padx=pair_padx, pady=pair_pady)
        ttk.Label(stats_left, text="Flux elements:", style="Body.TLabel").grid(row=2, column=0, sticky="w", pady=pair_pady)
        ttk.Label(stats_left, textvariable=self.ansys_flux_elements_var, style="BlueValue.TLabel").grid(row=2, column=1, sticky="w", padx=pair_padx, pady=pair_pady)

        stats_right = ttk.Frame(info_area, style="FlatCard.TFrame")
        stats_right.grid(row=0, column=1, sticky="ne", padx=(20, 0))
        ttk.Label(stats_right, textvariable=self.ansys_cache_var, style="Body.TLabel").grid(row=0, column=0, columnspan=3, sticky="e", pady=(0, 4))
        ttk.Button(stats_right, text="Delete current", style="BoxGhost.TButton", command=self._on_delete_current_ansys_cache).grid(row=1, column=0, sticky="e")
        ttk.Button(stats_right, text="Delete all", style="BoxGhost.TButton", command=self._on_clear_all_ansys_cache).grid(row=1, column=1, sticky="e", padx=(6, 0))
        ttk.Button(stats_right, text="Browse...", style="BoxGhost.TButton", command=self._open_cache_browser).grid(row=1, column=2, sticky="e", padx=(6, 0))

        info_area.columnconfigure(0, weight=1)
        info_area.columnconfigure(1, weight=0)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=0)

    def _build_spectra_card(self, parent: ttk.Frame) -> None:
        pair_padx = (8, 0)
        pair_pady = (2, 0)

        frame = self._create_card(parent)
        ttk.Label(frame, text="SPECTRA POWER DENSITY FILE", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Entry(frame, style="Card.TEntry", textvariable=self.spectra_path_var).grid(row=1, column=0, sticky="ew", padx=(0, 10))
        self.upload_spectra_btn = ttk.Button(frame, text="Upload SPECTRA file", width=19, style="Primary.TButton", command=self._on_upload_spectra)
        self.upload_spectra_btn.grid(row=1, column=1, sticky="ew")

        parsing = ttk.Frame(frame, style="Inset.TFrame", padding=8)
        parsing.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 6))
        ttk.Label(parsing, textvariable=self.spectra_status_var, style="Body.TLabel", wraplength=760).grid(row=0, column=0, sticky="w")
        ttk.Label(parsing, textvariable=self.spectra_percent_var, style="OrangeValue.TLabel").grid(row=0, column=1, sticky="e")
        self.spectra_progress = ttk.Progressbar(parsing, orient="horizontal", mode="determinate", maximum=100, style="Spectra.Horizontal.TProgressbar")
        self.spectra_progress.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        parsing.columnconfigure(0, weight=1)

        ttk.Separator(frame, orient="horizontal").grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 8))

        ttk.Label(frame, text="Columns:", style="Body.TLabel").grid(row=4, column=0, sticky="w", pady=pair_pady)
        ttk.Label(frame, textvariable=self.spectra_cols_var, style="Value.TLabel").grid(row=4, column=1, sticky="w", padx=pair_padx, pady=pair_pady)
        ttk.Label(frame, text="Rows:", style="Body.TLabel").grid(row=5, column=0, sticky="w", pady=pair_pady)
        ttk.Label(frame, textvariable=self.spectra_rows_var, style="Value.TLabel").grid(row=5, column=1, sticky="w", padx=pair_padx, pady=pair_pady)
        ttk.Label(frame, text="X range (mrad):", style="Body.TLabel").grid(row=6, column=0, sticky="w", pady=pair_pady)
        ttk.Label(frame, textvariable=self.spectra_x_range_var, style="Mono.TLabel").grid(row=6, column=1, sticky="w", padx=pair_padx, pady=pair_pady)
        ttk.Label(frame, text="Y range (mrad):", style="Body.TLabel").grid(row=7, column=0, sticky="w", pady=pair_pady)
        ttk.Label(frame, textvariable=self.spectra_y_range_var, style="Mono.TLabel").grid(row=7, column=1, sticky="w", padx=pair_padx, pady=pair_pady)
        ttk.Label(frame, text="Peak power density:", style="Body.TLabel").grid(row=8, column=0, sticky="w", pady=pair_pady)
        ttk.Label(frame, textvariable=self.spectra_peak_var, style="OrangeValue.TLabel").grid(row=8, column=1, sticky="w", padx=pair_padx, pady=pair_pady)
        ttk.Label(frame, text="Total power:", style="Body.TLabel").grid(row=9, column=0, sticky="w", pady=pair_pady)
        ttk.Label(frame, textvariable=self.spectra_total_power_var, style="OrangeValue.TLabel").grid(row=9, column=1, sticky="w", padx=pair_padx, pady=pair_pady)

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=0)

    def _build_output_card(self, parent: ttk.Frame) -> None:
        frame = self._create_card(parent)
        ttk.Label(frame, text="OUTPUT - ANSYS EXTERNAL DATA FILE", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        ttk.Label(frame, text="Output folder:", style="SectionTitle.TLabel").grid(row=1, column=0, columnspan=3, sticky="w")
        self.output_folder_entry = ttk.Entry(frame, style="Card.TEntry", textvariable=self.output_folder_var)
        self.output_folder_entry.grid(row=2, column=0, columnspan=2, sticky="ew", padx=(0, 8), pady=(4, 0))
        self.browse_output_btn = ttk.Button(frame, text="Browse...", width=9, style="BoxSecondary.TButton", command=self._on_browse_output_folder)
        self.browse_output_btn.grid(row=2, column=2, sticky="w", pady=(4, 0))

        ttk.Label(frame, text="Output filename:", style="SectionTitle.TLabel").grid(row=3, column=0, columnspan=3, sticky="w", pady=(12, 0))
        self.output_filename_entry = ttk.Entry(frame, style="Card.TEntry", textvariable=self.output_filename_var)
        self.output_filename_entry.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        ttk.Label(frame, text="Power ratio:", style="SectionTitle.TLabel").grid(row=5, column=0, columnspan=3, sticky="w", pady=(12, 0))
        self.power_ratio_entry = ttk.Entry(frame, style="Card.TEntry", textvariable=self.power_ratio_var, width=12)
        self.power_ratio_entry.grid(row=6, column=0, sticky="w", pady=(4, 0))
        ttk.Label(frame, text="(totalPowerRatio scales col 4 of output)", style="Body.TLabel").grid(row=6, column=1, columnspan=2, sticky="w", padx=(4, 0), pady=(4, 0))
        self.create_btn = ttk.Button(frame, text="Map elemental power density", style="Primary.TButton", command=self._on_create)
        self.create_btn.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=0)

    def _build_mapping_card(self, parent: ttk.Frame) -> None:
        frame = self._create_card(parent)
        ttk.Label(frame, text="MAPPING PROGRESS", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Label(frame, textvariable=self.mapping_percent_var, style="BlueValue.TLabel").grid(row=0, column=1, sticky="e", pady=(0, 8))
        self.mapping_progress = ttk.Progressbar(frame, orient="horizontal", mode="determinate", maximum=100, style="Mapping.Horizontal.TProgressbar")
        self.mapping_progress.grid(row=1, column=0, columnspan=2, sticky="ew")

        ttk.Separator(frame, orient="horizontal").grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 10))

        self._build_metric_chip(frame, row=3, label="Output elements:", value_var=self.output_elements_var, value_style="Value.TLabel")
        self._build_metric_chip(frame, row=4, label="Mapped elements:", value_var=self.mapped_elements_var, value_style="Value.TLabel")
        self._build_metric_chip(frame, row=5, label="Out-of-grid elements:", value_var=self.out_of_grid_var, value_style="Value.TLabel")
        self._build_metric_chip(frame, row=6, label="Total power out:", value_var=self.total_power_out_var, value_style="Value.TLabel")
        self._build_metric_chip(frame, row=7, label="Total power (SPECTRA):", value_var=self.total_power_spectra_var, value_style="OrangeValue.TLabel")
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

    def _build_metric_chip(self, parent: ttk.Frame, row: int, label: str, value_var: tk.StringVar, value_style: str) -> None:
        pair_padx = (8, 0)
        pair_pady = (2, 0)

        chip = ttk.Frame(parent, style="FlatCard.TFrame", padding=(10, 4))
        chip.grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 6))
        chip.columnconfigure(0, weight=0)
        chip.columnconfigure(1, weight=0)
        ttk.Label(chip, text=label, style="Body.TLabel").grid(row=0, column=0, sticky="w", pady=pair_pady)
        ttk.Label(chip, textvariable=value_var, style=value_style).grid(row=0, column=1, sticky="w", padx=pair_padx, pady=pair_pady)

    def _build_action_card(self, parent: ttk.Frame) -> None:
        frame = self._create_card(parent)

        utilities = ttk.Frame(frame, style="FlatCard.TFrame")
        utilities.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(utilities, text="Load backup file...", style="BoxGhost.TButton", command=self._on_load_backup_file).pack(side=tk.LEFT)
        ttk.Button(utilities, text="Rerun from backup", style="BoxGhost.TButton", command=self._on_rerun_from_backup).pack(side=tk.LEFT, padx=(8, 0))

        bottom = ttk.Frame(frame, style="Card.TFrame")
        bottom.pack(fill=tk.X, pady=(4, 0))
        self.view_btn = ttk.Button(bottom, text="View file location", style="BoxSecondary.TButton", command=self._on_view_location)
        self.view_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Button(bottom, text="Exit Session", style="BoxDanger.TButton", command=self._on_exit).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _attach_var_traces(self) -> None:
        self.output_folder_var.trace_add("write", lambda *_: self._update_ready_state())
        self.output_filename_var.trace_add("write", lambda *_: self._update_ready_state())
        self.power_ratio_var.trace_add("write", lambda *_: self._update_ready_state())

    @staticmethod
    def _normalize_path_text(path_text: str) -> str:
        text = str(path_text).strip().replace("/", "\\")
        if len(text) >= 2 and text[1] == ":":
            while "\\\\" in text:
                text = text.replace("\\\\", "\\")
        return text

    def _set_path_var(self, var: tk.StringVar, path_text: str | Path) -> None:
        var.set(self._normalize_path_text(str(path_text)))

    def _path_from_var(self, var: tk.StringVar) -> Path:
        return Path(self._normalize_path_text(var.get().strip()))

    def _dialog_initial_dir(self, *path_candidates: str) -> str:
        for candidate in path_candidates:
            normalized = self._normalize_path_text(candidate)
            if not normalized:
                continue
            p = Path(normalized)
            if p.is_file():
                return str(p.parent)
            if p.is_dir():
                return str(p)
        return str(Path.cwd())

    def _on_upload_ansys(self) -> None:
        _log.info("User action: upload ANSYS file")
        typed_path = self._normalize_path_text(self.ansys_path_var.get())
        if typed_path:
            source_path = Path(typed_path)
            if not source_path.exists():
                messagebox.showerror("ANSYS file not found", f"Path does not exist:\n{source_path}")
                return
            if source_path.suffix.lower() != ".dat":
                messagebox.showerror("Invalid ANSYS file", "ANSYS input must be a .dat file.")
                return
            self._load_ansys_for_path(source_path, interactive_cache_prompt=True, async_parse=True)
            return

        path = filedialog.askopenfilename(
            initialdir=str(Path.cwd()),
            filetypes=[("ANSYS APDL", "*.dat"), ("All files", "*.*")],
        )
        if not path:
            return
        self._load_ansys_for_path(Path(path), interactive_cache_prompt=True, async_parse=True)

    def _on_upload_spectra(self) -> None:
        _log.info("User action: upload SPECTRA file")
        typed_path = self._normalize_path_text(self.spectra_path_var.get())
        if typed_path:
            source_path = Path(typed_path)
            if not source_path.exists():
                messagebox.showerror("SPECTRA file not found", f"Path does not exist:\n{source_path}")
                return
            allowed = (".dta", ".data", ".dta2")
            if source_path.suffix.lower() not in allowed:
                messagebox.showerror("Invalid SPECTRA file", "SPECTRA input must be a .dta, .data, or .dta2 file.")
                return
            self._load_spectra_for_path(source_path)
            return

        path = filedialog.askopenfilename(
            initialdir=str(Path.cwd()),
            filetypes=[("SPECTRA", "*.dta *.data *.dta2"), ("All files", "*.*")],
        )
        if not path:
            return
        self._load_spectra_for_path(Path(path))

    def _on_browse_output_folder(self) -> None:
        path = filedialog.askdirectory(initialdir=self._dialog_initial_dir(self.output_folder_var.get(), self.ansys_path_var.get()))
        if path:
            self.output_folder_var.set(self._normalize_path_text(path))
            self._update_ready_state()

    def _load_ansys_for_path(self, source_path: Path, interactive_cache_prompt: bool, async_parse: bool = False) -> bool:
        source_path = Path(self._normalize_path_text(str(source_path))).resolve()
        self._set_path_var(self.ansys_path_var, source_path)
        self._set_state_ansys_loading()
        self.ansys_cache_var.set("Cache source: checking...")

        cached_result: AnsysParseResult | None = None
        try:
            cached_result = load_ansys_parse_cache(source_path)
        except Exception:
            cached_result = None

        use_cache = False
        if cached_result is not None:
            if interactive_cache_prompt:
                use_cache = messagebox.askyesno(
                    "Reuse ANSYS cache",
                    "A cached parsed ANSYS model exists for this file.\n"
                    "Do you want to reuse it for faster loading?",
                )
            else:
                use_cache = True

        if use_cache and cached_result is not None:
            self.ansys_result = cached_result
            self.ansys_loaded = True
            self._is_loading_ansys = False
            self._set_ansys_stats_from_result(self.ansys_result)
            self.ansys_cache_var.set("Cache source: loaded from cache")
            self._update_ansys_progress(
                100,
                f"Loaded cache: nodes={len(self.ansys_result.node_store.node_ids)}, "
                f"elements={self.ansys_result.total_elements}, "
                f"flux={len(self.ansys_result.heatflux_elements)}",
            )
            if not self.output_folder_var.get().strip():
                self.output_folder_var.set(self._normalize_path_text(str(source_path.parent)))
            self.upload_ansys_btn.configure(state=tk.NORMAL)
            if self._can_start_spectra():
                self.upload_spectra_btn.configure(state=tk.NORMAL)
            self._save_session_backup()
            self._update_ready_state()
            return True

        self._update_ansys_progress(0, "Reading ANSYS file...")
        if async_parse:
            self._start_ansys_parse_worker(source_path)
            return True

        try:
            self.ansys_result = read_ansys_file(source_path, progress_cb=self._on_ansys_parse_progress)
        except Exception as exc:
            self._handle_ansys_parse_failure(str(exc))
            return False

        self._handle_ansys_parse_success(source_path, self.ansys_result)
        return True

    def _start_ansys_parse_worker(self, source_path: Path) -> None:
        def worker() -> None:
            last_percent = -1
            last_stage = ""
            last_emit_ts = 0.0

            def progress_cb(current: int, total: int, stage: str) -> None:
                nonlocal last_percent, last_stage, last_emit_ts
                percent = 0 if total <= 0 else int(round((current / total) * 100))
                percent = max(0, min(percent, 100))
                now = time.monotonic()
                should_emit = (
                    percent == 100
                    or percent != last_percent
                    or stage != last_stage
                    or (now - last_emit_ts) >= 0.2
                )
                if not should_emit:
                    return
                last_percent = percent
                last_stage = stage
                last_emit_ts = now
                self.root.after(0, lambda p=percent, s=stage: self._update_ansys_progress(p, s))

            try:
                result = read_ansys_file(source_path, progress_cb=progress_cb)
            except Exception as exc:
                self.root.after(0, lambda err=str(exc): self._handle_ansys_parse_failure(err))
                return

            self.root.after(0, lambda parsed=result: self._handle_ansys_parse_success(source_path, parsed))

        threading.Thread(target=worker, daemon=True).start()

    def _handle_ansys_parse_failure(self, error_text: str) -> None:
        self.ansys_loaded = False
        self.ansys_result = None
        self._set_ansys_stats_from_result(None)
        self.ansys_cache_var.set("Cache source: parse failed")
        self._update_ansys_progress(0, "ANSYS parse failed")
        self._set_state_idle()
        messagebox.showerror("ANSYS Parse Error", error_text)

    def _handle_ansys_parse_success(self, source_path: Path, result: AnsysParseResult) -> None:
        self.ansys_result = result
        self.ansys_loaded = True
        self._set_ansys_stats_from_result(result)
        self.ansys_cache_var.set("Cache source: parsed from file")
        self._update_ansys_progress(
            100,
            f"Parsed nodes={len(result.node_store.node_ids)}, "
            f"elements={result.total_elements}, "
            f"flux={len(result.heatflux_elements)}",
        )
        if not self.output_folder_var.get().strip():
            self.output_folder_var.set(self._normalize_path_text(str(source_path.parent)))
        self._is_loading_ansys = False
        self.upload_ansys_btn.configure(state=tk.NORMAL)
        if self._can_start_spectra():
            self.upload_spectra_btn.configure(state=tk.NORMAL)
        self._save_ansys_cache_async(source_path, result)
        self._save_session_backup()
        self._update_ready_state()

    def _load_spectra_for_path(self, spectra_path: Path) -> bool:
        spectra_path = Path(self._normalize_path_text(str(spectra_path))).resolve()
        self._set_path_var(self.spectra_path_var, spectra_path)
        self._set_state_spectra_loading()
        self._update_spectra_progress(0, "Reading SPECTRA file...")
        try:
            self.spectra_result = read_spectra_file(spectra_path, progress_cb=self._on_spectra_parse_progress)
        except Exception as exc:
            self._set_state_idle()
            self.spectra_loaded = False
            self.spectra_result = None
            self._set_spectra_stats_from_result(None)
            self._update_spectra_progress(0, "SPECTRA parse failed")
            messagebox.showerror("SPECTRA Parse Error", str(exc))
            return False

        self.spectra_loaded = True
        self._set_spectra_stats_from_result(self.spectra_result)
        self._update_spectra_progress(
            100,
            f"Parsed grid {self.spectra_result.n_col}x{self.spectra_result.n_row}, "
            f"peak={self.spectra_result.peak_power_density_kw_mrad2:.4g} kW/mrad^2",
        )
        self.total_power_spectra_var.set(f"{self.spectra_result.total_power_kw:.3f} kW")
        self._is_loading_spectra = False
        self.upload_spectra_btn.configure(state=tk.NORMAL)
        self.upload_ansys_btn.configure(state=tk.NORMAL)
        self._save_session_backup()
        self._update_ready_state()
        return True

    def _validate_geometry(self) -> bool:
        try:
            source, target, horizontal = self.geometry_frame.get_points()
            _ = build_source_geometry(source=source, target=target, horizontal_point=horizontal)
        except ValueError:
            return False
        return True

    def _is_ratio_valid(self) -> bool:
        try:
            float(self.power_ratio_var.get())
        except ValueError:
            return False
        return True

    def _update_ready_state(self) -> None:
        self.geometry_valid = self._validate_geometry()
        output_valid = bool(self.output_folder_var.get().strip()) and bool(self.output_filename_var.get().strip())
        ready = (
            self.ansys_loaded
            and self.spectra_loaded
            and self.geometry_valid
            and output_valid
            and self._is_ratio_valid()
            and not self._is_loading_ansys
            and not self._is_loading_spectra
            and not self._is_mapping
        )
        self.create_btn.configure(state=tk.NORMAL if ready else tk.DISABLED)

    def _set_state_idle(self) -> None:
        self._is_loading_ansys = False
        self._is_loading_spectra = False
        self._is_mapping = False
        self.create_btn.configure(state=tk.DISABLED)
        self.view_btn.configure(state=tk.DISABLED)
        self.view_btn.configure(style="Disabled.TButton")
        self.upload_ansys_btn.configure(state=tk.NORMAL)
        self.upload_spectra_btn.configure(state=tk.NORMAL if self.ansys_loaded else tk.DISABLED)
        self._set_edit_controls_enabled(True)
        self.footer_status_var.set("SYSTEM OPERATIONAL")

    def _set_state_ansys_loading(self) -> None:
        self._is_loading_ansys = True
        self._is_loading_spectra = False
        self._is_mapping = False
        self.upload_ansys_btn.configure(state=tk.DISABLED)
        self.upload_spectra_btn.configure(state=tk.DISABLED)
        self.create_btn.configure(state=tk.DISABLED)
        self.footer_status_var.set("PARSING ANSYS MODEL")

    def _set_state_spectra_loading(self) -> None:
        self._is_loading_ansys = False
        self._is_loading_spectra = True
        self._is_mapping = False
        self.upload_spectra_btn.configure(state=tk.DISABLED)
        self.upload_ansys_btn.configure(state=tk.DISABLED)
        self.create_btn.configure(state=tk.DISABLED)
        self.footer_status_var.set("PARSING SPECTRA GRID")

    def _set_state_mapping(self) -> None:
        self._is_loading_ansys = False
        self._is_loading_spectra = False
        self._is_mapping = True
        self.upload_spectra_btn.configure(state=tk.DISABLED)
        self.upload_ansys_btn.configure(state=tk.DISABLED)
        self.create_btn.configure(state=tk.DISABLED)
        self.view_btn.configure(state=tk.DISABLED)
        self._set_edit_controls_enabled(False)
        self.footer_status_var.set("MAPPING IN PROGRESS")
        self.mapping_progress["value"] = 0
        self.mapping_percent_var.set("0%")

    def _set_state_done(self) -> None:
        self._is_loading_ansys = False
        self._is_loading_spectra = False
        self._is_mapping = False
        self.upload_spectra_btn.configure(state=tk.NORMAL)
        self.upload_ansys_btn.configure(state=tk.NORMAL)
        self.view_btn.configure(state=tk.NORMAL if self.last_output_path is not None else tk.DISABLED)
        self.view_btn.configure(style="Secondary.TButton" if self.last_output_path is not None else "Disabled.TButton")
        self._set_edit_controls_enabled(True)
        self.footer_status_var.set("SYSTEM OPERATIONAL")
        self._update_ready_state()

    def _sync_footer_state(self) -> None:
        if self._is_mapping:
            self.footer_status_var.set("MAPPING IN PROGRESS")
            return
        if self._is_loading_ansys:
            self.footer_status_var.set("PARSING ANSYS MODEL")
            return
        if self._is_loading_spectra:
            self.footer_status_var.set("PARSING SPECTRA GRID")
            return
        self.footer_status_var.set("SYSTEM OPERATIONAL")

    def _post_warning(self, msg: str, duration_ms: int = 12000) -> None:
        """Show a non-modal warning in the footer strip; auto-clears after duration_ms."""
        if self._warn_after_id is not None:
            self.root.after_cancel(self._warn_after_id)
        self.warn_strip_var.set(msg)
        _log.warning("UI warning posted: %s", msg)
        self._warn_after_id = self.root.after(duration_ms, self._clear_warning)

    def _clear_warning(self) -> None:
        self.warn_strip_var.set("")
        self._warn_after_id = None

    def _save_ansys_cache_async(self, source_path: Path, result: AnsysParseResult) -> None:
        self.footer_status_var.set("SAVING ANSYS CACHE")

        def worker() -> None:
            ok = True
            try:
                save_ansys_parse_cache(source_path, result)
            except Exception:
                ok = False

            def finalize() -> None:
                if ok:
                    self.ansys_cache_var.set("Cache source: parsed from file (cache saved)")
                else:
                    self.ansys_cache_var.set("Cache source: parsed from file (cache save failed)")
                self._sync_footer_state()

            self.root.after(0, finalize)

        threading.Thread(target=worker, daemon=True).start()


    def _save_spectra_cache_async(self, source_path: Path, result: SpectraParseResult) -> None:
        self.footer_status_var.set("SAVING SPECTRA CACHE")

        def worker() -> None:
            ok = True
            try:
                save_spectra_parse_cache(source_path, result)
            except Exception:
                ok = False

            def finalize() -> None:
                if ok:
                    self.spectra_cache_var.set("Cache source: parsed from file (cache saved)")
                else:
                    self.spectra_cache_var.set("Cache source: parsed from file (cache save failed)")
                self._sync_footer_state()

            self.root.after(0, finalize)

        threading.Thread(target=worker, daemon=True).start()
    def _set_edit_controls_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        readonly_state = "normal" if enabled else "disabled"
        widgets: list[ttk.Widget] = [
            self.output_folder_entry,
            self.output_filename_entry,
            self.power_ratio_entry,
            self.browse_output_btn,
            self.update_geometry_btn,
        ]
        for widget in widgets:
            try:
                widget.configure(state=state)
            except Exception:
                pass
        try:
            self.geometry_frame.set_inputs_state(readonly_state)
        except Exception:
            pass

    def _can_start_spectra(self) -> bool:
        return self.ansys_loaded and not self._is_loading_ansys

    def _on_update_geometry(self) -> None:
        self._update_ready_state()
        if self.geometry_valid:
            messagebox.showinfo("Geometry", "The geometry is updated.")
        else:
            messagebox.showerror("Geometry", "Invalid geometry. Check Source/Target/Horizontal points.")

    def _on_create(self) -> None:
        if self.ansys_result is None or self.spectra_result is None:
            messagebox.showerror("Missing Data", "Please upload ANSYS and SPECTRA files first.")
            return
        try:
            source, target, horizontal = self.geometry_frame.get_points()
            geometry = build_source_geometry(source=source, target=target, horizontal_point=horizontal)
        except ValueError as exc:
            messagebox.showerror("Invalid Geometry", str(exc))
            return

        try:
            total_power_ratio = float(self.power_ratio_var.get())
        except ValueError:
            messagebox.showerror("Invalid Ratio", "Power ratio must be a valid number.")
            return

        output_dir_text = self.output_folder_var.get().strip()
        output_name = self.output_filename_var.get().strip()
        if not output_dir_text or not output_name:
            messagebox.showerror("Invalid Output", "Output folder and filename are required.")
            return
        output_dir = Path(output_dir_text)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_name

        _log.info(
            "User action: start mapping — output=%s ratio=%.4g", output_path, total_power_ratio
        )
        self._set_state_mapping()
        try:
            mapped = run_mapping(
                hf_elements=self.ansys_result.heatflux_elements,
                spectra_elements=self.spectra_result.elements,
                grid=self.spectra_result.grid,
                geometry=geometry,
                source=source,
                progress_cb=self._on_mapping_progress,
                vectorized=True,
            )
            write_output_from_elements(output_path, mapped, total_power_ratio=total_power_ratio)
            self.last_output_path = output_path
            total_power_out_kw = sum(elem.total_power_w for elem in mapped) / 1000.0
            mapped_count = sum(1 for elem in mapped if float(elem.metadata.get("in_grid", 0.0)) > 0.5)
            out_of_grid_count = len(mapped) - mapped_count
            self.mapping_percent_var.set("100%")
            self.mapping_progress["value"] = 100
            self.output_elements_var.set(str(len(mapped)))
            self.mapped_elements_var.set(str(mapped_count))
            self.out_of_grid_var.set(str(out_of_grid_count))
            self.total_power_out_var.set(f"{total_power_out_kw:.3f} kW")
            _log.info(
                "Mapping UI complete: %d elements, %d out-of-grid, total power out=%.4g kW",
                len(mapped), out_of_grid_count, total_power_out_kw,
            )
            if len(mapped) > 0 and out_of_grid_count / len(mapped) > 0.05:
                self._post_warning(
                    f"WARNING: {out_of_grid_count}/{len(mapped)} elements outside SPECTRA grid"
                    f" ({100.0 * out_of_grid_count / len(mapped):.1f}%) — see heatflux.log"
                )
            self._save_session_backup()
            messagebox.showinfo(
                "Completed",
                "Heat flux file created.\n"
                f"Output: {output_path}\n"
                f"Elements: {len(mapped)}\n"
                f"Total power out: {total_power_out_kw:.3f} kW",
            )
        except Exception as exc:
            _log.error("Mapping failed: %s", exc, exc_info=True)
            messagebox.showerror("Mapping Error", str(exc))
        finally:
            self._set_state_done()

    def _on_view_location(self) -> None:
        if self.last_output_path is None:
            messagebox.showinfo("Not available", "Output location action will be enabled after first successful export.")
            return
        target = self.last_output_path
        if target.exists():
            try:
                os.startfile(str(target))  # type: ignore[attr-defined]
                return
            except Exception:
                pass
        parent = target.parent
        if parent.exists():
            try:
                os.startfile(str(parent))  # type: ignore[attr-defined]
                return
            except Exception:
                pass
        messagebox.showinfo("Output location", str(target))

    def _update_ansys_progress(self, percent: int, status: str) -> None:
        self.ansys_status_var.set(status)
        self._update_ansys_live_counts_from_status(status)
        self.ansys_percent_var.set(f"{percent}%")
        self.ansys_progress["value"] = percent
        self.root.update_idletasks()

    def _update_spectra_progress(self, percent: int, status: str) -> None:
        self.spectra_status_var.set(status)
        self.spectra_percent_var.set(f"{percent}%")
        self.spectra_progress["value"] = percent
        self.root.update_idletasks()

    def _on_ansys_parse_progress(self, current: int, total: int, stage: str) -> None:
        percent = 0 if total <= 0 else int(round((current / total) * 100))
        percent = max(0, min(percent, 100))
        self._update_ansys_progress(percent, stage)

    def _on_spectra_parse_progress(self, current: int, total: int, stage: str) -> None:
        percent = 0 if total <= 0 else int(round((current / total) * 100))
        percent = max(0, min(percent, 100))
        self._update_spectra_progress(percent, stage)

    def _on_mapping_progress(self, current: int, total: int, stage: str) -> None:
        del stage
        percent = 0 if total <= 0 else int(round((current / total) * 100))
        percent = max(0, min(percent, 100))
        self.mapping_percent_var.set(f"{percent}%")
        self.mapping_progress["value"] = percent
        self.root.update_idletasks()

    @staticmethod
    def _fmt_int(value: int) -> str:
        return f"{value:,}"

    def _set_ansys_stats_from_result(self, result: AnsysParseResult | None) -> None:
        if result is None:
            self.ansys_total_nodes_var.set("0")
            self.ansys_total_elements_var.set("0")
            self.ansys_flux_elements_var.set("0")
            return
        self.ansys_total_nodes_var.set(self._fmt_int(len(result.node_store.node_ids)))
        self.ansys_total_elements_var.set(self._fmt_int(result.total_elements))
        self.ansys_flux_elements_var.set(self._fmt_int(len(result.heatflux_elements)))

    def _reset_startup_ansys_counters_if_unloaded(self) -> None:
        if self.ansys_result is not None or self.ansys_loaded:
            return
        self.ansys_total_nodes_var.set("0")
        self.ansys_total_elements_var.set("0")
        self.ansys_flux_elements_var.set("0")

    def _reset_startup_mapping_summary(self) -> None:
        self.output_elements_var.set("0")
        self.mapped_elements_var.set("0")
        self.out_of_grid_var.set("0")
        self.total_power_out_var.set("0.000 kW")
        self.total_power_spectra_var.set("0.000 kW")

    def _update_ansys_live_counts_from_status(self, status: str) -> None:
        if not status:
            return
        for key, raw_val in self._COUNT_PATTERN.findall(status):
            value = raw_val.replace(",", "")
            if not value.isdigit():
                continue
            pretty = self._fmt_int(int(value))
            key_l = key.lower()
            if key_l == "nodes":
                self.ansys_total_nodes_var.set(pretty)
            elif key_l == "elements":
                self.ansys_total_elements_var.set(pretty)
            elif key_l == "flux":
                self.ansys_flux_elements_var.set(pretty)

    def _set_spectra_stats_from_result(self, result: SpectraParseResult | None) -> None:
        if result is None:
            self.spectra_cols_var.set("-")
            self.spectra_rows_var.set("-")
            self.spectra_x_range_var.set("-")
            self.spectra_y_range_var.set("-")
            self.spectra_peak_var.set("-")
            self.spectra_total_power_var.set("-")
            return

        self.spectra_cols_var.set(self._fmt_int(result.n_col))
        self.spectra_rows_var.set(self._fmt_int(result.n_row))
        self.spectra_x_range_var.set(f"{result.x_min:.3f} -> {result.x_max:.3f}")
        self.spectra_y_range_var.set(f"{result.y_min:.3f} -> {result.y_max:.3f}")
        self.spectra_peak_var.set(f"{result.peak_power_density_kw_mrad2:.3f} kW/mrad^2")
        self.spectra_total_power_var.set(f"{result.total_power_kw:.3f} kW")

    @staticmethod
    def _parse_numeric_text(text: str) -> float:
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text))
        if not match:
            return 0.0
        try:
            return float(match.group(0))
        except ValueError:
            return 0.0

    def _build_backup_payload(self) -> dict[str, object]:
        source, target, horizontal = self.geometry_frame.get_points()
        try:
            ratio_value = round(float(self.power_ratio_var.get().strip()), 1)
        except ValueError:
            ratio_value = 1.0

        spectra_peak = 0.0 if self.spectra_result is None else round(float(self.spectra_result.peak_power_density_kw_mrad2), 3)
        spectra_total_kw = 0.0 if self.spectra_result is None else round(float(self.spectra_result.total_power_kw), 3)
        spectra_min_x = 0.0 if self.spectra_result is None else round(float(self.spectra_result.x_min), 3)
        spectra_max_x = 0.0 if self.spectra_result is None else round(float(self.spectra_result.x_max), 3)
        spectra_min_y = 0.0 if self.spectra_result is None else round(float(self.spectra_result.y_min), 3)
        spectra_max_y = 0.0 if self.spectra_result is None else round(float(self.spectra_result.y_max), 3)
        total_power_out_kw = round(self._parse_numeric_text(self.total_power_out_var.get()), 3)

        return {
            "ansys_file": self._normalize_path_text(self.ansys_path_var.get().strip()),
            "spectra_file": self._normalize_path_text(self.spectra_path_var.get().strip()),
            "geometry": {
                "source": source.tolist(),
                "target": target.tolist(),
                "horizontal": horizontal.tolist(),
            },
            "output": {
                "directory": self._normalize_path_text(self.output_folder_var.get().strip()),
                "filename": self.output_filename_var.get().strip(),
                "total_power_ratio": ratio_value,
            },
            "stats": {
                "ansys_nodes": 0 if self.ansys_result is None else len(self.ansys_result.node_store.node_ids),
                "ansys_elements": 0 if self.ansys_result is None else self.ansys_result.total_elements,
                "ansys_heatflux_elements": 0 if self.ansys_result is None else len(self.ansys_result.heatflux_elements),
                "spectra_rows": 0 if self.spectra_result is None else self.spectra_result.n_row,
                "spectra_cols": 0 if self.spectra_result is None else self.spectra_result.n_col,
                "spectra_peak_kw_mrad2": spectra_peak,
                "spectra_total_power_kw": spectra_total_kw,
                "spectra_min_x_mrad": spectra_min_x,
                "spectra_max_x_mrad": spectra_max_x,
                "spectra_min_y_mrad": spectra_min_y,
                "spectra_max_y_mrad": spectra_max_y,
                "output_elements": self.output_elements_var.get(),
                "mapped_elements": self.mapped_elements_var.get(),
                "out_of_grid_elements": self.out_of_grid_var.get(),
                "total_power_out_kw": total_power_out_kw,
            },
            "last_output_path": "" if self.last_output_path is None else str(self.last_output_path),
        }

    def _save_session_backup(self) -> None:
        try:
            self.loaded_backup_path = save_session_backup(self._build_backup_payload())
        except Exception:
            pass

    def _restore_session_from_backup(self) -> None:
        backup = load_session_backup()
        if not backup:
            return
        self._apply_backup_payload(backup)

    def _apply_backup_payload(self, backup: dict[str, object]) -> None:
        self._set_path_var(self.ansys_path_var, str(backup.get("ansys_file", "") or ""))
        self._set_path_var(self.spectra_path_var, str(backup.get("spectra_file", "") or ""))
        output = backup.get("output", {})
        if isinstance(output, dict):
            self.output_folder_var.set(self._normalize_path_text(str(output.get("directory", "") or "")))
            self.output_filename_var.set(str(output.get("filename", self.output_filename_var.get()) or self.output_filename_var.get()))
            ratio = output.get("total_power_ratio", self.power_ratio_var.get())
            self.power_ratio_var.set(str(ratio))
        geometry = backup.get("geometry", {})
        if isinstance(geometry, dict):
            try:
                source = geometry.get("source", [0.0, 0.0, 0.0])
                target = geometry.get("target", [0.0, 0.0, 15000.0])
                horizontal = geometry.get("horizontal", [100.0, 0.0, 0.0])
                self.geometry_frame.set_points(source=np.array(source), target=np.array(target), horizontal=np.array(horizontal))
            except Exception:
                pass
        stats = backup.get("stats", {})
        if isinstance(stats, dict):
            def _safe_int_text(value: object) -> str:
                try:
                    return self._fmt_int(int(float(str(value))))
                except Exception:
                    return "-"

            self.output_elements_var.set(str(stats.get("output_elements", "-")))
            self.mapped_elements_var.set(str(stats.get("mapped_elements", "-")))
            self.out_of_grid_var.set(str(stats.get("out_of_grid_elements", "-")))
            out_kw = self._parse_numeric_text(str(stats.get("total_power_out_kw", "0.0")))
            self.total_power_out_var.set(f"{out_kw:.3f} kW")
            spectra_total = stats.get("spectra_total_power_kw", "-")
            spectra_kw = self._parse_numeric_text(str(spectra_total))
            self.total_power_spectra_var.set(f"{spectra_kw:.3f} kW")
            ansys_nodes = stats.get("ansys_nodes")
            ansys_elements = stats.get("ansys_elements")
            ansys_flux = stats.get("ansys_heatflux_elements")
            self.ansys_total_nodes_var.set(_safe_int_text(ansys_nodes))
            self.ansys_total_elements_var.set(_safe_int_text(ansys_elements))
            self.ansys_flux_elements_var.set(_safe_int_text(ansys_flux))

            spectra_rows = stats.get("spectra_rows")
            spectra_cols = stats.get("spectra_cols")
            self.spectra_rows_var.set(_safe_int_text(spectra_rows))
            self.spectra_cols_var.set(_safe_int_text(spectra_cols))

            min_x = stats.get("spectra_min_x_mrad")
            max_x = stats.get("spectra_max_x_mrad")
            min_y = stats.get("spectra_min_y_mrad")
            max_y = stats.get("spectra_max_y_mrad")
            peak = stats.get("spectra_peak_kw_mrad2")
            total_kw = stats.get("spectra_total_power_kw")
            if min_x is not None and max_x is not None:
                self.spectra_x_range_var.set(f"{float(min_x):.3f} -> {float(max_x):.3f}")
            if min_y is not None and max_y is not None:
                self.spectra_y_range_var.set(f"{float(min_y):.3f} -> {float(max_y):.3f}")
            if peak is not None:
                self.spectra_peak_var.set(f"{float(peak):.3f} kW/mrad^2")
            if total_kw is not None:
                self.spectra_total_power_var.set(f"{float(total_kw):.3f} kW")
        last_output = str(backup.get("last_output_path", "") or "")
        self.last_output_path = Path(last_output) if last_output else None
        self._update_ready_state()

    def _on_load_backup_file(self) -> None:
        backup_dir = Path("session_backups")
        kwargs: dict[str, object] = {"filetypes": [("JSON backups", "*.json"), ("All files", "*.*")]}
        if backup_dir.exists():
            kwargs["initialdir"] = str(backup_dir.resolve())
        path = filedialog.askopenfilename(**kwargs)
        if not path:
            return
        try:
            backup = load_session_backup(Path(path))
        except Exception as exc:
            messagebox.showerror("Load backup failed", str(exc))
            return
        if not backup:
            messagebox.showerror("Load backup failed", "Backup file is empty or invalid.")
            return
        self.loaded_backup_path = Path(path)
        self._apply_backup_payload(backup)
        messagebox.showinfo("Backup loaded", f"Loaded: {path}")

    def _on_rerun_from_backup(self) -> None:
        if self.loaded_backup_path is None:
            self._on_load_backup_file()
            if self.loaded_backup_path is None:
                return

        ansys_path = self._path_from_var(self.ansys_path_var)
        spectra_path = self._path_from_var(self.spectra_path_var)
        if not ansys_path.exists():
            messagebox.showerror("Rerun failed", f"ANSYS file not found:\n{ansys_path}")
            return
        if not spectra_path.exists():
            messagebox.showerror("Rerun failed", f"SPECTRA file not found:\n{spectra_path}")
            return

        if not self._load_ansys_for_path(ansys_path, interactive_cache_prompt=False):
            return
        if not self._load_spectra_for_path(spectra_path):
            return
        self._on_create()

    def _on_exit(self) -> None:
        self._save_session_backup()
        self.root.destroy()

    def _get_current_ansys_path(self) -> Path | None:
        p = self._normalize_path_text(self.ansys_path_var.get().strip())
        if not p:
            return None
        path = Path(p)
        return path if path.exists() else None

    def _get_current_spectra_path(self) -> Path | None:
        p = self._normalize_path_text(self.spectra_path_var.get().strip())
        if not p:
            return None
        path = Path(p)
        return path if path.exists() else None
    def _on_delete_current_ansys_cache(self) -> None:
        source_path = self._get_current_ansys_path()
        if source_path is None:
            messagebox.showinfo("Cache", "No valid ANSYS file path is set.")
            return
        if not has_valid_ansys_parse_cache(source_path):
            messagebox.showinfo("Cache", "No valid cache found for the current ANSYS file.")
            return
        confirmed = messagebox.askyesno("Delete cache", "Delete cached parsed model for this ANSYS file?")
        if not confirmed:
            return
        deleted = delete_ansys_parse_cache(source_path)
        if deleted:
            self.ansys_cache_var.set("Cache source: cache deleted for current file")
            messagebox.showinfo("Cache", "ANSYS cache deleted for current file.")
        else:
            messagebox.showinfo("Cache", "No cache file was deleted.")

    def _on_clear_all_ansys_cache(self) -> None:
        confirmed = messagebox.askyesno("Clear all cache", "Delete all ANSYS cache entries?")
        if not confirmed:
            return
        deleted = clear_all_ansys_parse_cache()
        self.ansys_cache_var.set("Cache source: all cache cleared")
        messagebox.showinfo("Cache", f"Deleted {deleted} cache file(s).")
        self._refresh_cache_browser()


    def _on_delete_current_spectra_cache(self) -> None:
        source_path = self._get_current_spectra_path()
        if source_path is None:
            messagebox.showinfo("Cache", "No valid SPECTRA file path is set.")
            return
        if not has_valid_spectra_parse_cache(source_path):
            messagebox.showinfo("Cache", "No valid cache found for the current SPECTRA file.")
            return
        confirmed = messagebox.askyesno("Delete cache", "Delete cached parsed model for this SPECTRA file?")
        if not confirmed:
            return
        deleted = delete_spectra_parse_cache(source_path)
        if deleted:
            self.spectra_cache_var.set("Cache source: cache deleted for current file")
            messagebox.showinfo("Cache", "SPECTRA cache deleted for current file.")
        else:
            messagebox.showinfo("Cache", "No cache file was deleted.")

    def _on_clear_all_spectra_cache(self) -> None:
        confirmed = messagebox.askyesno("Clear all cache", "Delete all SPECTRA cache entries?")
        if not confirmed:
            return
        deleted = clear_all_spectra_parse_cache()
        self.spectra_cache_var.set("Cache source: all cache cleared")
        messagebox.showinfo("Cache", f"Deleted {deleted} cache file(s).")
    def _open_cache_browser(self) -> None:
        if self.cache_browser_window is not None and self.cache_browser_window.winfo_exists():
            self.cache_browser_window.lift()
            self._refresh_cache_browser()
            return

        w = tk.Toplevel(self.root)
        w.title("ANSYS Cache Browser")
        w.geometry("900x320")
        self.cache_browser_window = w

        cols = ("source", "valid", "exists", "size_mb", "cache_file")
        tree = ttk.Treeview(w, columns=cols, show="headings", height=10)
        tree.heading("source", text="Source Path")
        tree.heading("valid", text="Valid")
        tree.heading("exists", text="Source Exists")
        tree.heading("size_mb", text="Size (MB)")
        tree.heading("cache_file", text="Cache File")
        tree.column("source", width=360, anchor="w")
        tree.column("valid", width=60, anchor="center")
        tree.column("exists", width=90, anchor="center")
        tree.column("size_mb", width=80, anchor="e")
        tree.column("cache_file", width=260, anchor="w")
        tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.cache_tree = tree

        actions = ttk.Frame(w)
        actions.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(actions, text="Refresh", command=self._refresh_cache_browser).pack(side=tk.LEFT)
        ttk.Button(actions, text="Delete Selected", command=self._delete_selected_cache_entry).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Delete All", command=self._on_clear_all_ansys_cache).pack(side=tk.LEFT, padx=(8, 0))

        self._refresh_cache_browser()

    def _refresh_cache_browser(self) -> None:
        if self.cache_tree is None or not self.cache_tree.winfo_exists():
            return
        for item in self.cache_tree.get_children():
            self.cache_tree.delete(item)

        entries = list_ansys_parse_cache_entries()
        for entry in entries:
            size_mb = entry.source_size / (1024.0 * 1024.0)
            self.cache_tree.insert(
                "",
                tk.END,
                iid=str(entry.cache_path),
                values=(
                    str(entry.source_path),
                    "Y" if entry.is_valid else "N",
                    "Y" if entry.source_exists else "N",
                    f"{size_mb:.2f}",
                    str(entry.cache_path.name),
                ),
            )

    def _delete_selected_cache_entry(self) -> None:
        if self.cache_tree is None or not self.cache_tree.winfo_exists():
            return
        selection = self.cache_tree.selection()
        if not selection:
            messagebox.showinfo("Cache", "No cache entry selected.")
            return
        if not messagebox.askyesno("Delete cache entry", "Delete selected cache entry?"):
            return
        deleted = 0
        for iid in selection:
            if delete_ansys_cache_entry_file(Path(iid)):
                deleted += 1
        messagebox.showinfo("Cache", f"Deleted {deleted} cache file(s).")
        self._refresh_cache_browser()

    def _on_run_sample_validation(self) -> None:
        try:
            result = run_default_sample_validation(Path(__file__).resolve().parents[2])
        except Exception as exc:
            messagebox.showerror("Sample validation failed", str(exc))
            return
        status = "exact match" if result.exact_match_first5 else "difference found"
        messagebox.showinfo(
            "Sample validation complete",
            f"Status: {status}\nReport: {result.report_path}\nOutput: {result.generated_output_path}",
        )





