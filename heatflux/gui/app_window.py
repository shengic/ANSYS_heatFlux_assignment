from __future__ import annotations

import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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
from heatflux.gui.geometry_frame import GeometryFrame
from heatflux.io.ansys_reader import AnsysParseResult, read_ansys_file
from heatflux.io.output_writer import write_output_from_elements
from heatflux.io.spectra_reader import SpectraParseResult, read_spectra_file
from heatflux.math_core.geometry import build_source_geometry
from heatflux.pipeline.mapping_pipeline import run_mapping
from heatflux.tools.sample_validation import run_default_sample_validation


class MainWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SPECTRA -> ANSYS Heat Flux Node Assignment")
        self.root.geometry("1400x900")

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
        self.spectra_status_var = tk.StringVar(value="No file loaded")
        self.spectra_percent_var = tk.StringVar(value="0%")
        self.mapping_percent_var = tk.StringVar(value="0%")
        self.output_elements_var = tk.StringVar(value="-")
        self.mapped_elements_var = tk.StringVar(value="-")
        self.out_of_grid_var = tk.StringVar(value="-")
        self.total_power_out_var = tk.StringVar(value="-")
        self.total_power_spectra_var = tk.StringVar(value="-")
        self.cache_browser_window: tk.Toplevel | None = None
        self.cache_tree: ttk.Treeview | None = None
        self.loaded_backup_path: Path | None = None

        self._build_layout()
        self._attach_var_traces()
        self.root.protocol("WM_DELETE_WINDOW", self._on_exit)
        self._restore_session_from_backup()
        self._set_state_idle()

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(container)
        right = ttk.Frame(container)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right.grid(row=0, column=1, sticky="nsew")
        container.columnconfigure(0, weight=2)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        self._build_ansys_card(left)
        self._build_spectra_card(left)
        self.geometry_frame = GeometryFrame(left)
        self.geometry_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(left, text="Update geometry", command=self._on_update_geometry).pack(fill=tk.X, pady=(0, 8))

        self._build_output_card(right)
        self._build_mapping_card(right)
        self._build_action_card(right)

    def _build_ansys_card(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="ANSYS APDL INPUT FILE")
        frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Entry(frame, textvariable=self.ansys_path_var).grid(row=0, column=0, padx=8, pady=8, sticky="ew")
        self.upload_ansys_btn = ttk.Button(frame, text="Upload ANSYS file", command=self._on_upload_ansys)
        self.upload_ansys_btn.grid(row=0, column=1, padx=8, pady=8)
        ttk.Label(frame, textvariable=self.ansys_status_var).grid(row=1, column=0, padx=8, sticky="w")
        ttk.Label(frame, textvariable=self.ansys_percent_var).grid(row=1, column=1, padx=8, sticky="e")
        self.ansys_progress = ttk.Progressbar(frame, orient="horizontal", mode="determinate", maximum=100)
        self.ansys_progress.grid(row=2, column=0, columnspan=2, padx=8, pady=(4, 8), sticky="ew")
        ttk.Label(frame, textvariable=self.ansys_cache_var).grid(row=3, column=0, columnspan=2, padx=8, pady=(0, 8), sticky="w")
        ttk.Button(frame, text="Delete cache for this file", command=self._on_delete_current_ansys_cache).grid(
            row=4, column=0, padx=8, pady=(0, 8), sticky="w"
        )
        ttk.Button(frame, text="Delete all caches", command=self._on_clear_all_ansys_cache).grid(
            row=4, column=1, padx=8, pady=(0, 8), sticky="e"
        )
        ttk.Button(frame, text="Cache browser...", command=self._open_cache_browser).grid(
            row=5, column=0, columnspan=2, padx=8, pady=(0, 8), sticky="w"
        )
        frame.columnconfigure(0, weight=1)

    def _build_spectra_card(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="SPECTRA POWER DENSITY FILE")
        frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Entry(frame, textvariable=self.spectra_path_var).grid(row=0, column=0, padx=8, pady=8, sticky="ew")
        self.upload_spectra_btn = ttk.Button(frame, text="Upload SPECTRA file", command=self._on_upload_spectra)
        self.upload_spectra_btn.grid(row=0, column=1, padx=8, pady=8)
        ttk.Label(frame, textvariable=self.spectra_status_var).grid(row=1, column=0, padx=8, sticky="w")
        ttk.Label(frame, textvariable=self.spectra_percent_var).grid(row=1, column=1, padx=8, sticky="e")
        self.spectra_progress = ttk.Progressbar(frame, orient="horizontal", mode="determinate", maximum=100)
        self.spectra_progress.grid(row=2, column=0, columnspan=2, padx=8, pady=(4, 8), sticky="ew")
        frame.columnconfigure(0, weight=1)

    def _build_output_card(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="OUTPUT - ANSYS EXTERNAL DATA FILE")
        frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(frame, text="Output folder").grid(row=0, column=0, padx=8, pady=(8, 2), sticky="w")
        ttk.Entry(frame, textvariable=self.output_folder_var).grid(row=1, column=0, padx=8, pady=2, sticky="ew")
        ttk.Button(frame, text="Browse...", command=self._on_browse_output_folder).grid(row=1, column=1, padx=8, pady=2)
        ttk.Label(frame, text="Output filename").grid(row=2, column=0, padx=8, pady=(8, 2), sticky="w")
        ttk.Entry(frame, textvariable=self.output_filename_var).grid(row=3, column=0, columnspan=2, padx=8, pady=2, sticky="ew")
        ttk.Label(frame, text="Power ratio").grid(row=4, column=0, padx=8, pady=(8, 2), sticky="w")
        ttk.Entry(frame, textvariable=self.power_ratio_var, width=10).grid(row=5, column=0, padx=8, pady=(0, 8), sticky="w")
        frame.columnconfigure(0, weight=1)

    def _build_mapping_card(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="MAPPING PROGRESS")
        frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(frame, textvariable=self.mapping_percent_var).grid(row=0, column=1, padx=8, pady=(8, 2), sticky="e")
        self.mapping_progress = ttk.Progressbar(frame, orient="horizontal", mode="determinate", maximum=100)
        self.mapping_progress.grid(row=1, column=0, columnspan=2, padx=8, pady=(0, 8), sticky="ew")
        ttk.Label(frame, text="Output elements").grid(row=2, column=0, padx=8, pady=(0, 2), sticky="w")
        ttk.Label(frame, textvariable=self.output_elements_var).grid(row=2, column=1, padx=8, pady=(0, 2), sticky="e")
        ttk.Label(frame, text="Mapped elements").grid(row=3, column=0, padx=8, pady=(0, 2), sticky="w")
        ttk.Label(frame, textvariable=self.mapped_elements_var).grid(row=3, column=1, padx=8, pady=(0, 2), sticky="e")
        ttk.Label(frame, text="Out-of-grid elements").grid(row=4, column=0, padx=8, pady=(0, 2), sticky="w")
        ttk.Label(frame, textvariable=self.out_of_grid_var).grid(row=4, column=1, padx=8, pady=(0, 2), sticky="e")
        ttk.Label(frame, text="Total power out (kW)").grid(row=5, column=0, padx=8, pady=(0, 2), sticky="w")
        ttk.Label(frame, textvariable=self.total_power_out_var).grid(row=5, column=1, padx=8, pady=(0, 2), sticky="e")
        ttk.Label(frame, text="Total power (SPECTRA) (kW)").grid(row=6, column=0, padx=8, pady=(0, 8), sticky="w")
        ttk.Label(frame, textvariable=self.total_power_spectra_var).grid(row=6, column=1, padx=8, pady=(0, 8), sticky="e")
        frame.columnconfigure(0, weight=1)

    def _build_action_card(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X)
        ttk.Button(frame, text="Load backup file...", command=self._on_load_backup_file).pack(fill=tk.X, pady=(0, 6))
        ttk.Button(frame, text="Rerun from backup", command=self._on_rerun_from_backup).pack(fill=tk.X, pady=(0, 6))
        self.create_btn = ttk.Button(frame, text="Create heat flux file", command=self._on_create)
        self.create_btn.pack(fill=tk.X, pady=(0, 6))
        self.view_btn = ttk.Button(frame, text="View file location", command=self._on_view_location)
        self.view_btn.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(frame, text="Run sample validation", command=self._on_run_sample_validation).pack(fill=tk.X, pady=(0, 6))
        ttk.Button(frame, text="Exit Session", command=self._on_exit).pack(fill=tk.X)

    def _attach_var_traces(self) -> None:
        self.output_folder_var.trace_add("write", lambda *_: self._update_ready_state())
        self.output_filename_var.trace_add("write", lambda *_: self._update_ready_state())
        self.power_ratio_var.trace_add("write", lambda *_: self._update_ready_state())

    def _on_upload_ansys(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("ANSYS APDL", "*.dat"), ("All files", "*.*")])
        if not path:
            return
        self._load_ansys_for_path(Path(path), interactive_cache_prompt=True)

    def _on_upload_spectra(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("SPECTRA", "*.dta *.data *.dta2"), ("All files", "*.*")])
        if not path:
            return
        self._load_spectra_for_path(Path(path))

    def _on_browse_output_folder(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_folder_var.set(path)
            self._update_ready_state()

    def _load_ansys_for_path(self, source_path: Path, interactive_cache_prompt: bool) -> bool:
        self.ansys_path_var.set(str(source_path))
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
            self.ansys_cache_var.set("Cache source: loaded from cache")
            self._update_ansys_progress(
                100,
                f"Loaded cache: nodes={len(self.ansys_result.node_store.node_ids)}, "
                f"elements={self.ansys_result.total_elements}, "
                f"flux={len(self.ansys_result.heatflux_elements)}",
            )
            if not self.output_folder_var.get().strip():
                self.output_folder_var.set(str(source_path.parent))
            self.upload_ansys_btn.configure(state=tk.NORMAL)
            if self._can_start_spectra():
                self.upload_spectra_btn.configure(state=tk.NORMAL)
            self._save_session_backup()
            self._update_ready_state()
            return True

        self._update_ansys_progress(0, "Reading ANSYS file...")
        try:
            self.ansys_result = read_ansys_file(source_path, progress_cb=self._on_ansys_parse_progress)
        except Exception as exc:
            self._set_state_idle()
            self.ansys_loaded = False
            self.ansys_result = None
            self.ansys_cache_var.set("Cache source: parse failed")
            self._update_ansys_progress(0, "ANSYS parse failed")
            messagebox.showerror("ANSYS Parse Error", str(exc))
            return False

        self.ansys_loaded = True
        self.ansys_cache_var.set("Cache source: parsed from file")
        self._update_ansys_progress(
            100,
            f"Parsed nodes={len(self.ansys_result.node_store.node_ids)}, "
            f"elements={self.ansys_result.total_elements}, "
            f"flux={len(self.ansys_result.heatflux_elements)}",
        )
        try:
            save_ansys_parse_cache(source_path, self.ansys_result)
        except Exception:
            pass
        if not self.output_folder_var.get().strip():
            self.output_folder_var.set(str(source_path.parent))
        self._is_loading_ansys = False
        self.upload_ansys_btn.configure(state=tk.NORMAL)
        if self._can_start_spectra():
            self.upload_spectra_btn.configure(state=tk.NORMAL)
        self._save_session_backup()
        self._update_ready_state()
        return True

    def _load_spectra_for_path(self, spectra_path: Path) -> bool:
        self.spectra_path_var.set(str(spectra_path))
        self._set_state_spectra_loading()
        self._update_spectra_progress(0, "Reading SPECTRA file...")
        try:
            self.spectra_result = read_spectra_file(spectra_path, progress_cb=self._on_spectra_parse_progress)
        except Exception as exc:
            self._set_state_idle()
            self.spectra_loaded = False
            self.spectra_result = None
            self._update_spectra_progress(0, "SPECTRA parse failed")
            messagebox.showerror("SPECTRA Parse Error", str(exc))
            return False

        self.spectra_loaded = True
        self._update_spectra_progress(
            100,
            f"Parsed grid {self.spectra_result.n_col}x{self.spectra_result.n_row}, "
            f"peak={self.spectra_result.peak_power_density_kw_mrad2:.4g} kW/mrad^2",
        )
        self.total_power_spectra_var.set(f"{self.spectra_result.total_power_kw:.6g}")
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
        self.create_btn.configure(state=tk.DISABLED)
        self.view_btn.configure(state=tk.DISABLED)
        self.upload_ansys_btn.configure(state=tk.NORMAL)
        self.upload_spectra_btn.configure(state=tk.NORMAL if self.ansys_loaded else tk.DISABLED)

    def _set_state_ansys_loading(self) -> None:
        self._is_loading_ansys = True
        self.upload_ansys_btn.configure(state=tk.DISABLED)
        self.upload_spectra_btn.configure(state=tk.DISABLED)
        self.create_btn.configure(state=tk.DISABLED)

    def _set_state_spectra_loading(self) -> None:
        self._is_loading_spectra = True
        self.upload_spectra_btn.configure(state=tk.DISABLED)
        self.upload_ansys_btn.configure(state=tk.DISABLED)
        self.create_btn.configure(state=tk.DISABLED)

    def _set_state_mapping(self) -> None:
        self._is_mapping = True
        self.upload_spectra_btn.configure(state=tk.DISABLED)
        self.upload_ansys_btn.configure(state=tk.DISABLED)
        self.create_btn.configure(state=tk.DISABLED)
        self.view_btn.configure(state=tk.DISABLED)
        self.mapping_progress["value"] = 0
        self.mapping_percent_var.set("0%")

    def _set_state_done(self) -> None:
        self._is_mapping = False
        self.upload_spectra_btn.configure(state=tk.NORMAL)
        self.upload_ansys_btn.configure(state=tk.NORMAL)
        self.view_btn.configure(state=tk.NORMAL if self.last_output_path is not None else tk.DISABLED)
        self._update_ready_state()

    def _can_start_spectra(self) -> bool:
        return self.ansys_loaded and not self._is_loading_ansys

    def _on_update_geometry(self) -> None:
        self._update_ready_state()
        if self.geometry_valid:
            messagebox.showinfo("Geometry", "Geometry basis is valid.")
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
            self.total_power_out_var.set(f"{total_power_out_kw:.6g}")
            self._save_session_backup()
            messagebox.showinfo(
                "Completed",
                "Heat flux file created.\n"
                f"Output: {output_path}\n"
                f"Elements: {len(mapped)}\n"
                f"Total power out: {total_power_out_kw:.6g} kW",
            )
        except Exception as exc:
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
        try:
            total_power_out_kw = round(float(self.total_power_out_var.get()), 3)
        except ValueError:
            total_power_out_kw = 0.0

        return {
            "ansys_file": self.ansys_path_var.get().strip(),
            "spectra_file": self.spectra_path_var.get().strip(),
            "geometry": {
                "source": source.tolist(),
                "target": target.tolist(),
                "horizontal": horizontal.tolist(),
            },
            "output": {
                "directory": self.output_folder_var.get().strip(),
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
        self.ansys_path_var.set(str(backup.get("ansys_file", "") or ""))
        self.spectra_path_var.set(str(backup.get("spectra_file", "") or ""))
        output = backup.get("output", {})
        if isinstance(output, dict):
            self.output_folder_var.set(str(output.get("directory", "") or ""))
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
            self.output_elements_var.set(str(stats.get("output_elements", "-")))
            self.mapped_elements_var.set(str(stats.get("mapped_elements", "-")))
            self.out_of_grid_var.set(str(stats.get("out_of_grid_elements", "-")))
            self.total_power_out_var.set(str(stats.get("total_power_out_kw", "-")))
            spectra_total = stats.get("spectra_total_power_kw", "-")
            self.total_power_spectra_var.set(str(spectra_total))
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

        ansys_path = Path(self.ansys_path_var.get().strip())
        spectra_path = Path(self.spectra_path_var.get().strip())
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
        p = self.ansys_path_var.get().strip()
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
