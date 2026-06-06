import atexit
import tkinter as tk
from tkinter import ttk
import sys
import os
import socket
import subprocess
import webbrowser
from pathlib import Path
import logging

_log = logging.getLogger(__name__)


_POWER_DENSITY_PORT = 8501
_SPECTRA_PLOT_PORT = 8502

_child_procs: list[subprocess.Popen] = []


def _terminate_child_streamlit_processes() -> None:
    """Kill any Streamlit child processes spawned during this session.

    `streamlit run` may run the actual server as a grandchild process, so
    on Windows we use `taskkill /F /T` to terminate the whole tree rooted
    at the launcher PID. Registered via atexit AND called explicitly from
    the Tk window-close handler so the kill is synchronous on app exit.
    """
    if not _child_procs:
        return
    _log.info("Terminating %d tracked Streamlit child process(es)", len(_child_procs))
    for proc in _child_procs:
        if proc.poll() is not None:
            continue
        try:
            if os.name == "nt":
                result = subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    timeout=5,
                )
                _log.info(
                    "taskkill PID %s -> rc=%s", proc.pid, result.returncode
                )
            else:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                _log.info("terminated PID %s", proc.pid)
        except Exception as e:
            _log.warning("Failed to terminate Streamlit child PID %s: %s", proc.pid, e)


def terminate_streamlit_children() -> None:
    """Public entrypoint: kill all spawned Streamlit servers.

    Safe to call multiple times; already-exited processes are skipped.
    Call this from your Tk window-close handler so cleanup happens
    deterministically before the interpreter exits.
    """
    _terminate_child_streamlit_processes()


atexit.register(_terminate_child_streamlit_processes)


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if something is already listening on (host, port)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.25)
    try:
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


def _launch_streamlit_app(
    root: tk.Tk,
    script_path: Path,
    port: int,
    env_overrides: dict,
    popup_title: str,
    popup_text: str,
) -> None:
    """Launch a Streamlit app on a fixed port and open its URL in the browser.

    If the port is already bound (e.g. the user re-clicked the same button),
    we skip spawning a new process and just point the browser at the existing
    server so each tool keeps its own dedicated tab.
    """
    if not script_path.exists():
        _log.error("Streamlit script not found: %s", script_path)
        return

    url = f"http://localhost:{port}"

    try:
        if _port_in_use(port):
            _log.info("Streamlit already running on port %d; reusing existing server.", port)
        else:
            env = os.environ.copy()
            env.update({k: str(v) for k, v in env_overrides.items()})

            cmd = [
                sys.executable,
                "-m", "streamlit", "run", str(script_path),
                "--server.port", str(port),
                "--server.headless", "true",
            ]
            proc = subprocess.Popen(cmd, env=env)
            _child_procs.append(proc)

        webbrowser.open_new_tab(url)

        popup = tk.Toplevel(root)
        popup.title(popup_title)
        ttk.Label(popup, text=popup_text + f"\n\nURL: {url}").pack(padx=20, pady=20)
        ttk.Button(popup, text="Close", command=popup.destroy).pack(pady=10)
    except Exception as e:
        _log.error("Failed to launch Streamlit app %s: %s", script_path.name, e)


def launch_3d_plot_window(root: tk.Tk, data_file_path: str = ""):
    """
    Launches the 3D mapped-power viewer (`app_3d_power_density.py`) in the
    browser on a dedicated port so it does not collide with the SPECTRA
    surface viewer.
    """
    _log.info("Attempting to launch 3D mapped-power plot for %s", data_file_path)

    _launch_streamlit_app(
        root,
        script_path=Path("app_3d_power_density.py").resolve(),
        port=_POWER_DENSITY_PORT,
        env_overrides={"DATA_FILE_PATH": str(data_file_path)},
        popup_title="3D Plot Running",
        popup_text=f"Launching plot for:\n{data_file_path}\n\n3D Plot has been launched in your browser.",
    )


def launch_3d_spectra_plot_window(root: tk.Tk, data_file_path: str = ""):
    """
    Launches the 3D SPECTRA surface/contour viewer (`app_3d_spectra_plot.py`)
    in the browser on its own dedicated port. The SPECTRA data-file path is
    passed via the SPECTRA_DATA_FILE environment variable so the Streamlit
    sidebar default points at the file the user just loaded in the Tk UI.
    """
    _log.info("Attempting to launch 3D SPECTRA surface plot for %s", data_file_path)

    _launch_streamlit_app(
        root,
        script_path=Path("app_3d_spectra_plot.py").resolve(),
        port=_SPECTRA_PLOT_PORT,
        env_overrides={"SPECTRA_DATA_FILE": str(data_file_path)},
        popup_title="3D Surface Plot Running",
        popup_text=(
            f"Launching 3D surface plot and contour for:\n{data_file_path}\n\n"
            "The plot has been launched in your browser."
        ),
    )
