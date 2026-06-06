from __future__ import annotations

import socket
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from heatflux.gui import plot_launcher
from heatflux.gui.plot_launcher import (
    _POWER_DENSITY_PORT,
    _SPECTRA_PLOT_PORT,
    _port_in_use,
    _terminate_child_streamlit_processes,
    launch_3d_plot_window,
    launch_3d_spectra_plot_window,
    terminate_streamlit_children,
)


@pytest.fixture(autouse=True)
def _isolate_child_procs():
    """Each test starts with an empty tracker and leaves it empty."""
    plot_launcher._child_procs.clear()
    yield
    plot_launcher._child_procs.clear()


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ---------------------------------------------------------------------------
# Port constants
# ---------------------------------------------------------------------------

def test_distinct_ports_for_power_density_and_spectra() -> None:
    assert _POWER_DENSITY_PORT != _SPECTRA_PLOT_PORT


# ---------------------------------------------------------------------------
# _port_in_use
# ---------------------------------------------------------------------------

def test_port_in_use_false_when_port_is_free() -> None:
    assert _port_in_use(_free_port()) is False


def test_port_in_use_true_when_port_is_bound() -> None:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]
    try:
        assert _port_in_use(port) is True
    finally:
        s.close()


# ---------------------------------------------------------------------------
# _terminate_child_streamlit_processes
# ---------------------------------------------------------------------------

def test_terminator_is_safe_with_no_children() -> None:
    _terminate_child_streamlit_processes()  # must not raise


def test_public_terminate_streamlit_children_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public entrypoint must invoke the same termination path that the Tk
    close handler relies on."""
    called = {"n": 0}

    def _spy() -> None:
        called["n"] += 1

    monkeypatch.setattr(plot_launcher, "_terminate_child_streamlit_processes", _spy)
    # Re-import via the module so we exercise the public wrapper's lookup.
    plot_launcher.terminate_streamlit_children()
    assert called["n"] == 1


def test_public_terminate_streamlit_children_is_idempotent() -> None:
    fake = MagicMock()
    fake.poll.return_value = None
    fake.pid = 7777
    plot_launcher._child_procs.append(fake)
    with patch.object(plot_launcher.subprocess, "run") as srun:
        terminate_streamlit_children()
        terminate_streamlit_children()
    # Two calls, but the second sees an unchanged tracker, so taskkill runs each time.
    assert srun.call_count == 2


def test_terminator_skips_already_exited_processes() -> None:
    fake = MagicMock()
    fake.poll.return_value = 0  # already exited
    plot_launcher._child_procs.append(fake)
    with patch.object(plot_launcher.subprocess, "run") as srun:
        _terminate_child_streamlit_processes()
    srun.assert_not_called()
    fake.terminate.assert_not_called()
    fake.kill.assert_not_called()


def test_terminator_uses_taskkill_tree_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock()
    fake.poll.return_value = None  # still running
    fake.pid = 99999
    plot_launcher._child_procs.append(fake)
    monkeypatch.setattr(plot_launcher.os, "name", "nt")
    with patch.object(plot_launcher.subprocess, "run") as srun:
        _terminate_child_streamlit_processes()
    srun.assert_called_once()
    args = srun.call_args[0][0]
    assert args[0] == "taskkill"
    assert "/F" in args
    assert "/T" in args
    assert "/PID" in args
    assert "99999" in args


def test_terminator_uses_proc_terminate_on_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock()
    fake.poll.return_value = None
    fake.pid = 12345
    fake.wait.return_value = 0
    plot_launcher._child_procs.append(fake)
    monkeypatch.setattr(plot_launcher.os, "name", "posix")
    _terminate_child_streamlit_processes()
    fake.terminate.assert_called_once()


def test_terminator_force_kills_when_terminate_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock()
    fake.poll.return_value = None
    fake.pid = 12345
    fake.wait.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=3)
    plot_launcher._child_procs.append(fake)
    monkeypatch.setattr(plot_launcher.os, "name", "posix")
    _terminate_child_streamlit_processes()
    fake.terminate.assert_called_once()
    fake.kill.assert_called_once()


# ---------------------------------------------------------------------------
# launch_3d_spectra_plot_window / launch_3d_plot_window
# ---------------------------------------------------------------------------

def _patch_launch_externals():
    """Patch tk.Toplevel + ttk widgets so the popup creation doesn't need a display."""
    return (
        patch.object(plot_launcher.webbrowser, "open_new_tab"),
        patch.object(plot_launcher.tk, "Toplevel"),
        patch.object(plot_launcher.ttk, "Label"),
        patch.object(plot_launcher.ttk, "Button"),
    )


def _enter_all(ctxs):
    return [c.__enter__() for c in ctxs]


def _exit_all(ctxs):
    for c in ctxs:
        c.__exit__(None, None, None)


def test_launch_spectra_skips_when_script_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # no app_3d_spectra_plot.py here
    ctxs = _patch_launch_externals() + (patch.object(plot_launcher.subprocess, "Popen"),)
    wopen, topl, lbl, btn, popen = _enter_all(list(ctxs))
    try:
        launch_3d_spectra_plot_window(root=MagicMock(), data_file_path="some.data")
    finally:
        _exit_all(list(ctxs))
    popen.assert_not_called()
    wopen.assert_not_called()
    assert plot_launcher._child_procs == []


def test_launch_power_density_skips_when_script_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    ctxs = _patch_launch_externals() + (patch.object(plot_launcher.subprocess, "Popen"),)
    wopen, topl, lbl, btn, popen = _enter_all(list(ctxs))
    try:
        launch_3d_plot_window(root=MagicMock(), data_file_path="dummy.inp")
    finally:
        _exit_all(list(ctxs))
    popen.assert_not_called()
    wopen.assert_not_called()


def test_launch_spectra_passes_env_var_and_port(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "app_3d_spectra_plot.py").write_text("# stub")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plot_launcher, "_port_in_use", lambda port: False)

    captured = {}

    class FakePopen:
        def __init__(self, cmd, env=None):
            captured["cmd"] = cmd
            captured["env"] = env
            self.pid = 1
        def poll(self):
            return None

    ctxs = _patch_launch_externals() + (
        patch.object(plot_launcher.subprocess, "Popen", side_effect=FakePopen),
    )
    wopen, topl, lbl, btn, popen = _enter_all(list(ctxs))
    try:
        launch_3d_spectra_plot_window(root=MagicMock(), data_file_path=r"K:\some\spectra.data")
    finally:
        _exit_all(list(ctxs))

    assert captured["env"]["SPECTRA_DATA_FILE"] == r"K:\some\spectra.data"
    cmd = captured["cmd"]
    assert "--server.port" in cmd
    assert cmd[cmd.index("--server.port") + 1] == str(_SPECTRA_PLOT_PORT)
    assert "--server.headless" in cmd
    assert cmd[cmd.index("--server.headless") + 1] == "true"
    # Tracked for atexit cleanup.
    assert len(plot_launcher._child_procs) == 1


def test_launch_power_density_passes_env_var_and_port(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "app_3d_power_density.py").write_text("# stub")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plot_launcher, "_port_in_use", lambda port: False)

    captured = {}

    class FakePopen:
        def __init__(self, cmd, env=None):
            captured["cmd"] = cmd
            captured["env"] = env
            self.pid = 1
        def poll(self):
            return None

    ctxs = _patch_launch_externals() + (
        patch.object(plot_launcher.subprocess, "Popen", side_effect=FakePopen),
    )
    wopen, topl, lbl, btn, popen = _enter_all(list(ctxs))
    try:
        launch_3d_plot_window(root=MagicMock(), data_file_path="out.inp")
    finally:
        _exit_all(list(ctxs))

    assert captured["env"]["DATA_FILE_PATH"] == "out.inp"
    cmd = captured["cmd"]
    assert cmd[cmd.index("--server.port") + 1] == str(_POWER_DENSITY_PORT)
    assert len(plot_launcher._child_procs) == 1


def test_launch_skips_popen_when_port_busy(monkeypatch, tmp_path: Path) -> None:
    """Re-clicking the same button must NOT spawn a duplicate Streamlit on the same port."""
    (tmp_path / "app_3d_spectra_plot.py").write_text("# stub")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plot_launcher, "_port_in_use", lambda port: True)

    ctxs = _patch_launch_externals() + (patch.object(plot_launcher.subprocess, "Popen"),)
    wopen, topl, lbl, btn, popen = _enter_all(list(ctxs))
    try:
        launch_3d_spectra_plot_window(root=MagicMock(), data_file_path="some.data")
    finally:
        _exit_all(list(ctxs))

    popen.assert_not_called()
    wopen.assert_called_once()
    url = wopen.call_args[0][0]
    assert str(_SPECTRA_PLOT_PORT) in url
    assert plot_launcher._child_procs == []


# ---------------------------------------------------------------------------
# Deprecated Streamlit kwarg has been removed from the viewer scripts
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "script_name",
    ["app_3d_spectra_plot.py", "app_3d_power_density.py"],
)
def test_streamlit_scripts_no_longer_use_deprecated_kwarg(script_name: str) -> None:
    script = _REPO_ROOT / script_name
    if not script.exists():
        pytest.skip(f"{script_name} not present in repo root")
    text = script.read_text(encoding="utf-8")
    assert "use_container_width" not in text, (
        f"{script_name} still contains deprecated `use_container_width` kwarg"
    )
    assert ("width='stretch'" in text) or ('width="stretch"' in text), (
        f"{script_name} does not use the replacement `width='stretch'` kwarg"
    )


def test_spectra_plot_script_reads_env_var_for_default_path() -> None:
    """The integration relies on SPECTRA_DATA_FILE being read by the Streamlit script."""
    script = _REPO_ROOT / "app_3d_spectra_plot.py"
    if not script.exists():
        pytest.skip("app_3d_spectra_plot.py not present in repo root")
    text = script.read_text(encoding="utf-8")
    assert "SPECTRA_DATA_FILE" in text
    assert "os.environ.get" in text or "os.getenv" in text
