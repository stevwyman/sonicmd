from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(os.environ.get("STENOPOD_ROOT", Path(__file__).resolve().parent.parent.parent)).resolve()
RUN_DIR = REPO_ROOT / ".run"
PID_FILE = RUN_DIR / "sidecar.pid"
LOG_FILE = RUN_DIR / "sidecar.log"
CONTROL_URL = os.environ.get("STENOPOD_CONTROL", "http://127.0.0.1:8781")


def wav_duration_s(path: str | Path) -> float:
    import wave

    with wave.open(str(path), "rb") as handle:
        n = handle.getnframes()
        sr = handle.getframerate() or 16000
        return n / float(sr)


def python_bin() -> Path:
    return REPO_ROOT / ".venv-host" / "bin" / "python"


def sidecar_script() -> Path:
    return REPO_ROOT / "host" / "sidecar.py"


_capture_proc: subprocess.Popen[bytes] | None = None


def _reap_child(pid: int) -> bool:
    """Reap a zombie child. A zombie still answers kill(pid, 0), so stop() would wait forever."""
    try:
        waited, _status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return False
    except OSError:
        return False
    return waited == pid


def _is_zombie(pid: int) -> bool:
    try:
        stat = subprocess.check_output(["ps", "-p", str(pid), "-o", "stat="], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False
    return stat.startswith("Z")


def _pid_alive(pid: int) -> bool:
    if _reap_child(pid):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return not _is_zombie(pid)


def running_pid() -> int | None:
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
        except ValueError:
            pid = None
        if pid and _pid_alive(pid):
            return pid
        PID_FILE.unlink(missing_ok=True)
    found = _pgrep()
    return found[0] if found else None


def _pgrep(include_file: bool = False) -> list[int]:
    try:
        out = subprocess.check_output(["ps", "-ax", "-o", "pid=,command="], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    pids = []
    needle = "host/sidecar.py"
    for line in out.splitlines():
        if needle not in line or "pgrep" in line:
            continue
        pid_s, _, cmd = line.strip().partition(" ")
        if not include_file and "--file" in cmd:
            continue
        if pid_s.isdigit() and "sidecar.py" in cmd:
            pids.append(int(pid_s))
    return pids


def status() -> dict[str, Any]:
    py = python_bin()
    pid = running_pid()
    return {
        "running": pid is not None,
        "pid": pid,
        "can_start": py.is_file() and sidecar_script().is_file() and os.uname().sysname == "Darwin",
        "venv": str(py) if py.is_file() else None,
        "log": str(LOG_FILE) if LOG_FILE.exists() else None,
    }


def _ctl(method: str, path: str, body: dict[str, Any] | None = None, timeout: float = 8.0) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        CONTROL_URL.rstrip("/") + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"ok": False, "error": raw or str(exc)}
        if not isinstance(parsed, dict):
            parsed = {"ok": False, "error": raw}
        parsed.setdefault("ok", False)
        parsed.setdefault("error", f"HTTP {exc.code}")
        return parsed
    except urllib.error.URLError as exc:
        return {"ok": False, "error": str(exc.reason or exc)}
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {"ok": False, "error": raw}
    return parsed if isinstance(parsed, dict) else {"ok": False, "error": raw}


def control_status() -> dict[str, Any]:
    return _ctl("GET", "/status", timeout=3.0)


def wait_control(timeout: float = 20.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    last: dict[str, Any] = {"ok": False, "error": "control port did not come up"}
    while time.time() < deadline:
        last = control_status()
        if last.get("ok") or last.get("wav") is not None:
            last["ok"] = True
            return last
        time.sleep(0.2)
    return last


def _transcribe_failure(code: int) -> str:
    text = LOG_FILE.read_text(errors="replace") if LOG_FILE.exists() else ""
    lines = text.splitlines()
    start = 0
    for index, line in enumerate(lines):
        if line.startswith("Traceback") or line.startswith("SyntaxError") or "Error:" in line:
            start = index
    excerpt = "\n".join(lines[max(0, start - 3):][-12:]).strip()
    if not excerpt:
        return f"file transcribe exited {code}"
    return f"file transcribe exited {code}\n{excerpt}"


def process_file(
    wav: str,
    start_s: float,
    end_s: float,
    app_url: str,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict[str, Any]:
    py = python_bin()
    if not py.is_file():
        return {"ok": False, "error": "Host venv missing. Run `make sidecar-venv`."}
    cmd = [
        str(py),
        str(sidecar_script()),
        "--app",
        app_url,
        "--file",
        wav,
        "--start-s",
        str(start_s),
        "--end-s",
        str(end_s),
    ]
    if min_speakers and max_speakers and min_speakers == max_speakers:
        cmd.extend(["--num-speakers", str(min_speakers)])
    else:
        if min_speakers:
            cmd.extend(["--min-speakers", str(min_speakers)])
        if max_speakers:
            cmd.extend(["--max-speakers", str(max_speakers)])
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_FILE.open("ab")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        return {"ok": False, "error": _transcribe_failure(proc.returncode)}
    return {"ok": True, "start_s": start_s, "end_s": end_s, "wav": wav}


def start(
    source: str = "both",
    mic_device: str | None = None,
    app_url: str = "http://127.0.0.1:8780",
    session_id: str | None = None,
    wav: str | None = None,
) -> dict[str, Any]:
    if os.uname().sysname != "Darwin":
        raise RuntimeError("Capture only runs on macOS")
    py = python_bin()
    if not py.is_file():
        raise RuntimeError("Host venv missing. Run `make sidecar-venv`.")
    existing = running_pid()
    if existing:
        return {"ok": True, "already": True, "pid": existing}

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(py),
        str(sidecar_script()),
        "--app",
        app_url,
        "--source",
        source,
    ]
    if mic_device:
        cmd.extend(["--mic-device", mic_device])
    if session_id:
        cmd.extend(["--session-id", session_id])
    if wav and Path(wav).is_file():
        cmd.extend(["--wav", wav])
    log = LOG_FILE.open("ab")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    global _capture_proc
    _capture_proc = proc
    PID_FILE.write_text(str(proc.pid))
    time.sleep(0.4)
    if proc.poll() is not None:
        _capture_proc = None
        try:
            proc.wait(timeout=1)
        except Exception:
            pass
        tail = LOG_FILE.read_text(errors="replace")[-2000:] if LOG_FILE.exists() else ""
        PID_FILE.unlink(missing_ok=True)
        raise RuntimeError(f"Sidecar exited immediately.\n{tail}")
    ctl = wait_control()
    return {"ok": True, "already": False, "pid": proc.pid, "wav": ctl.get("wav"), "control": ctl}


def stop(include_file: bool = False) -> dict[str, Any]:
    global _capture_proc
    pids = set(_pgrep(include_file=include_file))
    tracked = running_pid()
    if tracked:
        pids.add(tracked)
    if _capture_proc is not None:
        pids.add(_capture_proc.pid)
    if not pids:
        PID_FILE.unlink(missing_ok=True)
        _capture_proc = None
        return {"ok": True, "stopped": False}
    for pid in list(pids):
        if not _pid_alive(pid):
            pids.discard(pid)
            continue
        try:
            os.kill(pid, signal.SIGINT)
        except OSError:
            continue
    # Recording only writes a wav. The sidecar closes it after ffmpeg exits.
    deadline = time.time() + 12
    while time.time() < deadline and any(_pid_alive(pid) for pid in pids):
        time.sleep(0.2)
    for pid in pids:
        if not _pid_alive(pid):
            continue
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        _pid_alive(pid)
    if _capture_proc is not None:
        try:
            _capture_proc.wait(timeout=1)
        except Exception:
            pass
        _capture_proc = None
    PID_FILE.unlink(missing_ok=True)
    return {"ok": True, "stopped": True}
