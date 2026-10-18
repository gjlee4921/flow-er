"""The sandbox a script step runs in (spec §3, v1 for a laptop).

A script runs as a subprocess of this environment's Python (`-I`) in a throwaway folder: its inputs are
copied in read-only, its outputs go to one folder, its environment carries no keys, and it has a time
limit and limits on CPU, file size and memory. On macOS a sandbox-exec profile denies the network,
writing anywhere but its folder, reading anything in the home folder but the Python it runs on, and
starting any other program. On Linux bubblewrap does the same. Without either, scripts do not run.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

RUNNER = Path(__file__).with_name("sandbox_runner.py")
LIMITS = {"seconds": 60, "cpu_seconds": 60, "file_mb": 512, "memory_mb": 4096}


class SandboxUnavailable(RuntimeError):
    pass


def _profile(work: Path, python_dirs: list[Path], python_bins: list[Path]) -> str:
    q = lambda p: json.dumps(str(p))  # SBPL strings use the same quoting as JSON
    home = Path.home().resolve()
    reads = " ".join(f"(subpath {q(d)})" for d in python_dirs)
    return f"""(version 1)
(allow default)
(deny network*)
(deny process-exec)
(allow process-exec {" ".join(f"(literal {q(b)})" for b in python_bins)})
(deny file-write*)
(allow file-write* (subpath {q(work)}) (literal "/dev/null"))
(deny file-read-data (subpath {q(home)}))
(allow file-read-data {reads} (subpath {q(work)}))
"""


def _python() -> tuple[Path, list[Path], list[Path]]:
    """This environment's interpreter (by its own path, so it finds its packages), the paths it may be
    started by, and the folders it reads its code from."""
    exe = Path(sys.executable)
    real = exe.resolve()
    dirs = {Path(sys.prefix), Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(), real.parent.parent}
    return exe, sorted({exe, real}), sorted(dirs)


def _rss_mb(pid: int) -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-g", str(pid)], check=False, capture_output=True, text=True).stdout
    return sum(int(x) for x in out.split() if x.isdigit()) / 1024


def run(code: str, inputs: dict[str, Path], out_dir: Path, params: dict | None = None, limits: dict | None = None) -> dict:
    """Run one script's `run(inputs, params)` in the sandbox. `inputs` maps each input port to a CSV
    table or a folder of per-file .npz events; what it returns is copied to `out_dir`. Returns the
    status, the outputs (name -> path), a hash of them, the seconds it took and the end of what it printed."""
    lim = {**LIMITS, **(limits or {})}
    work = Path(tempfile.mkdtemp(prefix="flower-sbx-")).resolve()
    try:
        (work / "in").mkdir()
        (work / "out").mkdir()
        (work / "tmp").mkdir()
        manifest = {}
        for name, src in inputs.items():
            dst = work / "in" / name
            if src.is_dir():
                shutil.copytree(src, dst)
                manifest[name] = {"kind": "events", "path": f"in/{name}"}
            else:
                shutil.copy2(src, dst.with_suffix(src.suffix))
                manifest[name] = {"kind": "table", "path": f"in/{name}{src.suffix}"}
        for p in (work / "in").rglob("*"):
            p.chmod(0o444 if p.is_file() else 0o555)
        (work / "script.py").write_text(code)
        # the runner sets its own CPU, file-size and open-file limits before it loads the script
        (work / "job.json").write_text(json.dumps({"inputs": manifest, "params": params or {},
                                                   "limits": {"cpu_seconds": lim["cpu_seconds"], "file_mb": lim["file_mb"]}}))
        shutil.copy2(RUNNER, work / "runner.py")
        py, bins, dirs = _python()
        cmd = [str(py), "-I", "-B", str(work / "runner.py")]
        if platform.system() == "Darwin" and Path("/usr/bin/sandbox-exec").exists():
            cmd = ["/usr/bin/sandbox-exec", "-p", _profile(work, dirs, bins), *cmd]
        elif shutil.which("bwrap"):
            ro = [a for d in dirs for a in ("--ro-bind", str(d), str(d))]
            cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
                   *ro, "--bind", str(work), str(work), "--dev", "/dev", "--proc", "/proc", "--chdir", str(work), *cmd]
        else:
            raise SandboxUnavailable("no sandbox on this machine (sandbox-exec or bubblewrap), so scripts cannot run")
        env = {"PATH": "/usr/bin:/bin", "HOME": str(work), "TMPDIR": str(work / "tmp"), "PYTHONHASHSEED": "0",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MPLBACKEND": "Agg", "LC_ALL": "C"}
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                start_new_session=True)
        killed: list[str] = []

        def watch():
            while proc.poll() is None:
                if time.time() - t0 > lim["seconds"]:
                    killed.append(f"took longer than {lim['seconds']} s")
                elif _rss_mb(proc.pid) > lim["memory_mb"]:
                    killed.append(f"used more than {lim['memory_mb']} MB of memory")
                if killed:
                    os.killpg(proc.pid, 9)
                    return
                time.sleep(0.25)

        w = threading.Thread(target=watch, daemon=True)
        w.start()
        log, _ = proc.communicate()
        w.join(1)
        seconds = round(time.time() - t0, 2)
        result = json.loads((work / "out" / "_result.json").read_text()) if (work / "out" / "_result.json").exists() else None
        if killed or proc.returncode != 0 or not result or not result.get("ok"):
            why = killed[0] if killed else (result or {}).get("error") or f"it stopped with code {proc.returncode}"
            return {"status": "failed", "error": why, "log": log[-4000:], "seconds": seconds}
        out_dir.mkdir(parents=True, exist_ok=True)
        h, outs = hashlib.sha256(), {}
        for n, rel in sorted(result["outputs"].items()):
            src, dst = work / "out" / rel, out_dir / rel
            for f in sorted([src] if src.is_file() else src.rglob("*")):
                if f.is_file():
                    h.update(f.relative_to(work).as_posix().encode())
                    h.update(f.read_bytes())
            shutil.rmtree(dst, ignore_errors=True) if dst.is_dir() else dst.unlink(missing_ok=True)
            shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)
            outs[n] = dst
        return {"status": "done", "outputs": outs, "hash": h.hexdigest()[:16], "log": log[-4000:], "seconds": seconds}
    finally:
        shutil.rmtree(work, ignore_errors=True)
