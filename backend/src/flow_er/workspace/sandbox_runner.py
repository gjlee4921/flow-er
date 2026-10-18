"""Runs inside the sandbox: loads the script, hands it its inputs, writes what it returns. Standalone on
purpose (it cannot import flow_er), and copied into each run's throwaway folder."""
import importlib.util
import json
import resource
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

work = Path(__file__).parent
out = work / "out"


def load(spec: dict):
    p = work / spec["path"]
    if spec["kind"] == "table":
        return pd.read_csv(p)
    files = {}
    for f in sorted(p.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        df = pd.DataFrame(z["data"], columns=[str(c) for c in z["columns"]])
        kw = f.with_suffix(".json")
        df.attrs["keywords"] = json.loads(kw.read_text()) if kw.exists() else {}
        files[f.stem] = df
    return files


def save(name: str, value) -> str:
    if isinstance(value, pd.DataFrame):
        value.to_csv(out / f"{name}.csv", index=False, float_format="%.10g")
        return f"{name}.csv"
    if isinstance(value, dict) and all(isinstance(v, pd.DataFrame) for v in value.values()):
        (out / name).mkdir()
        for sid, df in sorted(value.items()):
            np.savez(out / name / f"{sid}.npz", data=df.to_numpy(dtype="float64"), columns=np.array(df.columns, dtype=str))
        return name
    raise TypeError(f"output '{name}' must be a table (DataFrame) or one table per file (dict of DataFrames)")


def main() -> int:
    job = json.loads((work / "job.json").read_text())
    lim = job.get("limits") or {}
    cpu, fsize = int(lim.get("cpu_seconds", 60)), int(lim.get("file_mb", 512)) << 20
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    try:
        spec = importlib.util.spec_from_file_location("script", work / "script.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        inputs = {n: load(s) for n, s in job["inputs"].items()}
        result = mod.run(inputs, job.get("params") or {})
        if not isinstance(result, dict):
            raise TypeError("run(inputs, params) must return a dict of outputs")
        outputs = {n: save(n, v) for n, v in result.items()}
        (out / "_result.json").write_text(json.dumps({"ok": True, "outputs": outputs}))
        return 0
    except Exception as e:  # noqa: BLE001 - anything the script raises is reported back, not lost
        tb = traceback.extract_tb(e.__traceback__)
        line = next((f.lineno for f in reversed(tb) if f.filename.endswith("script.py")), None)
        msg = f"{e.__class__.__name__}: {e}" + (f" (script line {line})" if line else "")
        (out / "_result.json").write_text(json.dumps({"ok": False, "error": msg}))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
