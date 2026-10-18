"""Retraining on a GPU through a shared folder (Colab with Google Drive), agreed design 18 Oct.

The worker writes a job folder under settings.colab_jobs_dir: the lab's accepted examples (cache
files), the start models (fp16 for the transfer) and the parameters. A notebook loop on the GPU
(run_job) runs the same retrain() there and writes the new version folder plus summary.json back.
The worker then copies the version into models/labs/<lab>/ and applies the normal promotion.
The notebook touches HEARTBEAT every few seconds; "auto" uses the CPU when it is older than
STALE_SECONDS.

Job folder:
    <jobs>/<job id>/job.json                       status: queued | running | done | failed
    <jobs>/<job id>/params.json                    retrain() keyword arguments and the base version
    <jobs>/<job id>/data/_cache/v1/lab_<lab>/...   the lab's examples
    <jobs>/<job id>/models/...                     the base version and the gate models it names
    <jobs>/<job id>/summary.json                   retrain()'s summary, written by the notebook
"""
import json
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path

from flow_er.config import settings
from flow_er.ml import cache, registry

HEARTBEAT = "heartbeat"
STALE_SECONDS = 180


def jobs_dir() -> Path:
    if not settings.colab_jobs_dir:
        raise ValueError("COLAB_JOBS_DIR is not set, so there is nowhere to send a GPU job")
    return Path(settings.colab_jobs_dir).expanduser()


def notebook_alive() -> bool:
    """Whether the notebook touched its heartbeat in the last STALE_SECONDS."""
    try:
        hb = jobs_dir() / HEARTBEAT
    except ValueError:
        return False
    return hb.exists() and time.time() - hb.stat().st_mtime < STALE_SECONDS


def _status(job: Path, status: str, **extra) -> None:
    (job / "job.json").write_text(json.dumps({"status": status, "at": datetime.now().isoformat(timespec="seconds"), **extra}))


def _copy_weights(src: Path, dst: Path, half: bool) -> None:
    """A model folder (model.pt + recipe.json), its float weights in fp16 (half) or fp32."""
    import torch

    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name != "model.pt" and f.is_file():
            shutil.copy2(f, dst / f.name)
    state = torch.load(src / "model.pt", map_location="cpu", weights_only=True)
    state = {k: (v.half() if half else v.float()) if v.is_floating_point() else v for k, v in state.items()}
    torch.save(state, dst / "model.pt")


def submit(lab: str, params: dict) -> Path:
    """Write a job folder for `lab` and return it: examples, start models and parameters."""
    from flow_er.ml import finetune

    job = jobs_dir() / f"{lab}-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    job.mkdir(parents=True)
    src = cache.cache_dir() / finetune.lab_dataset(lab)
    shutil.copytree(src, job / "data" / "_cache" / "v1" / finetune.lab_dataset(lab))
    base = registry.current_version(lab)
    if base:
        info = registry.version_info(lab)
        vdir = job / "models" / "labs" / lab / base
        vdir.mkdir(parents=True)
        (vdir / "version.json").write_text(json.dumps(info, indent=1))
        (vdir.parent / "current").write_text(base + "\n")
        for d in info.get("gates", {}).values():
            _copy_weights(registry._abs(d), job / "models" / d, half=True)
    (job / "params.json").write_text(json.dumps({"lab": lab, "base": base, "retrain": params}, indent=1))
    _status(job, "queued")
    return job


def run_job(job: Path, log=print) -> dict:
    """What the notebook runs for one job: retrain() inside the job folder (on the GPU when
    there is one), new gate models saved in fp16, summary.json written back."""
    from flow_er.ml import finetune

    p = json.loads((job / "params.json").read_text())
    keep = settings.data_dir, settings.models_dir
    _status(job, "running")
    try:
        settings.data_dir, settings.models_dir = job / "data", job / "models"
        out = finetune.retrain(p["lab"], log=log, **p["retrain"])
        if out["version"]:
            for gid in out["graduated"]:
                d = job / "models" / "labs" / p["lab"] / out["version"] / "gates" / gid
                if d.exists():
                    _copy_weights(d, d, half=True)
        (job / "summary.json").write_text(json.dumps(out, indent=1, default=str))
        _status(job, "done")
        return out
    except Exception as e:  # the worker reads the failure from the job folder
        (job / "summary.json").write_text(json.dumps({"error": f"{e.__class__.__name__}: {e}"}))
        _status(job, "failed")
        raise
    finally:
        settings.data_dir, settings.models_dir = keep


def wait(job: Path, timeout: float | None = None, poll: float = 5.0, log=print) -> dict:
    """Wait for the notebook to finish the job; stop if its heartbeat goes stale."""
    t0 = time.time()
    while True:
        status = json.loads((job / "job.json").read_text())["status"]
        if status in ("done", "failed"):
            out = json.loads((job / "summary.json").read_text())
            if status == "failed":
                raise RuntimeError(f"the GPU job failed: {out.get('error')}")
            return out
        if not notebook_alive():
            raise RuntimeError(f"the notebook's heartbeat is older than {STALE_SECONDS} s; job left in {job}")
        if timeout and time.time() - t0 > timeout:
            raise TimeoutError(f"the GPU job took longer than {timeout:.0f} s; it is still in {job}")
        time.sleep(poll)


def collect(lab: str, job: Path, out: dict, log=print) -> dict:
    """Copy the job's new version into models/labs/<lab>/ (weights back to fp32) and apply the
    normal promotion: current only if it was promoted against the version that is current here."""
    from flow_er.ml import finetune

    if not out.get("version"):
        return out
    p = json.loads((job / "params.json").read_text())
    remote_v = job / "models" / "labs" / lab / out["version"]
    info = json.loads((remote_v / "version.json").read_text())
    version = registry.next_version_name(lab)
    gates = {}
    for gid, rel in info.get("gates", {}).items():
        if rel.startswith(f"labs/{lab}/{out['version']}/"):          # trained in this job
            dst = registry.lab_dir(lab) / version / "gates" / gid
            _copy_weights(job / "models" / rel, dst, half=False)
            gates[gid] = dst
        else:
            gates[gid] = registry._abs(rel)                         # carried over from the base
    metrics = info.get("metrics", {})
    promoted, reason = finetune.promotion_decision(metrics.get("held_out_error_pp", {}).get("current", {}),
                                                   metrics.get("held_out_error_pp", {}).get("candidate", {}))
    if p.get("base") != registry.current_version(lab):
        promoted, reason = False, f"the current version changed while the GPU job ran (was {p.get('base')})"
    registry.register_version(lab, gates=gates, parent=registry.current_version(lab), version=version,
                              metrics={**metrics, "promoted": promoted, "reason": reason, "trained_on": "gpu job " + job.name},
                              training_samples=info.get("training_samples", []),
                              notes=info.get("notes", "") + " (trained on the GPU)", make_current=promoted)
    log(f"{lab}: {version} from the GPU job {'promoted' if promoted else 'kept as candidate'}: {reason}")
    return {**out, "version": version, "promoted": promoted, "reason": reason, "job": str(job)}
