"""GPU retraining through a shared folder, with a fake notebook: a local folder and a thread that
runs the same retrain() (ml.remote.run_job), as the Colab loop does."""
import threading
import time

import pytest
import torch

from flow_er.config import settings
from flow_er.ml import finetune, registry, remote

from .test_finetune import TINY, _label, donors


@pytest.fixture(autouse=True)
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    monkeypatch.setattr(settings, "colab_jobs_dir", tmp_path / "drive" / "jobs")


def _notebook(stop: threading.Event):
    """The notebook loop: keep the heartbeat fresh and run every queued job."""
    jobs = remote.jobs_dir()
    jobs.mkdir(parents=True, exist_ok=True)
    while not stop.is_set():
        (jobs / remote.HEARTBEAT).touch()
        for job in sorted(p for p in jobs.iterdir() if p.is_dir()):
            if '"queued"' in (job / "job.json").read_text():
                remote.run_job(job, log=lambda *a: None)
        time.sleep(0.2)


def test_retrain_on_colab_round_trip(tmp_path):
    _label(tmp_path, donors(n_val=1, n_train=4))
    stop = threading.Event()
    nb = threading.Thread(target=_notebook, args=(stop,), daemon=True)
    nb.start()
    try:
        time.sleep(0.5)
        out = finetune.retrain("lab1", on="auto", epochs=2, patience=1, unito_arch=TINY, graduate=3,
                               wait_minutes=5, log=lambda *a: None)
    finally:
        stop.set()
        nb.join(5)
    assert out["job"] and out["version"] == "v001"
    info = registry.version_info("lab1", "v001")
    assert info["shared"] is None and "gpu job" in info["metrics"]["trained_on"]
    assert registry.current_version("lab1") == ("v001" if out["promoted"] else None)
    for rel in info["gates"].values():                         # weights came back, in fp32
        state = torch.load(registry._abs(rel) / "model.pt", weights_only=True)
        assert all(v.dtype != torch.float16 for v in state.values())


def test_auto_uses_the_cpu_when_the_notebook_is_quiet(tmp_path):
    _label(tmp_path, donors(n_val=1, n_train=4))
    jobs = remote.jobs_dir()
    jobs.mkdir(parents=True)
    hb = jobs / remote.HEARTBEAT
    hb.touch()
    old = time.time() - remote.STALE_SECONDS - 5
    import os
    os.utime(hb, (old, old))
    out = finetune.retrain("lab1", on="auto", epochs=2, patience=1, unito_arch=TINY, graduate=3, log=lambda *a: None)
    assert "job" not in out and out["version"] == "v001"
    assert not [p for p in jobs.iterdir() if p.is_dir()]     # nothing was sent


def test_submitted_start_models_travel_in_fp16(tmp_path):
    from .test_runtime import tiny_model
    gate = tiny_model(settings.models_dir / "elsewhere" / "cells", ["density"])
    registry.register_version("lab1", gates={"Cells": gate}, make_current=True)
    _label(tmp_path, donors(n_val=1, n_train=2))
    job = remote.submit("lab1", {"epochs": 1})
    [rel] = registry.version_info("lab1")["gates"].values()
    state = torch.load(job / "models" / rel / "model.pt", weights_only=True)
    assert any(v.dtype == torch.float16 for v in state.values())
    assert '"queued"' in (job / "job.json").read_text()
