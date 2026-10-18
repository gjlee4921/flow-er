"""The low-confidence threshold: chosen to catch the wrong gates, stored in the version, used by autogate."""
import pytest

from flow_er.config import settings
from flow_er.ml import confidence, registry
from flow_er.ml.runtime import LOW_CONFIDENCE


@pytest.fixture(autouse=True)
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")


def test_threshold_separates_the_wrong_gates():
    rows = [{"confidence": c, "error_pp": e} for c, e in
            [(0.95, 0.1), (0.9, 0.2), (0.85, 0.3), (0.8, 0.1), (0.7, 2.5), (0.65, 3.0), (0.5, 4.0), (None, 9.0)]]
    s = confidence.choose_threshold(rows, large_pp=1.0)
    assert s["gates"] == 7 and s["wrong"] == 3                 # template-copy gates (no confidence) left out
    assert s["wrong_flagged"] == 3 and s["right_flagged"] == 0  # every wrong gate caught, no right one
    assert 0.7 < s["threshold"] <= 0.8
    assert s["default_wrong_flagged"] == 1                     # 0.6 would catch only one of the three


def test_stored_threshold_is_what_autogate_uses():
    assert confidence.threshold("lab1") == LOW_CONFIDENCE       # no version yet: the default
    registry.register_version("lab1", make_current=True)
    v = confidence.store("lab1", {"threshold": 0.82, "wrong": 3, "wrong_flagged": 3}, donors=["human:4191"])
    info = registry.version_info("lab1", v)
    assert info["confidence_threshold"] == 0.82 and info["metrics"]["confidence"]["calibrated_on"] == ["human:4191"]
    assert confidence.threshold("lab1") == 0.82
