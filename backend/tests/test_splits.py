"""The frozen split must keep every individual on one side and use no excluded sample.

Needs the registry and split in settings.splits_dir (outside git); skipped when they are absent.
"""

import copy

import pytest

from flow_er.config import settings
from flow_er.ml import splits

pytestmark = pytest.mark.skipif(not (settings.splits_dir / splits.SPLIT).exists(),
                                reason="no frozen split in data/_splits")


@pytest.fixture(scope="module")
def registry():
    return splits.load_registry()


@pytest.fixture(scope="module")
def frozen():
    return splits.load_split(check=False)


@pytest.fixture
def split(frozen):
    return copy.deepcopy(frozen)


def keys_where(registry, **match):
    return [splits.key(s) for s in registry["samples"]
            if all(s.get(k) == v for k, v in match.items())]


def test_frozen_split_is_valid_and_registry_unchanged(frozen, registry):
    splits.validate(frozen, registry, settings.splits_dir)


def test_split_is_reproducible_from_the_rules(frozen, registry):
    rebuilt = splits.build_split(registry, settings.data_dir / "_inspect")
    for part in ("pretrain", "evaluations", "excluded"):
        assert rebuilt[part] == frozen[part]


def test_small_files_match_registry_md5(frozen, registry):
    assert splits.verify_files(frozen, registry, max_mb=20) > 0


def test_test_donor_in_training_fails(split, registry):
    leak = keys_where(registry, individual="human:3740", dataset="als_ics_screen4")[0]
    split["evaluations"]["als"]["stages"]["train"].append(leak)
    with pytest.raises(splits.SplitError, match="training side|split across roles"):
        splits.validate(split, registry)


def test_test_donor_in_pretraining_pool_fails(split, registry):
    leak = keys_where(registry, individual="human:8830", dataset="als_ics_screen4")[0]
    split["pretrain"]["stages"]["train"].append(leak)
    with pytest.raises(splits.SplitError, match="training side"):
        splits.validate(split, registry)


def test_mo280_test_mouse_in_training_fails(split, registry):
    leak = keys_where(registry, individual="mouse:aorta_olfr2:MO280-2", tissue="blood")[0]
    split["evaluations"]["mo280_round1"]["stages"]["train"].append(leak)
    with pytest.raises(splits.SplitError, match="training side"):
        splits.validate(split, registry)


def test_control_anywhere_fails(split, registry):
    control = keys_where(registry, dataset="macrophages_peritoneal", control=True)[0]
    split["pretrain"]["stages"]["train"].append(control)
    with pytest.raises(splits.SplitError, match="control"):
        splits.validate(split, registry)


@pytest.mark.parametrize("match", [
    {"dataset": "als_ics_screen1"},                           # identity unknown
    {"cohort": "MO CVR Aorta (Aurora)", "control": False},    # dropped everywhere
    {"workspace": "#4 Spectral.wsp"},                         # labels unreliable
    {"dataset": "als_spectral", "individual": "human:6936"},  # ALS donor
])
def test_excluded_samples_fail(split, registry, match):
    split["pretrain"]["stages"]["train"].append(keys_where(registry, **match)[0])
    with pytest.raises(splits.SplitError):
        splits.validate(split, registry)


def test_unknown_identity_in_test_fails(split, registry):
    group = set(keys_where(registry, individual="?mouse:macrophages_peritoneal:SM_PM_7:A"))
    pool = split["pretrain"]["stages"]["train"]
    split["pretrain"]["stages"]["train"] = [k for k in pool if k not in group]
    split["evaluations"]["mo280_round1"]["stages"]["test"].append(sorted(group & set(pool))[0])
    with pytest.raises(splits.SplitError, match="unknown identity"):
        splits.validate(split, registry)


def test_donor_wells_split_across_roles_fails(split, registry):
    stages = split["evaluations"]["als"]["stages"]
    well = next(k for k in stages["val"] if "Screen #3 HC" in k)
    stages["val"].remove(well)
    stages["finetune"].append(well)
    with pytest.raises(splits.SplitError, match="training side|split across roles"):
        splits.validate(split, registry)


def test_same_content_on_both_sides_fails(split, registry):
    reg = copy.deepcopy(registry)
    test_key = split["evaluations"]["als"]["stages"]["test"][0]
    train_key = split["evaluations"]["als"]["stages"]["train"][0]
    by_key = {splits.key(s): s for s in reg["samples"]}
    by_key[train_key]["md5"] = by_key[test_key]["md5"]     # a renamed copy of a test file
    with pytest.raises(splits.SplitError, match="identical file content"):
        splits.validate(split, reg)
