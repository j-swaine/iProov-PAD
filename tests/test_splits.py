"""Split hygiene: every check in audit.leakage must pass on a freshly made split."""

from __future__ import annotations

import pytest

from screenpad.audit.leakage import run_checks
from screenpad.data.splits import make_splits

CFG = {"dev_attacks": 2, "test_extra_bonafide": 10, "val_frac": 0.2}


@pytest.fixture
def splits(toy_manifest):
    return make_splits(toy_manifest, CFG, seed=0)


@pytest.mark.parametrize("check", ["ids_crossing_splits", "real_attacks_in_training", "twins_in_training", "dev_without_both_classes"])
def test_leakage_checks_pass(splits, check):
    assert run_checks(splits)[check].empty


def test_split_sizes(splits):
    counts = splits.groupby(["split", "label"]).size()
    assert counts[("dev", 1)] == 2 and counts[("dev", 0)] == 2  # dev = twin pairs only
    assert counts[("test", 1)] == 4 and counts[("test", 0)] == 4 + 10  # remaining twins + extra
    assert (splits["split"] == "val").sum() == round(0.2 * (60 - 6 - 10))


def test_dev_size_does_not_move_bonafide(toy_manifest):
    a = make_splits(toy_manifest, {**CFG, "dev_attacks": 1}, seed=3)
    b = make_splits(toy_manifest, {**CFG, "dev_attacks": 3}, seed=3)
    unpaired = ~a["paired"]
    assert a.loc[unpaired, "split"].equals(b.loc[unpaired, "split"])


def test_deterministic(toy_manifest):
    a = make_splits(toy_manifest, CFG, seed=1)["split"]
    b = make_splits(toy_manifest, CFG, seed=1)["split"]
    assert a.equals(b)


def test_leakage_check_catches_crossing(splits):
    bad = splits.copy()
    bad.loc[(bad["id"] == "00000") & (bad["label"] == 0), "split"] = "train"
    assert not run_checks(bad)["ids_crossing_splits"].empty
    assert not run_checks(bad)["twins_in_training"].empty
