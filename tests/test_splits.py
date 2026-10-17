"""Split hygiene: every check in audit.leakage must pass on a freshly made split."""

from __future__ import annotations

import pytest

from screenpad.audit.leakage import run_checks
from screenpad.data.splits import make_splits

CFG = {"test_extra_bonafide": 10, "val_frac": 0.2}


@pytest.fixture
def splits(toy_manifest):
    return make_splits(toy_manifest, CFG, seed=0)


@pytest.mark.parametrize("check", ["ids_crossing_splits", "real_attacks_outside_test", "twins_outside_test"])
def test_leakage_checks_pass(splits, check):
    assert run_checks(splits)[check].empty


def test_split_sizes(splits):
    test_bf = splits[(splits["split"] == "test") & (splits["label"] == 0)]
    assert len(test_bf) == 6 + 10  # twins + extra
    assert (splits["split"] == "val").sum() == round(0.2 * (60 - 16))


def test_deterministic(toy_manifest):
    a = make_splits(toy_manifest, CFG, seed=1)["split"]
    b = make_splits(toy_manifest, CFG, seed=1)["split"]
    assert a.equals(b)


def test_leakage_check_catches_crossing(splits):
    bad = splits.copy()
    bad.loc[(bad["id"] == "00000") & (bad["label"] == 0), "split"] = "train"
    assert not run_checks(bad)["ids_crossing_splits"].empty
    assert not run_checks(bad)["twins_outside_test"].empty
