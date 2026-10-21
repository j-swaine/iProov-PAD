"""EpochSampler and eval_sets on a synthetic table; GPU augmentation on small in-memory batches."""

from __future__ import annotations

import copy
import inspect

import numpy as np
import pandas as pd
import pytest
import torch

from screenpad.train.augment import apply, sample_params, to_input
from screenpad.train.data import EpochSampler, eval_sets

DEVICE = torch.device("cpu")

CFG = {
    "p_flip": 0.5,
    "p_affine": 0.5,
    "max_rotate": 5.0,
    "max_shift": 0.05,
    "max_log_scale": 0.1,
    "p_rescale": 0.5,
    "max_rescale": 4.0,
    "p_colour": 0.8,
    "brightness": 0.15,
    "contrast": 0.15,
    "saturation": 0.15,
    "channel_gain": 0.05,
    "p_grey": 0.2,
}


def make_table() -> pd.DataFrame:
    rows = []

    def add(id_, split, label, kind, variant, train_ok=True):
        rows.append((id_, split, label, kind, variant, train_ok))

    def face(id_, split, orig_ok=True):
        add(id_, split, 0, "real_bonafide", 0, orig_ok)
        for v in (0, 1):
            add(id_, split, 0, "camera_only", v)
        for v in range(4):
            add(id_, split, 1, "synthetic_attack", v)

    for i in ("0", "1", "2", "3"):
        face(i, "train")
    face("4", "train", orig_ok=False)
    face("10", "val")
    face("11", "val")
    add("20", "dev", 0, "real_bonafide", 0)
    add("20", "dev", 1, "real_attack", 0)
    return pd.DataFrame(rows, columns=["id", "split", "label", "kind", "variant", "train_ok"])


def test_epoch_has_two_bonafide_and_two_attacks_per_train_face():
    table = make_table()
    idx = EpochSampler(table, seed=42, camera_per_face=1, attacks_per_face=2)(0)
    sub = table.loc[idx]
    assert len(idx) == 5 * 4
    assert (sub.split == "train").all()
    assert sorted(sub.id.unique()) == ["0", "1", "2", "3", "4"]
    for id_, g in sub.groupby("id"):
        assert (g.label == 0).sum() == 2, id_
        assert (g.label == 1).sum() == 2, id_
    face4 = sub[sub.id == "4"]
    assert (face4[face4.label == 0].kind == "camera_only").all()


def test_no_index_repeats_within_an_epoch():
    table = make_table()
    sampler = EpochSampler(table, seed=42)
    for epoch in range(10):
        idx = sampler(epoch)
        assert len(set(idx.tolist())) == len(idx)


def test_same_seed_and_epoch_identical_different_epochs_differ():
    table = make_table()
    a = EpochSampler(table, seed=42)
    b = EpochSampler(table, seed=42)
    assert np.array_equal(a(3), b(3))
    assert any(not np.array_equal(a(0), a(e)) for e in range(1, 6))


def test_filtered_original_of_face4_never_appears():
    table = make_table()
    filtered = table.index[(table.id == "4") & (table.kind == "real_bonafide")]
    assert len(filtered) == 1
    sampler = EpochSampler(table, seed=42)
    for epoch in range(10):
        assert filtered[0] not in set(sampler(epoch).tolist())


def test_every_train_synthetic_attack_is_used_over_epochs():
    table = make_table()
    pool = set(table.index[(table.split == "train") & (table.kind == "synthetic_attack")].tolist())
    assert len(pool) == 5 * 4
    sampler = EpochSampler(table, seed=42)
    seen = set()
    for epoch in range(20):
        seen |= set(sampler(epoch).tolist())
    assert pool <= seen


def test_p_original_reweights_bonafide_and_keeps_balance():
    table = make_table()
    orig = set(table.index[(table.split == "train") & (table.kind == "real_bonafide") & table.train_ok].tolist())
    default = EpochSampler(table, seed=42)
    explicit = EpochSampler(table, seed=42, p_original=1.0)
    assert all(np.array_equal(default(e), explicit(e)) for e in range(3))  # the default draws exactly as before
    for p, lo, hi in ((0.0, 0, 0), (0.5, 1, 39)):
        sampler = EpochSampler(table, seed=42, p_original=p)
        used = 0
        for e in range(10):
            idx = sampler(e)
            assert (table.loc[idx].label == 0).sum() == (table.loc[idx].label == 1).sum()  # still 1:1
            used += len(orig & set(idx.tolist()))
        assert lo <= used <= hi


def test_eval_sets_shrink_set_is_separate():
    table = make_table()
    extra = pd.DataFrame([("10", "shrink", 0, "shrink_cf", v, True) for v in range(3)], columns=table.columns)
    t = pd.concat([table, extra], ignore_index=True)
    sets = eval_sets(t)
    assert set(t.loc[sets["shrink"]].split) == {"shrink"}
    assert not (t.loc[np.r_[sets["val"], sets["dev"]]].split == "shrink").any()
    assert not (t.loc[EpochSampler(t, seed=42)(0)].split == "shrink").any()


def test_eval_sets_val_and_dev_composition():
    table = make_table()
    sets = eval_sets(table)

    kind, variant, split, label = table.kind, table.variant, table.split, table.label
    exp_val = table.index[
        (split == "val")
        & (
            (kind == "real_bonafide")
            | ((kind == "camera_only") & (variant == 0))
            | ((kind == "synthetic_attack") & (variant <= 1))
        )
    ]
    exp_dev = table.index[(split == "dev") | ((split == "val") & (kind == "real_bonafide"))]

    assert np.array_equal(np.sort(sets["val"]), np.sort(exp_val.to_numpy()))
    assert np.array_equal(np.sort(sets["dev"]), np.sort(exp_dev.to_numpy()))
    assert not (table.loc[sets["val"]].kind == "real_attack").any()
    assert not (table.loc[sets["val"]].split != "val").any()
    assert set(table.loc[sets["dev"]].kind) == {"real_bonafide", "real_attack"}


def _batch():
    return torch.rand(8, 3, 64, 64, generator=torch.Generator().manual_seed(0)).to(DEVICE)


def _only(**on):
    a = copy.deepcopy(CFG)
    for k in a:
        if k.startswith("p_"):
            a[k] = 0.0
    a.update(on)
    return a


def test_apply_keeps_shape_dtype_and_range():
    x = _batch()
    p = sample_params(np.random.default_rng([1, 2, 3]), 8, CFG)
    y = apply(x, p)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_apply_is_deterministic_for_fixed_rng_seed():
    x = _batch()
    y1 = apply(x, sample_params(np.random.default_rng([1, 2, 3]), 8, CFG))
    y2 = apply(x, sample_params(np.random.default_rng([1, 2, 3]), 8, CFG))
    assert torch.equal(y1, y2)


def test_all_probabilities_zero_is_identity():
    x = _batch()
    p = sample_params(np.random.default_rng(0), 8, _only())
    y = apply(x, p)
    assert torch.allclose(y, x, atol=1e-6)


def test_p_grey_one_gives_equal_channels():
    x = _batch()
    p = sample_params(np.random.default_rng(0), 8, _only(p_grey=1.0))
    y = apply(x, p)
    assert torch.allclose(y[:, 0], y[:, 1], atol=1e-6)
    assert torch.allclose(y[:, 1], y[:, 2], atol=1e-6)


def test_p_flip_one_is_horizontal_flip():
    x = _batch()
    p = sample_params(np.random.default_rng(0), 8, _only(p_flip=1.0))
    y = apply(x, p)
    assert torch.allclose(y, x.flip(-1), atol=1e-6)


def test_to_input_resizes_to_model_size():
    x = _batch()
    down = to_input(x, 32)
    assert down.shape == (8, 3, 32, 32)

    same = to_input(x, 64)
    assert torch.equal(same, x)

    up = to_input(x, 96)
    assert up.shape == (8, 3, 96, 96)
    assert float(up.min()) >= 0.0 and float(up.max()) <= 1.0


def test_sample_params_is_class_blind():
    assert list(inspect.signature(sample_params).parameters) == ["rng", "n", "a"]


PHOTO = {**CFG, "p_blur": 1.0, "max_blur": 2.0, "p_gamma": 1.0, "max_gamma": 1.5, "p_noise": 1.0, "max_noise": 0.03}


def test_photometric_off_keeps_rng_stream():
    """Configs without the photometric keys draw exactly what they drew before they existed."""
    a, b = np.random.default_rng(0), np.random.default_rng(0)
    p = sample_params(a, 8, CFG)
    q = sample_params(b, 8, {**CFG, "p_blur": 0.0, "p_gamma": 0.0, "p_noise": 0.0})
    assert set(p) == set(q) and not {"blur", "gamma", "noise"} & set(p)
    assert a.random() == b.random()


def test_photometric_shapes_range_and_determinism():
    x = torch.rand(4, 3, 32, 32)
    p = sample_params(np.random.default_rng(1), 4, PHOTO)
    y1, y2 = apply(x, p), apply(x, p)
    assert y1.shape == x.shape and y1.min() >= 0 and y1.max() <= 1
    assert torch.equal(y1, y2)


def test_blur_zero_sigma_is_identity_and_blur_smooths():
    from screenpad.train.augment import _blur

    x = torch.rand(2, 3, 32, 32)
    out = _blur(x, np.array([0.0, 2.0]))
    assert torch.allclose(out[0], x[0], atol=1e-5)
    assert out[1].std() < x[1].std()
