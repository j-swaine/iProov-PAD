from __future__ import annotations

import pandas as pd
import pytest

from screenpad.constants import ATTACK, BONAFIDE, LABEL_NAMES


@pytest.fixture
def toy_manifest() -> pd.DataFrame:
    """60 bona fide IDs, the first 6 of which also have a recaptured attack - same shape as the real data, scaled down."""
    rows = [{"id": f"{i:05d}", "label": BONAFIDE} for i in range(60)]
    rows += [{"id": f"{i:05d}", "label": ATTACK} for i in range(6)]
    df = pd.DataFrame(rows)
    df["label_name"] = df["label"].map(LABEL_NAMES)
    df["paired"] = df["id"].map(df.groupby("id")["label"].nunique()).eq(2)
    return df
