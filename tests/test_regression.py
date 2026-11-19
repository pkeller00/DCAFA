"""The packaged CA1 / FA1 must reproduce the original research implementation (references in tests/data/)."""
import os

import numpy as np
import pandas as pd
import pytest

from make_reference import load, run_all

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = run_all(*load())


@pytest.mark.parametrize("name", sorted(RESULTS))
def test_matches_reference(name):
    ref = pd.read_csv(os.path.join(HERE, "data", f"{name}.csv"))
    got = RESULTS[name].reset_index(drop=True)
    assert list(got.columns) == list(ref.columns)
    assert len(got) == len(ref)
    for c in ref.columns:
        if pd.api.types.is_numeric_dtype(ref[c]):
            np.testing.assert_allclose(got[c].to_numpy(float), ref[c].to_numpy(float), rtol=1e-8, atol=1e-10,
                                       equal_nan=True, err_msg=f"{name}: column {c}")
        else:
            assert got[c].astype(str).tolist() == ref[c].astype(str).tolist(), f"{name}: column {c}"


def test_ca1_finds_planted_effects():
    r = RESULTS["ca1_nb_unadjusted"].set_index("response")
    r = r[r["term"] == "Response"] if "term" in r.columns else r
    assert r.loc["c_2", "percent_change"] > 0 and r.loc["c_5", "percent_change"] < 0
