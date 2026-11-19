"""The packaged CA1 / CA2 / FA1 / FA2 must reproduce the original implementations (references in tests/data/)."""
import os

import numpy as np
import pandas as pd
import pytest

from make_reference import load, load_level2, run_all, run_level2

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = {**run_all(*load()), **run_level2(*load_level2())}


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


def test_ca2_finds_planted_effects():
    # m_4 is the reference; m_2 raises the odds of t relative to it the most
    r = RESULTS["ca2_unadjusted"].set_index("term")
    assert list(r.index[r["is_membership"]]) == ["m_1", "m_2", "m_3"]
    assert r.loc["m_2", "effect_plot"] > r.loc["m_1", "effect_plot"] > 1 and r.loc["m_2", "significant"]


def test_fa2_finds_planted_effects():
    r = RESULTS["fa2_global"].set_index("term")
    assert r.loc["xg_x_1", "coef"] > 0 and r.loc["xg_x_1", "significant"]
    assert not r.drop("xg_x_1")["significant"].any()
