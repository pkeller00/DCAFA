"""Write the regression references in tests/data/ from a given implementation.

    python tests/make_reference.py                      # from this package (src/dcafa)
    python tests/make_reference.py /path/to/methods     # from a legacy methods/ folder (CA1.py, FA1.py)

The committed references were produced from the original research code (dcafa/ca1_correction/methods,
2026-11-19), so the tests check that the packaged version reproduces it exactly.
"""
import contextlib
import io
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from synthetic import CASES, COMM, K, community_counts, patch_features  # noqa: E402


def load(legacy=None):
    if legacy:
        sys.path.insert(0, legacy)
        from CA1 import fit_ca1_multi_targets
        from FA1 import fit_fa1_univariate_multi_targets
    else:
        sys.path.insert(0, os.path.join(HERE, "..", "src"))
        from dcafa import fit_ca1_multi_targets, fit_fa1_univariate_multi_targets
    return fit_ca1_multi_targets, fit_fa1_univariate_multi_targets


def run_all(ca1, fa1):
    out = {}
    df = community_counts()
    for name, kw in CASES.items():
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ca1(df=df.copy(), K=K, target_cols=["Response"], community_cols=COMM, offset_col="n",
                    cov_type="HC1", outdir=None, plot=False, handle_collinearity="auto", ci_level=0.90,
                    fdr_alpha=0.10, **kw)["Response"]
        out[f"ca1_{name}"] = r
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = fa1(df=patch_features(), target_cols=["t_r1"], bag_id_col="bag_id", families={"t_r1": "binomial"})
    out["fa1"] = f["t_r1"]
    return out


if __name__ == "__main__":
    ca1, fa1 = load(sys.argv[1] if len(sys.argv) > 1 else None)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    for name, tab in run_all(ca1, fa1).items():
        tab.to_csv(os.path.join(HERE, "data", f"{name}.csv"), index=False)
        print(name, tab.shape)
