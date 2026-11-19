"""Write the regression references in tests/data/ from a given implementation.

    python tests/make_reference.py                                  # all references from this package
    python tests/make_reference.py --legacy /path/to/methods        # CA1/FA1 from a legacy methods/ folder
    python tests/make_reference.py --upstream /path/to/DCAFA        # CA2/FA2 from the original DCAFA.py

The committed CA1/FA1 references were produced from the original research code (dcafa/ca1_correction/methods,
2026-11-19) and the CA2/FA2 references from DCAFA.py in https://github.com/wgrgwrght/DCAFA (commit 0c8fe7b), so
the tests check that the packaged version reproduces both exactly. The CA1 voomclr reference has its `coding`
label renamed from "voomclr_mode_r" to "voomclr"; the numbers are unchanged. In the original CA2/FA2 code `significant` is
left empty for non-tested rows (covariates); the package reports False there, so references are normalised to that.
"""
import argparse
import contextlib
import io
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from synthetic import CASES, COMM, K, community_counts, patch_features, patch_memberships  # noqa: E402

MEMB = [f"m_{i}" for i in range(1, 5)]
FEATS = [f"x_{i}" for i in range(1, 5)]


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def load(legacy=None):
    if legacy:
        sys.path.insert(0, legacy)
        from CA1 import fit_ca1_multi_targets
        from FA1 import fit_fa1_univariate_multi_targets
    else:
        sys.path.insert(0, os.path.join(HERE, "..", "src"))
        from dcafa import fit_ca1_multi_targets, fit_fa1_univariate_multi_targets
    return fit_ca1_multi_targets, fit_fa1_univariate_multi_targets


def load_level2(upstream=None):
    """(fit_ca2, aggregate_bag_features, fit_fa2) from the package or from the original DCAFA.py."""
    if upstream:
        sys.path.insert(0, upstream)
        import DCAFA
        return DCAFA.fit_ca_inst, DCAFA.aggregate_bag_features, DCAFA.fit_fa_aggregate
    sys.path.insert(0, os.path.join(HERE, "..", "src"))
    from dcafa import aggregate_bag_features, fit_ca2_memberships, fit_fa2_aggregate_multi_targets
    return fit_ca2_memberships, aggregate_bag_features, fit_fa2_aggregate_multi_targets


def run_all(ca1, fa1):
    out = {}
    df = community_counts()
    for name, kw in CASES.items():
        with _quiet():
            r = ca1(df=df.copy(), K=K, target_cols=["Response"], community_cols=COMM, offset_col="n",
                    cov_type="HC1", outdir=None, plot=False, handle_collinearity="auto", ci_level=0.90,
                    fdr_alpha=0.10, **kw)["Response"]
        out[f"ca1_{name}"] = r
    with _quiet():
        f = fa1(df=patch_features(), target_cols=["t_r1"], bag_id_col="bag_id", families={"t_r1": "binomial"})
    out["fa1"] = f["t_r1"]
    return out


def run_level2(ca2, agg, fa2):
    out = {}
    df = patch_memberships()
    with _quiet():
        out["ca2_unadjusted"] = ca2(df.copy(), "t", MEMB, [], "bag_id", "binomial")
        out["ca2_age"] = ca2(df.copy(), "t", MEMB, ["age"], "bag_id", "binomial")
        design = agg(df, FEATS, MEMB, "bag_id")
        bags = df[["bag_id", "y", "age"]].drop_duplicates()
        out["fa2_aggregate_design"] = design
        out["fa2_global"] = fa2(design, bags, ["y"], [], True, False, None, "bag_id")["y"]
        out["fa2_global_community_age"] = fa2(design, bags, ["y"], ["age"], True, True, None, "bag_id")["y"]
    for name, tab in out.items():
        if "significant" in tab.columns:
            tab["significant"] = tab["significant"].fillna(False).astype(bool)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", help="folder with legacy CA1.py / FA1.py")
    ap.add_argument("--upstream", help="folder with the original DCAFA.py")
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    tables = {**run_all(*load(a.legacy)), **run_level2(*load_level2(a.upstream))}
    for name, tab in tables.items():
        tab.to_csv(os.path.join(HERE, "data", f"{name}.csv"), index=False)
        print(name, tab.shape)
