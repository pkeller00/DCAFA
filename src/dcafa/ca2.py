"""CA2 (Community Abundance, instance-level targets): one GLM of an instance-level target on community memberships.

    target ~ m_1 + ... + m_K (+ covariates)

with bag-clustered standard errors and BH FDR across the membership terms. Ported from the original DCAFA
implementation (``fit_ca_inst`` in https://github.com/wgrgwrght/DCAFA, MIT licence);
the numbers are unchanged, the output schema is made stable (``q``/``significant`` always present).
"""
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial, Gaussian, Poisson
from statsmodels.stats.multitest import multipletests

_FAMILIES = {"binomial": Binomial, "gaussian": Gaussian, "poisson": Poisson}


def fit_ca2_memberships(
    df: pd.DataFrame,
    target_col: str = "t",
    membership_cols: Optional[List[str]] = None,  # e.g. ['m_1', ..., 'm_K']; auto-detected if None
    covariates: Optional[List[str]] = None,       # e.g. ['x_num', 'site']
    bag_id_col: str = "bag_id",
    family: str = "binomial",                     # 'binomial' | 'gaussian' | 'poisson'
    drop_reference_if_exclusive: bool = True,     # drop the last membership column as reference
    cov_type: str = "cluster",                    # 'cluster' = clustered SEs by bag; else any statsmodels cov_type
) -> pd.DataFrame:
    """
    Fit CA2: does an instance's community membership predict its instance-level outcome?

    Parameters
    ----------
    df : instance-level DataFrame (one row per patch / cell) with the target, membership columns, covariates
        and the bag id.
    target_col : instance-level outcome column.
    membership_cols : hard (0/1) or soft ([0, 1]) membership columns. If None, ``m_*`` columns are used.
    covariates : extra RHS terms (Patsy syntax, e.g. ``"C(site)"``).
    bag_id_col : bag (patient / slide) column used for clustering the SEs.
    family : GLM family of the target.
    drop_reference_if_exclusive : exclusive memberships sum to one and are collinear with the intercept, so the
        last membership column is dropped and the others are effects relative to it. Set False for overlapping
        memberships.
    cov_type : "cluster" for bag-clustered SEs, otherwise passed to statsmodels unchanged.

    Returns
    -------
    Tidy DataFrame, one row per non-intercept term (membership terms first): term, is_membership, coef, se, p,
    ci_lo, ci_hi, effect_plot/effect_lo/effect_hi (exp scale for binomial/poisson, raw for gaussian), nobs,
    q, significant (BH FDR across membership terms only; NaN / False for covariates), deviance, pearson_chi2.
    """
    if membership_cols is None:
        membership_cols = [c for c in df.columns if c.startswith("m_")]
        membership_cols = sorted(membership_cols, key=lambda s: int(s.split("_")[1]))
    if covariates is None:
        covariates = []

    used_memberships = list(membership_cols)
    if drop_reference_if_exclusive and len(membership_cols) >= 2:
        used_memberships = membership_cols[:-1]

    rhs_terms = used_memberships + list(covariates)
    rhs = " + ".join(rhs_terms) if rhs_terms else "1"
    formula = f"{target_col} ~ {rhs}"

    model = smf.glm(formula=formula, data=df, family=_FAMILIES[family]())
    res = model.fit(
        cov_type=cov_type,
        cov_kwds={"groups": df[bag_id_col].values} if cov_type == "cluster" else None,
    )

    ci = res.conf_int()
    rows = []
    for term in res.params.index:
        if term == "Intercept":
            continue
        coef = float(res.params[term])
        lo, hi = ci.loc[term].values
        if family in ("binomial", "poisson"):
            eff, eff_lo, eff_hi = np.exp(coef), np.exp(lo), np.exp(hi)
        else:
            eff, eff_lo, eff_hi = coef, lo, hi
        rows.append({
            "term": term,
            "is_membership": term in membership_cols,
            "coef": coef,
            "se": float(res.bse[term]),
            "p": float(res.pvalues[term]),
            "ci_lo": lo,
            "ci_hi": hi,
            "effect_plot": eff,
            "effect_lo": eff_lo,
            "effect_hi": eff_hi,
            "nobs": res.nobs,
        })

    cols = ["term", "is_membership", "coef", "se", "p", "ci_lo", "ci_hi",
            "effect_plot", "effect_lo", "effect_hi", "nobs"]
    out = pd.DataFrame(rows, columns=cols)
    out["is_membership"] = out["is_membership"].astype(bool)

    out["q"] = np.nan
    out["significant"] = False
    mem_mask = out["is_membership"].to_numpy()
    if mem_mask.any():
        rej, qvals, _, _ = multipletests(out.loc[mem_mask, "p"].values, method="fdr_bh")
        out.loc[mem_mask, "q"] = qvals
        out.loc[mem_mask, "significant"] = rej

    out["deviance"] = res.deviance
    out["pearson_chi2"] = getattr(res, "pearson_chi2", np.nan)

    return out.sort_values(["is_membership", "term"], ascending=[False, True]).reset_index(drop=True)


def plot_ca2_forest(
    effects_df: pd.DataFrame,
    family: str = "binomial",
    title: str = "CA2: community effects on instance outcome",
    only_memberships: bool = True,
    outpath: Optional[str] = None,
):
    """Forest plot of CA2 membership effects (exp scale for binomial/poisson, raw coef for gaussian)."""
    df = effects_df[effects_df["is_membership"]] if only_memberships else effects_df
    if df.empty:
        print("[warn] No rows to plot.")
        return None
    df = df.sort_values("term", ascending=False).reset_index(drop=True)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7, 0.35 * len(df) + 1.5))
    ax.hlines(y, df["effect_lo"], df["effect_hi"], lw=2)
    ax.scatter(df["effect_plot"], y, zorder=3)
    if family in ("binomial", "poisson"):
        ax.axvline(1.0, ls="--", color="grey")
        ax.set_xlabel("Effect (exp(coef))")
    else:
        ax.axvline(0.0, ls="--", color="grey")
        ax.set_xlabel("Effect (coef, identity link)")
    ax.set_yticks(y, df["term"])
    ax.set_title(title)
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=160)
    return fig
