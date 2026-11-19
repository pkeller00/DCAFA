"""FA2 (Feature Association, bag-level targets): bag-level targets regressed on aggregated patch features.

    global means:     y ~ xg_1 + ... + xg_d (+ covariates)
    community means:  y ~ sum_k xk{k}_1 + ... + xk{k}_d (+ covariates)

with robust (HC) standard errors and BH FDR within the global-feature and community-feature families separately.
Ported from the original DCAFA implementation (``aggregate_bag_features`` / ``fit_fa_aggregate`` in
https://github.com/wgrgwrght/DCAFA, MIT licence); the numbers are unchanged, the
output schema is made stable (``q``/``significant`` always present).
"""
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial, Gaussian, Poisson
from statsmodels.stats.multitest import multipletests

_FAMILIES = {"binomial": Binomial, "gaussian": Gaussian, "poisson": Poisson}


def aggregate_bag_features(
    df_instances: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,     # x_*; auto-detected if None
    membership_cols: Optional[List[str]] = None,  # m_1..m_K (optional)
    bag_id_col: str = "bag_id",
    min_count_for_mean: float = 1,
) -> pd.DataFrame:
    """
    Aggregate instance-level features to one row per bag.

    Columns of the result:
        bag_id_col, n   : bag id and instance count
        xg_{feature}    : global mean of each feature in the bag (e.g. xg_x_1)
        c_{k}           : community "count" sum_p m_pk (only if membership_cols given)
        xk{k}_{j}       : community-local mean of x_j, weighted by m_pk (soft memberships allowed); NaN when the
                          bag's c_k < min_count_for_mean (only if membership_cols given)

    Feature and membership names are expected as ``<prefix>_<index>`` (e.g. ``x_3``, ``m_2``).
    """
    if feature_cols is None:
        feature_cols = [c for c in df_instances.columns if c.startswith("x_")]
        feature_cols = sorted(feature_cols, key=lambda s: int(s.split("_")[1]))
    has_memberships = membership_cols is not None and len(membership_cols) > 0

    g = df_instances.groupby(bag_id_col, observed=True)
    out = g.size().rename("n").to_frame().reset_index()
    out = out.merge(g[feature_cols].mean().add_prefix("xg_").reset_index(), on=bag_id_col, how="left")

    if has_memberships:
        cks = g[membership_cols].sum().reset_index()
        cks = cks.rename(columns={m: f"c_{m.split('_')[1]}" for m in membership_cols})
        out = out.merge(cks, on=bag_id_col, how="left")

        for kcol in membership_cols:
            k = kcol.split("_")[1]
            w = df_instances[kcol]
            wx = df_instances[feature_cols].mul(w, axis=0)
            wx[bag_id_col] = df_instances[bag_id_col].values
            sum_wx = wx.groupby(bag_id_col, observed=True)[feature_cols].sum()
            sum_w = w.groupby(df_instances[bag_id_col], observed=True).sum().reindex(sum_wx.index)
            den = sum_w.to_numpy()
            block = pd.DataFrame(index=sum_wx.index)
            for f in feature_cols:
                num = sum_wx[f].to_numpy()
                block[f"xk{k}_{f.split('_')[1]}"] = np.where(
                    den >= min_count_for_mean, num / np.where(den == 0, np.nan, den), np.nan)
            out = out.merge(block.reset_index(), on=bag_id_col, how="left")

    return out


def fit_fa2_aggregate_multi_targets(
    df_bag_design: pd.DataFrame,
    df_bags_targets_covs: pd.DataFrame,                 # bag_id + targets + optional covariates
    target_cols: List[str],
    covariate_cols: Optional[List[str]] = None,
    use_global_means: bool = True,
    use_community_means: bool = False,
    family_by_target: Optional[Dict[str, str]] = None,  # target -> 'binomial'|'gaussian'|'poisson'
    bag_id_col: str = "bag_id",
    cov_type: str = "HC1",
) -> Dict[str, pd.DataFrame]:
    """
    Fit FA2 on the bag-level design from :func:`aggregate_bag_features`.

    For each target one multivariable GLM is fitted on the selected aggregated features (``xg_*`` and/or
    ``xk*``) plus covariates. Unlike FA1 (univariate), effects here are adjusted for the other features.

    Returns
    -------
    dict target -> tidy table, one row per non-intercept term: target, term, is_global_feat, is_comm_feat, coef,
    se, p, q, ci_lo, ci_hi, effect_plot/effect_lo/effect_hi (exp scale for binomial/poisson), family, nobs,
    significant. q is BH within the global-feature terms and within the community-feature terms separately;
    covariate rows have q = NaN.
    """
    if covariate_cols is None:
        covariate_cols = []

    df = df_bag_design.merge(df_bags_targets_covs[[bag_id_col] + list(target_cols) + list(covariate_cols)],
                             on=bag_id_col, how="left", validate="one_to_one")

    global_feats = [c for c in df.columns if c.startswith("xg_")] if use_global_means else []
    community_feats = [c for c in df.columns if c.startswith("xk")] if use_community_means else []

    cols = ["target", "term", "is_global_feat", "is_comm_feat", "coef", "se", "p", "q", "ci_lo", "ci_hi",
            "effect_plot", "effect_lo", "effect_hi", "family", "nobs"]
    results: Dict[str, pd.DataFrame] = {}
    for tgt in target_cols:
        fam_name = family_by_target.get(tgt, "gaussian") if family_by_target else "gaussian"
        rhs_terms = global_feats + community_feats + list(covariate_cols)
        formula = f"{tgt} ~ {' + '.join(rhs_terms) if rhs_terms else '1'}"

        res = smf.glm(formula=formula, data=df, family=_FAMILIES[fam_name]()).fit(cov_type=cov_type)
        ci = res.conf_int()

        rows = []
        for term in res.params.index:
            if term == "Intercept":
                continue
            coef = float(res.params[term])
            lo, hi = ci.loc[term].values
            if fam_name in ("binomial", "poisson"):
                eff, eff_lo, eff_hi = np.exp(coef), np.exp(lo), np.exp(hi)
            else:
                eff, eff_lo, eff_hi = coef, lo, hi
            rows.append({
                "target": tgt, "term": term,
                "is_global_feat": term in global_feats, "is_comm_feat": term in community_feats,
                "coef": coef, "se": float(res.bse[term]), "p": float(res.pvalues[term]), "q": np.nan,
                "ci_lo": lo, "ci_hi": hi,
                "effect_plot": eff, "effect_lo": eff_lo, "effect_hi": eff_hi,
                "family": fam_name, "nobs": res.nobs,
            })
        eff_df = pd.DataFrame(rows, columns=cols)
        eff_df["significant"] = False

        for mask in (eff_df["is_global_feat"].to_numpy(bool), eff_df["is_comm_feat"].to_numpy(bool)):
            if mask.any():
                rej, qvals, _, _ = multipletests(eff_df.loc[mask, "p"].values, method="fdr_bh")
                eff_df.loc[mask, "q"] = qvals
                eff_df.loc[mask, "significant"] = rej

        results[tgt] = eff_df.sort_values("term").reset_index(drop=True)

    return results


def plot_fa2_forest(
    effects_df: pd.DataFrame,
    title: str = "FA2: aggregated feature effects",
    highlight: str = "global",   # 'global' (xg_*), 'community' (xk*) or 'all'
    outpath: Optional[str] = None,
):
    """Forest plot of one target's FA2 effects (exp scale for binomial/poisson, raw coef for gaussian)."""
    df = effects_df
    if highlight == "global":
        df = df[df["is_global_feat"]]
    elif highlight == "community":
        df = df[df["is_comm_feat"]]
    if df.empty:
        print(f"[warn] No rows to plot for highlight={highlight}.")
        return None
    fam = df["family"].iloc[0]
    df = df.sort_values("term", ascending=False).reset_index(drop=True)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(df) + 1.5))
    ax.hlines(y, df["effect_lo"], df["effect_hi"], lw=2)
    ax.scatter(df["effect_plot"], y, zorder=3)
    if fam in ("binomial", "poisson"):
        ax.axvline(1.0, ls="--", color="grey")
        ax.set_xlabel("Effect (exp(coef))")
    else:
        ax.axvline(0.0, ls="--", color="grey")
        ax.set_xlabel("Effect (coef)")
    ax.set_yticks(y, df["term"])
    ax.set_title(title)
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=160)
    return fig
