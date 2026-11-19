"""FA1 (Feature Association): univariate GLMs of each target on each patch-level feature, BH-FDR across features."""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial, Gaussian, Poisson
from statsmodels.stats.multitest import multipletests


def fit_fa1_univariate_multi_targets(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,    # e.g., ['x_1', ... 'x_d']
    target_cols: Optional[List[str]] = None,     # e.g., ['t_r1','t_r2',...]
    covariates: Optional[List[str]] = None,      # e.g., ['site','z_num']
    bag_id_col: str = "bag_id",
    families: Optional[Dict[str, str]] = None,   # map target -> 'binomial'|'gaussian'|'poisson'
    cov_type: str = "cluster"                    # clustered SEs by bag
) -> Dict[str, pd.DataFrame]:
    """
    Fit FA1‑style *univariate* GLMs:
        each target column is modelled separately as:
            target ~ x_j
        for each feature x_j, plus optional covariates if provided.
 
    Returns a dict of tidy tables (one per target) with BH‑FDR across *all* features,
    not just within each target.
 
    Includes robust error handling and skips degenerate cases.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c.startswith("x_")]
        feature_cols = sorted(feature_cols, key=lambda s: int(s.split("_")[1]))
    if target_cols is None:
        target_cols = [c for c in df.columns if c.startswith("t_r")]
        target_cols = sorted(target_cols, key=lambda s: int(s.split("t_r")[1]))
    if covariates is None:
        covariates = []
 
    def _fam(name: str):
        return {"binomial": Binomial(), "gaussian": Gaussian(), "poisson": Poisson()}[name]
 
    # 1) Collect all rows globally (for FDR across all feature‑target pairs)
    all_rows = []
 
    # Covariate string (same for all models)
    cov_rhs = " + ".join(covariates) if covariates else "1"
 
    for tgt in target_cols:
        fam_name = families[tgt] if families and tgt in families else "gaussian"
        fam = _fam(fam_name)
 
        rows = []
 
        for xcol in feature_cols:
            # formula: target ~ x + covariates
            if covariates:
                formula = f"{tgt} ~ {xcol} + {cov_rhs}"
            else:
                formula = f"{tgt} ~ {xcol}"
 
            # --- PRE‑FIT SANITY CHECKS ---
            df_pair = df[[bag_id_col, tgt, xcol]].dropna()
            if len(df_pair) == 0:
                print(f"⚠️  No valid data for {tgt} ~ {xcol}")
                continue
 
            y_vals = df_pair[tgt].values
            x_vals = df_pair[xcol].values
 
            if np.allclose(y_vals, 0.0) or np.var(y_vals) < 1e-10:
                print(f"⚠️  {tgt} is all 0 or near‑constant for {xcol} — skipping")
                continue
 
            if np.allclose(x_vals, 0.0) or np.var(x_vals) < 1e-10:
                print(f"⚠️  {xcol} is all 0 or near‑constant for {tgt} — skipping")
                continue
 
            # --- TRY FIT WITH EXCEPTION CATCHING ---
            try:
                model = smf.glm(formula=formula, data=df_pair, family=fam)
                res = model.fit(
                    cov_type="cluster" if cov_type == "cluster" else cov_type,
                    cov_kwds={"groups": df_pair[bag_id_col].values} if cov_type == "cluster" else None
                )
 
                # Skip if patsy / glm somehow failed to give a normal parameter
                if xcol not in res.params.index:
                    print(f"⚠️  {xcol} not in params for {tgt} — skipped")
                    continue
 
                coef = float(res.params[xcol])
                se = float(res.bse[xcol])
                p = float(res.pvalues[xcol])
                lo, hi = res.conf_int().loc[xcol].values
 
                if np.any(np.isnan([coef, se, p, lo, hi])):
                    print(f"⚠️  NaN coef/SE/p/CI for {tgt} ~ {xcol} — skipped")
                    continue
 
            except Exception as e:
                print(f"⚠️  Failed to fit {tgt} ~ {xcol}: {str(e)}")
                continue
 
            # --- TRANSFORM TO EFFECT PLOT SCALE ---
            if fam_name in ("binomial", "poisson"):
                eff = np.exp(coef)
                eff_lo = np.exp(lo)
                eff_hi = np.exp(hi)
            else:
                eff = coef
                eff_lo = lo
                eff_hi = hi
 
            row = {
                "target": tgt,
                "term": xcol,
                "is_feature": True,
                "coef": coef,
                "se": se,
                "p": p,
                "ci_lo": lo,
                "ci_hi": hi,
                "effect_plot": eff,
                "effect_lo": eff_lo,
                "effect_hi": eff_hi,
                "family": fam_name,
                "nobs": res.nobs,
            }
            rows.append(row)
            all_rows.append(row)
 
        # Every feature was skipped for this target (e.g. an unoccupied cluster
        # gives an all-zero indicator). Leave it out of the global pool; step 3
        # still returns an empty, correctly-columned table for it.
        if not rows:
            print(f"\u26a0\ufe0f  No usable fits for {tgt} \u2014 target had no variance")
 
    # 2) Global FDR across all feature‑target pairs
    _COLS = ["target", "term", "is_feature", "coef", "se", "p", "ci_lo", "ci_hi",
             "effect_plot", "effect_lo", "effect_hi", "family", "nobs"]
    # columns= keeps the schema when all_rows is empty, so the callers below and
    # downstream (pivot on "term"/"target") never see a column-less frame
    df_all = pd.DataFrame(all_rows, columns=_COLS)
 
    if not df_all.empty:
        rej, qvals, _, _ = multipletests(df_all["p"].values, method="fdr_bh")
        df_all["q"] = qvals
        df_all["significant"] = rej
    else:
        df_all["q"] = np.nan
        df_all["significant"] = False
 
    # 3) Rebuild per‑target dict with global q
    results: Dict[str, pd.DataFrame] = {}
    for tgt in target_cols:
        tgt_df = df_all[df_all["target"] == tgt].copy()
        tgt_df = tgt_df.sort_values("term").reset_index(drop=True)
        results[tgt] = tgt_df
 
    return results