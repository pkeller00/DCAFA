# =================================================================================================
# CA1 (Community Abundance Analysis): compositional handling + MoM vs fixed alpha
# =================================================================================================
# Requires: numpy, pandas, matplotlib, statsmodels, patsy, scipy
#
# New in this version:
#   - handle_collinearity: {"none","auto","deviation","reference"}
#       * "auto": detect rows summing to n and apply "deviation"
#       * "deviation": center target effects across communities to sum to zero
#         (with conservative SE/CI/p adjustment)
#       * "reference": report effects vs a reference community (auto = most abundant)
#
# Keeps:
#   - Original fixed-alpha GLM helper + docstring
#   - Stable per-community dispersion via Method-of-Moments (MoM) + GLM NB
#   - Diagnostics comparing MoM vs fixed on the same data
#
# voomclr backend: reproduces official R voomCLR 0.99.41 in pure Python (no R needed);
#   coef/t/p match R to ~1e-7 for voom_var_calc="analytical" and "empirical". edgeR dispersion,
#   limma squeezeVar, locfit and R lowess are ported in _edger.py; see docs/voomclr_port.md.
# =================================================================================================
 
import os
import warnings
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices, dmatrix
from statsmodels.stats.multitest import multipletests
from scipy.special import digamma, polygamma
from scipy.stats import norm, t as student_t

from ._edger import (edger_tagwise_dispersion, r_lowess, r_approxfun_rule2,  # pure-Python R ports
                        trigamma_inverse)
 
 
# =============================================================================
# Helper (kept): Fit Negative Binomial GLM with offset log(n) and fixed alpha
# =============================================================================
def _fit_nb_glm_one_formula(
    df: pd.DataFrame,
    formula: str,
    offset_col: str = "n",
    cov_type: str = "HC1",
    alpha: float = 1.0
):
    """
    Fit a single Negative Binomial GLM using statsmodels.
 
    Intuition:
    ----------
    - Each community count c_k is modeled as:
        c_k ~ NegBin(mu, alpha)
    - With log(mu) = linear predictor + log(n) offset
      where n = total bag size.
    - The offset ensures we model *relative abundance*
      rather than absolute counts.
 
    Parameters
    ----------
    df : DataFrame
        Data with counts, targets, and covariates.
        Actionable: Must contain the response column referenced on the left-hand side of
        `formula`, all predictors on the right-hand side, and the `offset_col`. Missing values
        should be handled upstream (drop/impute) to avoid fitting errors.
 
    formula : str
        Patsy formula (e.g., "c_1 ~ y_resp + age + site").
        Actionable: Ensure the left-hand side is a **single** community count column (e.g., "c_7").
        Use `C(var)` in your data prep or cast categorical predictors to category dtype.
 
    offset_col : str, default="n"
        Column name with bag total size (log used as offset).
        Actionable: Values must be ≥1; zeros are automatically clamped to 1 to avoid log(0).
 
    cov_type : {"nonrobust","HC0","HC1","HC2","HC3"}, default="HC1"
        Covariance estimator for robust standard errors.
        Actionable: Keep "HC1" for general use; consider cluster-robust SEs (not implemented here)
        if you have repeated measures per bag.
 
    alpha : float, default=1.0
        Dispersion parameter for Negative Binomial (fixes variance scaling).
        Actionable: Use 0.0 to recover a Poisson GLM; set to ~0.5–1.0 for mild overdispersion,
        or estimate α per community via the MoM path elsewhere.
 
    Returns
    -------
    res : statsmodels GLMResults
        Fitted model object with coef/SE/CI/p for all RHS terms (except Intercept).
 
    Notes
    -----
    • This routine does **not** estimate alpha; it **uses** the provided alpha.
    • If you need per-community dispersion, use the MoM estimator + `fit_nb_glm_with_alpha`.
    """
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.NegativeBinomial(alpha=alpha),
        offset=np.log(df[offset_col].clip(lower=1).values)
    )
    res = model.fit(cov_type=cov_type)
    return res
 
 
# =============================================================================
# Stable per-community alpha estimation via Method-of-Moments (MoM)
# =============================================================================
def estimate_nb_alpha_moments(
    df: pd.DataFrame,
    formula: str,
    offset_col: str = "n",
    winsor: float = 0.98,
    min_mu: float = 1e-8,
) -> Tuple[float, np.ndarray]:
    """
    Estimate NB2 dispersion alpha using a Poisson GLM mean, then Pearson residuals (MoM).
 
    Steps:
    1) Fit Poisson GLM with log(exposure) offset to get mu-hat.
    2) Compute Pearson residuals: r_i^2 = (y_i - mu_i)^2 / mu_i.
    3) Method-of-moments for NB2: E[r_i^2] ≈ 1 + alpha * mu_i
       ⇒ alpha_hat ≈ mean(clip(r_i^2, q_{winsor}) - 1) / mean(mu_i).
 
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the response and predictors identified by `formula` and `offset_col`.
        Actionable: Clean NA values prior to calling; ensure `offset_col` is present and ≥1.
 
    formula : str
        Patsy formula with a **single** response (e.g., "c_7 ~ y + covs").
        Actionable: Use the same RHS you will use for the NB fit (target + covariates).
 
    offset_col : str, default="n"
        Exposure column name (bag size).
        Actionable: Must be positive; zeros are automatically clamped to 1 for the log-offset.
 
    winsor : float in (0,1), default=0.98
        Upper-quantile clip for r_i^2 to reduce the influence of extreme outliers.
        Actionable: Increase to 0.995 if your data have occasional spiky bags; decrease to 0.95
        if you want less robustification.
 
    min_mu : float, default=1e-8
        Lower bound for fitted means μ_i to avoid numerical divisions by ~0.
        Actionable: Leave at default unless you encounter numerical warnings.
 
    Returns
    -------
    alpha_hat : float
        Estimated dispersion (≥0). If ≈0, data are consistent with Poisson variance.
 
    mu_hat : np.ndarray
        The fitted Poisson means (useful for diagnostics).
 
    Practical guidance
    ------------------
    • If alpha_hat ≈ 0 across many communities, Poisson is likely sufficient.
    • If alpha_hat > 0, carry it into the NB GLM via `fit_nb_glm_with_alpha` for calibrated SEs.
    """
    y_df, X_df = dmatrices(formula, df, return_type="dataframe")
    y = np.asarray(y_df).ravel()
    exposure = df[offset_col].clip(lower=1).to_numpy()
 
    pois = sm.GLM(y, X_df, family=sm.families.Poisson(), exposure=exposure).fit()
    mu = np.maximum(pois.fittedvalues.values, min_mu)
 
    r2 = (y - mu) ** 2 / mu
    q = np.quantile(r2, winsor)
    r2_w = np.minimum(r2, q)
    w = np.ones_like(mu)
 
    num = np.sum((r2_w - 1.0) * w)
    den = np.sum(mu * w)
    alpha_hat = float(np.clip(num / den if den > 0 else 0.0, a_min=0.0, a_max=np.inf))
    return alpha_hat, mu
 
 
def fit_nb_glm_with_alpha(
    df: pd.DataFrame,
    formula: str,
    offset_col: str,
    alpha_hat: float,
    cov_type: str = "HC1",
    refit_once: bool = True,
    min_mu: float = 1e-8,
):
    """
    Fit GLM-NegativeBinomial with a given alpha; optionally refit once with updated mu.
 
    Why:
    ----
    Fixing α to a MoM estimate avoids NB MLE numerical issues while still honoring
    overdispersion at the community level.
 
    Parameters
    ----------
    df : pd.DataFrame
        Bag-level dataset including response, predictors, and `offset_col`.
        Actionable: Pre-clean NA, ensure integer counts for the response.
 
    formula : str
        Patsy formula with a **single** community response (e.g., "c_3 ~ y + covs").
        Actionable: Match exactly the RHS used for MoM estimation for consistency.
 
    offset_col : str
        Exposure column name (e.g., "n").
        Actionable: Values must be ≥1; zeros are clamped internally to 1.
 
    alpha_hat : float
        Pre-estimated dispersion (≥0). If 0, this effectively becomes Poisson.
        Actionable: Use `estimate_nb_alpha_moments` per community to get this.
 
    cov_type : {"nonrobust","HC0","HC1","HC2","HC3"}, default="HC1"
        Sandwich covariance type for SEs.
        Actionable: Keep "HC1" unless you need a different small-sample correction.
 
    refit_once : bool, default=True
        If True, recompute a MoM α from the first NB fit’s μ, then refit NB once.
        Actionable: Keep True for extra stability; set False for speed.
 
    min_mu : float, default=1e-8
        Lower bound on fitted means when recomputing residuals.
        Actionable: Leave at default unless encountering numerical issues.
 
    Returns
    -------
    res : GLMResults (NegativeBinomial)
        Fitted NB GLM result with coef/SE/CI/p for RHS terms (except Intercept).
 
    Notes
    -----
    • This function does **not** change model terms or design; it only sets α.
    • If the first fit yields μ very close to zero for many rows, consider raising `min_mu`.
    """
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.NegativeBinomial(alpha=max(alpha_hat, 1e-12)),
        offset=np.log(df[offset_col].clip(lower=1).values)
    )
    res = model.fit(cov_type=cov_type)
 
    if not refit_once:
        return res
 
    mu1 = np.maximum(res.fittedvalues.values, min_mu)
    y = df[formula.split("~")[0].strip()].values
    r2 = (y - mu1) ** 2 / mu1
    num = np.sum(r2 - 1.0)
    den = np.sum(mu1)
    alpha2 = float(np.clip(num / den if den > 0 else 0.0, 0.0, np.inf))
 
    model2 = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.NegativeBinomial(alpha=max(alpha2, 1e-12)),
        offset=np.log(df[offset_col].clip(lower=1).values)
    )
    res2 = model2.fit(cov_type=cov_type)
    return res2
 
 
# =============================================================================
# Compositionality handling helpers
# =============================================================================
def _detect_collinearity(
    df: pd.DataFrame,
    kept_cols: List[str],
    offset_col: str,
    r2_threshold: float = 0.999,      # near-compositionality if R^2 of sumC ~ n exceeds this
    rel_err_thresh: float = 1e-3,     # or mean relative error below this
    cond_threshold: float = 1e10      # condition number threshold for near-singularity
) -> Tuple[bool, str]:
    """
    Detect (near-)collinearity between community counts and the total n.
 
    Returns
    -------
    (flag, reason) : (bool, str)
        flag   = True if exact or near collinearity is detected
        reason = "exact", "r2", "relerr", or "condition" to indicate what triggered it
 
    Actionable guidance
    -------------------
    • If True, treat as compositional for coding purposes ("deviation" or "reference").
    • Tighten/loosen thresholds depending on dataset size/noise (see parameters below).
 
    Parameters
    ----------
    df : pd.DataFrame
        Must include kept community columns and `offset_col`.
 
    kept_cols : list[str]
        Community count columns considered (post-filtering).
 
    offset_col : str
        Name of total bag size column.
 
    r2_threshold : float, default=0.999
        If R^2 of sum_counts ~ n exceeds this, flag near-collinearity.
 
    rel_err_thresh : float, default=1e-3
        If mean(|sum_counts - n| / max(n,1)) ≤ this, flag near-collinearity.
 
    cond_threshold : float, default=1e10
        If the (scaled) design has condition number above this, flag near-singularity.
    """
    import numpy as np
 
    # 1) Exact compositionality (your original check).
    sum_counts = df[kept_cols].sum(axis=1).to_numpy()
    nvec = df[offset_col].to_numpy()
    exact = np.allclose(sum_counts, nvec, rtol=1e-6, atol=1e-6)
    if exact:
        return True, "exact"
 
    # 2) Near-equality: R^2 of sum_counts ~ n close to 1
    x = nvec.astype(float)
    y = sum_counts.astype(float)
    xm = x - x.mean()
    ym = y - y.mean()
    ssx = np.sum(xm * xm)
    ssy = np.sum(ym * ym)
    r2 = (np.sum(xm * ym) ** 2) / (ssx * ssy + 1e-12) if ssx > 0 and ssy > 0 else 0.0
    if r2 >= r2_threshold:
        return True, "r2"
 
    # 3) Mean relative error small
    denom = np.maximum(nvec, 1.0)
    rel_err = np.mean(np.abs(sum_counts - nvec) / denom)
    if rel_err <= rel_err_thresh:
        return True, "relerr"
 
    # 4) Condition number of [counts | n] (scaled) is huge → near-singularity
    #    Center/scale columns to avoid trivial scale-driven condition inflation.
    X = df[kept_cols + [offset_col]].to_numpy(dtype=float)
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    scale = np.nanstd(Xc, axis=0, ddof=1, keepdims=True)
    scale[scale == 0] = 1.0
    Xcs = Xc / scale
    # cond = ratio of largest to smallest singular value
    u, s, vT = np.linalg.svd(Xcs, full_matrices=False)
    cond = (s[0] / (s[-1] + 1e-12)) if s.size > 1 else np.inf
    if cond >= cond_threshold:
        return True, "condition"
 
    return False, "none"
 
 
def _apply_deviation_coding(effects: pd.DataFrame, term: str, ci_level: float = 0.95) -> pd.DataFrame:
    """
    Center the term's coefficients across communities (sum-to-zero coding).
 
    Parameters
    ----------
    effects : pd.DataFrame
        Output rows from the per-community fits for one target (mix of terms).
 
    term : str
        The target/predictor whose effects need centering (e.g., "y_resp").
 
    Returns
    -------
    pd.DataFrame
        Effects with the given term’s coefficients centered across communities and
        SE/CI/p updated via a conservative variance-centering approximation.
 
    Actionable guidance
    -------------------
    • Use this when counts are compositional (rows sum to n).
    • Interpretation: positive values indicate communities that increase relative
      to the average community as the term increases, and vice versa.
    """
    ef = effects.copy()
    m = ef["term"] == term
    if not m.any():
        return ef
 
    beta = ef.loc[m, "coef_log"].to_numpy()
    se = ef.loc[m, "se"].to_numpy()
    K = len(beta)
 
    beta_c = beta - np.nanmean(beta)
 
    # Approximate variance adjustment (no covariances available):
    # Var(centered_i) ≈ Var_i * (1 - 2/K) + mean(Var)/K
    var = se ** 2
    mean_var = np.nanmean(var)
    var_c = var * (1.0 - 2.0 / K) + (mean_var / K)
    var_c = np.clip(var_c, 0.0, np.inf)
    se_c = np.sqrt(var_c)
 
    # Recompute Wald z, p, CI (normal)
    z = np.divide(beta_c, se_c, out=np.zeros_like(beta_c), where=se_c > 0)
    p = 2 * norm.sf(np.abs(z))
    zcrit = norm.ppf(0.5 + ci_level / 2)
    lo = beta_c - zcrit * se_c
    hi = beta_c + zcrit * se_c
 
    ef.loc[m, "coef_log"] = beta_c
    ef.loc[m, "se"] = se_c
    ef.loc[m, "p"] = p
    ef.loc[m, "ci_lo"] = lo
    ef.loc[m, "ci_hi"] = hi
    ef.loc[m, "exp_coef"] = np.exp(beta_c)
    ef.loc[m, "exp_ci_lo"] = np.exp(lo)
    ef.loc[m, "exp_ci_hi"] = np.exp(hi)
    ef.loc[m, "percent_change"] = (ef.loc[m, "exp_coef"] - 1.0) * 100.0
    ef.loc[m, "coding"] = "deviation"
    return ef
 
 
def _apply_reference_coding(effects: pd.DataFrame, term: str, ref_response: str,
                            ci_level: float = 0.95) -> pd.DataFrame:
    """
    Express term effects relative to a reference community response (e.g., 'c_3').
 
    Parameters
    ----------
    effects : pd.DataFrame
        Output rows from the per-community fits for one target (mix of terms).
 
    term : str
        The target/predictor to re-express (e.g., "y_resp").
 
    ref_response : str
        The response column name to use as the reference (e.g., "c_3").
 
    Returns
    -------
    pd.DataFrame
        Effects where the selected term is transformed to β_k - β_ref on log-scale,
        with SE/CI/p via independent-difference approximation (conservative).
 
    Actionable guidance
    -------------------
    • Choose a **stable, abundant** reference (or let the caller auto-pick).
    • Interpretation becomes “fold-change versus reference community.”
    • The reference row itself is set to 0 (log-scale) with NA SE/CI/p.
    """
    ef = effects.copy()
    m = ef["term"] == term
    if not m.any():
        return ef
 
    # rows for target term
    E = ef.loc[m].copy()
    # find ref row
    ref_row = E[E["response"] == ref_response]
    if ref_row.empty:
        # nothing to do if reference not present
        return ef
 
    ref_beta = float(ref_row["coef_log"].iloc[0])
    ref_se = float(ref_row["se"].iloc[0])
    ref_var = ref_se ** 2
 
    # compute diffs for all (including ref → set to 0 with NA inferential stats)
    diffs = E["coef_log"].values - ref_beta
    var_diff = (E["se"].values ** 2) + ref_var  # ignore covariance (conservative)
    se_diff = np.sqrt(np.clip(var_diff, 0.0, np.inf))
 
    # fill back
    ef.loc[E.index, "coef_log"] = diffs
    ef.loc[E.index, "se"] = se_diff
    z = np.divide(diffs, se_diff, out=np.zeros_like(diffs), where=se_diff > 0)
    p = 2 * norm.sf(np.abs(z))
    zcrit = norm.ppf(0.5 + ci_level / 2)
    lo = diffs - zcrit * se_diff
    hi = diffs + zcrit * se_diff
 
    ef.loc[E.index, "p"] = p
    ef.loc[E.index, "ci_lo"] = lo
    ef.loc[E.index, "ci_hi"] = hi
    ef.loc[E.index, "exp_coef"] = np.exp(diffs)
    ef.loc[E.index, "exp_ci_lo"] = np.exp(lo)
    ef.loc[E.index, "exp_ci_hi"] = np.exp(hi)
    ef.loc[E.index, "percent_change"] = (ef.loc[E.index, "exp_coef"] - 1.0) * 100.0
    ef.loc[E.index, "coding"] = f"reference:{ref_response}"
 
    # For the reference itself: set estimate to exactly 0 and mark p/CI appropriately
    ef.loc[(ef["term"] == term) & (ef["response"] == ref_response), ["coef_log", "se"]] = [0.0, np.nan]
    ef.loc[(ef["term"] == term) & (ef["response"] == ref_response), ["p", "ci_lo", "ci_hi",
                                                                     "exp_coef", "exp_ci_lo", "exp_ci_hi",
                                                                     "percent_change"]] = [np.nan, np.nan, np.nan,
                                                                                           1.0, np.nan, np.nan, 0.0]
    return ef


# =============================================================================
# voomCLR backend helpers
# =============================================================================
# CLR -> voom precision weights -> weighted linear model per community ->
# modal bias correction -> limma empirical-Bayes variance moderation.
# Reproduces official R voomCLR 0.99.41 (coef/t/p to ~1e-7) in pure Python; the
# edgeR, limma, locfit and R stats pieces are ported in _edger.py.
# =============================================================================
def _voomclr_clr_transform(
    counts: np.ndarray,
    pseudocount: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CLR transform of a bags x communities count matrix:
        E_ik = log((c_ik + pseudocount) / g_i),  g_i = geometric mean of row i.
    The original voomCLR R implementation uses pseudocount=0.5.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2:
        raise ValueError("counts must be a 2D matrix (bags x communities).")
    if np.any(~np.isfinite(counts)):
        raise ValueError("Community counts contain NaN or infinite values.")
    if np.any(counts < 0):
        raise ValueError("Community counts must be non-negative.")
    if pseudocount <= 0:
        raise ValueError("pseudocount must be > 0.")

    x = counts + pseudocount
    geo_means = np.exp(np.mean(np.log(x), axis=1))
    E = np.log(x / geo_means[:, None])
    return E, geo_means


# Port of voomCLR:::.getMode = modeest::mlv(sqrt(n)*beta, method="meanshift",
# kernel="gaussian")/sqrt(n). Meanshift climbs from shorth(x) to the nearest KDE peak with
# bandwidth bw.nrd0, so it can stop at a local peak.
def _voomclr_mode(x: np.ndarray, n: Optional[int] = None) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if x.size == 1 or np.std(x) < 1e-12:
        return float(np.mean(x))
    scale = np.sqrt(float(n)) if n is not None and n > 0 else 1.0
    xs = scale * x
    # modeest::shorth: mean of the shortest window of ceiling(m/2) sorted points
    # (on a tie R's .deal.ties takes the mean position; the first window is used here)
    y = np.sort(xs)
    m = y.size
    k = int(np.ceil(m / 2)) - 1
    i = int(np.argmin(y[k:] - y[:m - k]))
    par = float(np.mean(y[i:i + k + 1]))
    # stats::bw.nrd0
    hi = float(np.std(xs, ddof=1))
    lo = min(hi, float(np.subtract(*np.percentile(xs, [75, 25]))) / 1.34)
    lo = lo or hi or abs(float(xs[0])) or 1.0
    bw = 0.9 * lo * m ** -0.2
    tol = np.sqrt(np.finfo(float).eps)
    for _ in range(1000):
        w = np.exp(-0.5 * ((xs - par) / bw) ** 2)
        M = float(np.sum(xs * w) / np.sum(w))
        done = abs(M / par - 1.0) < tol
        par = M
        if done:
            break
    return par / scale


# voomCLR:::.estimateNBDispersion: edgeR TMM norm factors, then estimateDisp, tagwise
# (line-by-line Python port in _edger.py; matches R edgeR 4.10.6 to ~1e-12).
def _voomclr_edger_dispersion(counts: np.ndarray, design: np.ndarray) -> np.ndarray:
    phi = edger_tagwise_dispersion(counts, design)
    if phi.shape != (np.asarray(counts).shape[1],):
        raise RuntimeError(f"edgeR port returned {phi.shape} dispersions for {np.asarray(counts).shape[1]} communities.")
    return phi


# limma::squeezeVar -> fitFDist (no covariate, equal residual df), line by line:
# s2 floored at 1e-5*median, df_prior = Inf whenever evar <= 0, and then s2_prior = mean(s2).
def _voomclr_ebayes_variances(s2: np.ndarray, df_residual: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Returns (posterior_variances, df_prior, s2_prior)."""
    var = np.asarray(s2, dtype=float)
    df = np.asarray(df_residual, dtype=float)
    if len(var) < 3:
        return var.copy(), 0.0, float(np.mean(var))
    x = np.maximum(var, 0)
    m = np.median(x)
    x = np.maximum(x, 1e-5 * (m if m > 0 else 1.0))
    e = np.log(x) + np.log(df / 2.0) - digamma(df / 2.0)
    emean = float(np.mean(e))
    evar = float(np.sum((e - emean) ** 2) / (len(e) - 1)) - float(np.mean(polygamma(1, df / 2.0)))
    if evar > 0:
        df_prior = 2.0 * trigamma_inverse(evar)
        s2_prior = float(np.exp(emean - (np.log(df_prior / 2.0) - digamma(df_prior / 2.0))))
        return (df * var + df_prior * s2_prior) / (df + df_prior), float(df_prior), s2_prior
    s2_prior = float(np.mean(x))
    return np.full_like(var, s2_prior), float(np.inf), s2_prior


def _voomclr_transform_and_weights(
    counts: np.ndarray,
    design: np.ndarray,
    pseudocount: float = 0.5,
    var_calc: str = "analytical",
    var_distribution: str = "NB",
    span: float = 0.8
):
    """
    Python port of voomCLR(): returns (E, weights, fitted_counts, dispersion).
    """
    counts = np.asarray(counts, dtype=float)
    design = np.asarray(design, dtype=float)
    if counts.shape[0] != design.shape[0]:
        raise ValueError("counts and design must contain the same bags.")
    n_bags, K = counts.shape
    if K < 2:
        raise ValueError("voomCLR requires at least two communities.")

    var_calc = str(var_calc).lower()
    var_distribution = str(var_distribution).lower()
    if var_calc not in {"analytical", "empirical"}:
        raise ValueError("voom_var_calc must be 'analytical' or 'empirical'.")
    if var_distribution not in {"nb", "poisson"}:
        raise ValueError("voom_var_distribution must be 'NB' or 'poisson'.")

    E, geo_means = _voomclr_clr_transform(counts, pseudocount=pseudocount)

    # Initial unweighted lmFit analogue
    fitted_values = np.zeros_like(E)
    residual_sd = np.zeros(K)
    Amean = np.zeros(K)
    for k in range(K):
        fit = sm.OLS(E[:, k], design).fit()
        fitted_values[:, k] = fit.fittedvalues
        residual_sd[k] = np.sqrt(max(float(fit.mse_resid), 1e-12))
        Amean[k] = float(np.mean(E[:, k]))

    fitted_counts = np.exp(fitted_values) * geo_means[:, None]
    fitted_logcount = np.log(np.clip(fitted_counts, 1e-12, None))
    dispersion = None

    if var_calc == "empirical":
        # Trend by stats::lowess (iter=3, delta=0.01*range) over the communities with
        # non-zero total, read off with approxfun(rule=2, ties=mean).
        sx = Amean + np.mean(np.log(geo_means))
        sy = np.sqrt(residual_sd)
        nz = counts.sum(axis=0) > 0
        lx, ly = r_lowess(sx[nz], sy[nz], f=span)
        weights = 1.0 / r_approxfun_rule2(lx, ly)(fitted_logcount) ** 4
    else:
        factor = ((K - 1.0) / K) ** 2
        if var_distribution == "poisson":
            est_var = factor * (1.0 / np.clip(fitted_counts, 1e-12, None))
        else:
            dispersion = _voomclr_edger_dispersion(counts, design)
            est_var = factor * (1.0 / np.clip(fitted_counts, 1e-12, None) + dispersion[None, :])
        weights = 1.0 / np.clip(est_var, 1e-12, None)

    return E, weights, fitted_counts, dispersion


def _fit_voomclr_one_target(
    df: pd.DataFrame,
    target: str,
    rhs: str,
    kept_cols: List[str],
    ci_level: float = 0.90,
    var_calc: str = "analytical",
    var_distribution: str = "NB",
    pseudocount: float = 0.5,
    span: float = 0.8
) -> List[dict]:
    """
    Fit voomCLR jointly over all kept communities for one target and return
    rows in the same schema as the NB path of `fit_ca1_multi_targets`.

    Effects are bias-corrected log-ratios: exp(coef) is the fold change of a
    community relative to the modal (typical) community, so no further
    deviation/reference coding is applied.
    """
    design_df = dmatrix(rhs, data=df, return_type="dataframe", NA_action="drop")
    if design_df.shape[0] == 0:
        raise ValueError("No complete observations remain after building the design.")
    design = design_df.to_numpy(dtype=float)
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The CA1 design matrix is rank deficient. Check target/covariate collinearity.")

    counts = df.loc[design_df.index, kept_cols].to_numpy(dtype=float)
    if np.any(counts.sum(axis=1) <= 0):
        raise ValueError("Every analysed bag must contain at least one community count.")

    E, weights, _, dispersion = _voomclr_transform_and_weights(
        counts=counts, design=design, pseudocount=pseudocount,
        var_calc=var_calc, var_distribution=var_distribution, span=span
    )

    # Weighted lmFit analogue
    K = len(kept_cols)
    p = design.shape[1]
    coef_raw = np.zeros((K, p))
    s2 = np.zeros(K)
    df_resid = np.zeros(K)
    cov_unscaled = []
    for k in range(K):
        fit = sm.WLS(E[:, k], design, weights=weights[:, k]).fit()
        coef_raw[k, :] = fit.params
        s2[k] = max(float(fit.ssr / fit.df_resid), 1e-12)
        df_resid[k] = float(fit.df_resid)
        XtWX = design.T @ (weights[:, k, None] * design)
        cov_unscaled.append(np.linalg.pinv(XtWX))

    # applyBiasCorrection(): subtract the across-community mode per coefficient
    n = design.shape[0]
    bias = np.array([_voomclr_mode(coef_raw[:, j], n=n) for j in range(p)])
    coef = coef_raw - bias[None, :]

    s2_post, df_prior, s2_prior = _voomclr_ebayes_variances(s2, df_resid)
    # limma::eBayes caps df.total at the pooled residual df
    df_cap = float(np.sum(df_resid))

    rows = []
    for j, term in enumerate(design_df.columns):
        if term == "Intercept":
            continue
        for k, response in enumerate(kept_cols):
            beta = float(coef[k, j])
            se = float(np.sqrt(max(s2_post[k] * cov_unscaled[k][j, j], 1e-20)))
            df_total = float(min(df_resid[k] + df_prior, df_cap))
            t_stat = beta / se
            pval = float(2.0 * student_t.sf(abs(t_stat), df=df_total))
            crit = float(student_t.ppf(0.5 + ci_level / 2, df=df_total))
            ci_lo, ci_hi = beta - crit * se, beta + crit * se

            rows.append({
                "target": target, "community": f"k={k + 1}", "response": response,
                "term": str(term), "coef_log": beta, "se": se, "p": pval,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "exp_coef": np.exp(beta),
                "exp_ci_lo": np.exp(ci_lo),
                "exp_ci_hi": np.exp(ci_hi),
                "percent_change": (np.exp(beta) - 1.0) * 100.0,
                "dispersion_alpha": float(dispersion[k]) if dispersion is not None else np.nan,
                "converged": True, "nobs": float(n),
                "coding": "voomclr",
                # voomCLR diagnostics
                "raw_coef": float(coef_raw[k, j]), "bias": float(bias[j]),
                "t": float(t_stat), "df_resid": float(df_resid[k]),
                "df_prior": float(df_prior), "s2_prior": float(s2_prior),
            })
    return rows


# =============================================================================
# CA1 main (supports "moments" for fitted alpha or "fixed" for fixed alpha)
# =============================================================================
def fit_ca1_multi_targets(
    df: pd.DataFrame,
    K: int,
    target_cols: List[str],
    covariates: Optional[List[str]] = None,
    community_cols: Optional[List[str]] = None,
    offset_col: str = "n",
    cov_type: str = "HC1",
    handle_collinearity: str = "auto",  # "none","auto","deviation","reference"
    reference_response: Optional[str] = None,  # e.g., "c_3"; if None and reference mode, pick most abundant
    ci_level: float = 0.90,
    fdr_alpha: float = 0.05,
    outdir: Optional[str] = None,
    plot: bool = True,
    scale: str = "percent",   # "percent" or "ratio"
    min_total: int = 10,
    min_bags: int = 1,
    dispersion_mode: str = "moments",   # "moments" (per-community MoM alpha) or "fixed"
    alpha_fixed: float = 1.0,
    backend: str = "nb",                # "nb" (per-community NB GLM) or "voomclr"
    voom_var_calc: str = "analytical",
    voom_var_distribution: str = "NB",
    voom_pseudocount: float = 0.5,
    voom_span: float = 0.8
) -> Dict[str, pd.DataFrame]:
    """
    Fit CA1 (Community Abundance analysis) for multiple bag-level targets.
 
    Intuition
    ---------
    For each target and each community (neighborhood), regress the community's
    counts on the target + covariates with a log(n) offset so effects are on
    *relative abundance*. This function supports:
      - Per-community dispersion estimation ("moments"): robust MoM + GLM NB.
      - Fixed-alpha GLM ("fixed"): faster and mirrors a global-α approach.
      - Compositional collinearity handling ("auto","deviation","reference"): post-hoc
        reparameterization of the target effect across communities to respect unit-sum.
 
    Parameters
    ----------
    df : pd.DataFrame
        Bag-level data with:
          • "n" (offset/exposure), ≥1
          • Community count columns "c_1..c_K" (subset OK)
          • All `target_cols` and `covariates`
        Actionable: Ensure counts are non-negative integers; drop/impute NAs before fitting.
 
    K : int
        Maximum community index (defines expected columns up to c_K).
        Actionable: You can pass a larger K than observed; function filters to present columns.
 
    target_cols : list[str]
        Names of target variables to test (e.g., ["y_resp", "y_score"]).
        Actionable: Binary targets should be 0/1; for multi-level factors, encode dummies or use C() upstream.
 
    covariates : list[str] or None, default=None
        Extra RHS predictors (e.g., ["age", "C(site)"]).
        Actionable: Cast categorical covariates to category dtype or pre-dummy them.
 
    offset_col : str, default="n"
        Exposure column used as log-offset.
        Actionable: Must be ≥1; zeros are clamped internally to 1.
 
    cov_type : {"nonrobust","HC0","HC1","HC2","HC3"}, default="HC1"
        Robust SE type.
        Actionable: Keep "HC1" unless you have a reason to change small-sample corrections.
 
    handle_collinearity : {"none","auto","deviation","reference"}, default="auto"
        Strategy for the **unit-sum** dependency across communities.
        - "auto": detect compositional rows (sum c_k ≈ n); if True → "deviation", else "none".
        - "deviation": center target coefficients across communities (sum-to-zero).
        - "reference": re-express as differences vs one reference community.
        - "none": no special handling.
        Actionable: Keep "auto" for safety; switch to "reference" if a specific baseline is desired.
 
    reference_response : str or None, default=None
        When using "reference", which community column (e.g., "c_3") to use as baseline.
        Actionable: If None, the most abundant community (by total counts) is auto-selected.
 
    ci_level : float in (0,1), default=0.90
        Wald confidence level (also used by deviation/reference coding).

    fdr_alpha : float, default=0.05
        BH-FDR level for the `significant` flag (q < fdr_alpha).
        Actionable: Use 0.90 for exploratory screens; 0.99 for strict confirmatory analysis.
 
    outdir : str or None, default=None
        Directory to write CSV and plots. None = do not save to disk.
        Actionable: Provide a path to persist both per-target effects and forest plots.
 
    plot : bool, default=True
        If True, produce a forest plot per target.
        Actionable: Disable in non-interactive/CI runs.
 
    scale : {"percent","ratio"}, default="percent"
        Forest-plot display scale.
        Actionable: "percent" shows (exp(coef)-1)*100; "ratio" shows exp(coef).
 
    min_total : int, default=10
        Minimum total counts (across all bags) to retain a community.
        Actionable: Increase (e.g., 20–50) in large datasets to avoid tiny/noisy communities.
 
    min_bags : int, default=1
        Minimum number of bags with >0 counts to keep a community.
        Actionable: Use 2–3 in small studies to reduce quasi/separation issues.
 
    dispersion_mode : {"moments","fixed"}, default="moments"
        How to set dispersion α.
        - "moments": estimate α per community via `estimate_nb_alpha_moments`, then fit NB with it.
        - "fixed"  : fit NB with a global α = `alpha_fixed`.
        Actionable: Prefer "moments"; use "fixed" for speed or benchmarking (α=0 → Poisson).
 
    alpha_fixed : float, default=1.0
        Global α used when `dispersion_mode="fixed"`.
        Actionable: Set 0.0 for Poisson baseline, or 0.5–1.0 for mild overdispersion.

    backend : {"nb","voomclr"}, default="nb"
        - "nb": per-community NB GLM with log(n) offset (uses dispersion_mode/alpha_fixed,
          cov_type and handle_collinearity).
        - "voomclr": CLR over the kept communities + voom precision weights + weighted
          linear model + modal bias correction + eBayes variance moderation. log(n) is not
          used as an offset, and handle_collinearity / dispersion_mode / cov_type are ignored
          (the modal bias correction already handles compositionality).
        Actionable: Use "voomclr" only when c_k form one disjoint composition (hard communities).

    voom_var_calc : {"analytical","empirical"}, default="analytical"
    voom_var_distribution : {"NB","poisson"}, default="NB" (used when voom_var_calc="analytical")
    voom_pseudocount : float, default=0.5 (pseudocount before CLR)
    voom_span : float, default=0.8 (LOWESS span for empirical weights)

    Returns
    -------
    results_by_target : dict[str, pd.DataFrame]
        Per-target effects table with columns:
        ["target","community","response","term","coef_log","se","p","ci_lo","ci_hi",
         "exp_coef","exp_ci_lo","exp_ci_hi","percent_change","dispersion_alpha",
         "converged","nobs","q","significant","coding" (if applied)]
 
    Practical guidance
    ------------------
    • If many α estimates are ≈0, Poisson is adequate (tighter CIs).
    • Use deviation coding whenever counts are compositional to avoid spurious global shifts.
    • Always inspect diagnostics: dropped communities, convergence messages, CI widths.
    """
    if covariates is None:
        covariates = []

    if community_cols is None:
        community_cols = [c for c in df.columns if c.startswith("c_")]
        
    if outdir:
        os.makedirs(outdir, exist_ok=True)
 
     # Filter rare communities
    if community_cols is not None:
        missing = [c for c in community_cols if c not in df.columns]
        if missing:
            raise ValueError(f"community_cols not found in df: {missing}")
        count_cols = list(community_cols)
    else:
        all_count_cols = [f"c_{k}" for k in range(1, K + 1)]
        count_cols = [c for c in all_count_cols if c in df.columns]
    if len(count_cols) == 0:
        raise ValueError("No community count columns found (expected c_1..c_K or via community_cols).")
 
    totals = df[count_cols].sum(axis=0)
    presence = (df[count_cols] > 0).sum(axis=0)
    keep_mask = (totals >= min_total) & (presence >= min_bags)
 
    dropped = [c for c, keep in zip(count_cols, keep_mask) if not keep]
    if dropped:
        print(f"[filter] Dropping {len(dropped)} low-membership communities: {dropped}")
    kept_cols = [c for c in count_cols if c not in dropped]
    print(f"[filter] Retained {len(kept_cols)} communities for testing after filtering.")
    if len(kept_cols) == 0:
        raise ValueError("All communities dropped by filters; relax 'min_total'/'min_bags'.")
 
    if backend not in {"nb", "voomclr"}:
        raise ValueError("backend must be 'nb' or 'voomclr'.")

    # Decide compositionality handling mode
    if handle_collinearity not in {"none", "auto", "deviation", "reference"}:
        raise ValueError("handle_collinearity must be one of {'none','auto','deviation','reference'}")

    flag, why = _detect_collinearity(df, kept_cols, offset_col)
    if backend == "voomclr":
        # Modal bias correction replaces deviation/reference coding
        comp_mode = "none"
        print(f"[info] backend=voomclr; collinearity check → {why}; post-hoc coding skipped")
        if not flag:
            warnings.warn(
                f"backend='voomclr' treats {kept_cols} as one complete composition, but their "
                f"row sums do not track '{offset_col}'. voomCLR is intended for disjoint "
                "(hard) communities.",
                RuntimeWarning
            )
    elif handle_collinearity == "auto":
        comp_mode = "deviation" if flag else "none"
        print(f"[info] Collinearity check → {why}; mode = {comp_mode}")
    else:
        comp_mode = handle_collinearity


    results_by_target: Dict[str, pd.DataFrame] = {}

    for target in target_cols:
        rhs_terms = [target] + list(covariates)
        rhs = " + ".join(rhs_terms) if rhs_terms else "1"

        rows = []
        if backend == "voomclr":
            rows = _fit_voomclr_one_target(
                df, target=target, rhs=rhs, kept_cols=kept_cols, ci_level=ci_level,
                var_calc=voom_var_calc, var_distribution=voom_var_distribution,
                pseudocount=voom_pseudocount, span=voom_span
            )
        for idx_k, response in enumerate(kept_cols if backend == "nb" else [], start=1):
            formula = f"{response} ~ {rhs}"
 
            try:
                if dispersion_mode == "moments":
                    alpha_hat, _ = estimate_nb_alpha_moments(df, formula, offset_col=offset_col)
                    res = fit_nb_glm_with_alpha(
                        df=df, formula=formula, offset_col=offset_col,
                        alpha_hat=alpha_hat, cov_type=cov_type, refit_once=True
                    )
                elif dispersion_mode == "fixed":
                    res = _fit_nb_glm_one_formula(df, formula, offset_col, cov_type, alpha_fixed)
                    alpha_hat = float(alpha_fixed)
                else:
                    raise ValueError("dispersion_mode must be 'moments' or 'fixed'.")
 
                params = res.params
                bse = res.bse
                conf = res.conf_int(alpha=1 - ci_level)
                conf.columns = ["ci_lo", "ci_hi"]
                pvals = res.pvalues
                converged = True
                nobs = res.nobs
 
            except Exception:
                rows.append({
                    "target": target, "community": response, "response": response,
                    "term": "FIT_FAILED", "coef_log": np.nan, "se": np.nan, "p": np.nan,
                    "ci_lo": np.nan, "ci_hi": np.nan, "exp_coef": np.nan,
                    "exp_ci_lo": np.nan, "exp_ci_hi": np.nan, "percent_change": np.nan,
                    "dispersion_alpha": np.nan, "converged": False, "nobs": df.shape[0],
                })
                continue
 
            for term in params.index:
                if term == "Intercept":
                    continue
                coef = float(params.get(term, np.nan))
                se = float(bse.get(term, np.nan))
                pval = float(pvals.get(term, np.nan))
                ci_lo = float(conf.loc[term, "ci_lo"]) if term in conf.index else np.nan
                ci_hi = float(conf.loc[term, "ci_hi"]) if term in conf.index else np.nan
 
                rows.append({
                    "target": target, "community": f"k={idx_k}", "response": response,
                    "term": term, "coef_log": coef, "se": se, "p": pval,
                    "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "exp_coef": np.exp(coef) if np.isfinite(coef) else np.nan,
                    "exp_ci_lo": np.exp(ci_lo) if np.isfinite(ci_lo) else np.nan,
                    "exp_ci_hi": np.exp(ci_hi) if np.isfinite(ci_hi) else np.nan,
                    "percent_change": (np.exp(coef) - 1.0) * 100.0 if np.isfinite(coef) else np.nan,
                    "dispersion_alpha": alpha_hat, "converged": converged, "nobs": nobs,
                })
 
        effects = pd.DataFrame(rows)
 
        # --- Compositional coding (post-hoc on the 'target' term across communities)
        if comp_mode == "deviation":
            effects = _apply_deviation_coding(effects, term=target, ci_level=ci_level)
 
        elif comp_mode == "reference":
            # Pick reference response if not provided: most abundant community overall
            if reference_response is None:
                total_by_resp = df[kept_cols].sum(axis=0).sort_values(ascending=False)
                reference_response = total_by_resp.index[0]
                print(f"[info] Reference community auto-selected: {reference_response}")
            effects = _apply_reference_coding(effects, term=target, ref_response=reference_response,
                                             ci_level=ci_level)
 
        # --- BH-FDR correction per term (after coding)
        effects["q"] = np.nan
        effects["significant"] = False
        for term_name, grp in effects.groupby("term", sort=False):
            if term_name == "FIT_FAILED":
                continue
            # drop rows with NaN p (e.g., reference line), but preserve their rows
            valid = np.isfinite(grp["p"].values)
            if valid.sum() == 0:
                continue
            rej, qvals, _, _ = multipletests(grp["p"].values[valid], alpha=fdr_alpha, method="fdr_bh")
            effects.loc[grp.index[valid], "q"] = qvals
            effects.loc[grp.index[valid], "significant"] = rej
 
        if outdir:
            csv_path = os.path.join(outdir, f"effects_{target}.csv")
            effects.to_csv(csv_path, index=False)
            print(f"[saved] {csv_path}")
 
        if plot:
            forest_plot_publication(
                effects=effects,
                term=target,
                ci_level=ci_level,
                title=f"CA1: {target}",
                scale=scale,
                sort_by_abs=False,
                outpath=os.path.join(outdir, f"forest_{target}.png") if outdir else None
            )
 
        results_by_target[target] = effects
 
    return results_by_target
 
 
# =============================================================================
# Forest plot
# =============================================================================
def forest_plot_publication(
    effects: pd.DataFrame,
    term: str,
    ci_level: float = 0.90,
    title: str = "",
    scale: str = "percent",    # "percent" or "ratio"
    sort_by_abs: bool = True,
    top_C: Optional[int] = 20,
    outpath: Optional[str] = None
):
    """
    Draw a compact forest plot of a single term's effects across communities.
 
    Parameters
    ----------
    effects : pd.DataFrame
        Output from `fit_ca1_multi_targets` for a given target.
 
    term : str
        The predictor to visualize (e.g., "y_resp").
 
    ci_level : float in (0,1), default=0.90
        Displayed confidence level for intervals.
        Actionable: Keep consistent with the model's CI level for clarity.
 
    title : str, default=""
        Plot title text.
 
    scale : {"percent","ratio"}, default="percent"
        X-axis scale:
        • "percent" → (exp(coef)-1)*100
        • "ratio"   → exp(coef)
        Actionable: Use "percent" for readability; "ratio" for direct multiplicative effects.
 
    sort_by_abs : bool, default=True
        If True, sort rows by |effect| descending to surface the largest effects first.
 
    top_C : int or None, default=20
        If not None, show only the top C communities by |effect| (after sorting).
        Actionable: Increase for larger panels; set None to show all.
 
    outpath : str or None, default=None
        If provided, save the figure to this path (PNG). Nothing is saved if None.
 
    Notes
    -----
    • Significant results (FDR-adjusted) are indicated in the labels if you add such markers upstream.
    • This is a compact, publication-friendly plot; customize as needed for your pipeline.
    """
    df = effects.loc[effects["term"] == term].copy()
    if df.empty:
        print(f"[warn] No rows for term '{term}'")
        return
 
    if scale == "percent":
        df["effect"] = (df["exp_coef"] - 1.0) * 100
        df["ci_lo_plot"] = (df["exp_ci_lo"] - 1.0) * 100
        df["ci_hi_plot"] = (df["exp_ci_hi"] - 1.0) * 100
        ref_line = 0.0
        x_label = "% change in expected relative abundance"
    elif scale == "ratio":
        df["effect"] = df["exp_coef"]
        df["ci_lo_plot"] = df["exp_ci_lo"]
        df["ci_hi_plot"] = df["exp_ci_hi"]
        ref_line = 1.0
        x_label = "Multiplicative effect on expected relative abundance"
    else:
        raise ValueError("scale must be 'percent' or 'ratio'")
 
    if sort_by_abs:
        df = df.reindex(df["effect"].abs().sort_values(ascending=False).index)
    else:
        df = df.sort_values("community")
 
    if top_C is not None and len(df) > top_C:
        df = df.head(top_C)
 
    df["label"] = df["community"] + df["significant"].map(lambda x: " [*]" if x else "")
    y_pos = np.arange(len(df))
 
    fig, ax = plt.subplots(figsize=(6.7, 0.44 * len(df) + 1.3))
    for i, row in enumerate(df.itertuples(index=False)):
        colour = f"C{i % 10}"
        # CI line with end caps, in the marker's colour
        ax.hlines(y=y_pos[i], xmin=row.ci_lo_plot, xmax=row.ci_hi_plot, lw=1.5, color=colour, zorder=2)
        ax.vlines([row.ci_lo_plot, row.ci_hi_plot], y_pos[i] - 0.12, y_pos[i] + 0.12, lw=1.2, color=colour, zorder=2)
        ax.plot(row.effect, y_pos[i], "s", markersize=6, color=colour, zorder=3)
 
    ax.axvline(ref_line, linestyle="--", color="gray", lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_title(f"{title} ({int(ci_level*100)}% CI)", fontsize=12)
 
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
 
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"[saved] {outpath}")
 
    plt.show()
    plt.close(fig)
 
 
# =============================================================================
# Diagnostics: compare MoM vs fixed-alpha on the same term
# =============================================================================
def compare_moments_vs_fixed(eff_mom: pd.DataFrame, eff_fix: pd.DataFrame, term: str) -> pd.DataFrame:
    """
    Compare per-community effects between MoM-α fits and fixed-α fits for one term.
 
    Parameters
    ----------
    eff_mom : pd.DataFrame
        Effects DataFrame for a target produced by `fit_ca1_multi_targets` with
        `dispersion_mode="moments"`.
 
    eff_fix : pd.DataFrame
        Effects DataFrame for the same target produced with `dispersion_mode="fixed"`.
 
    term : str
        The predictor to compare (e.g., "y_resp").
 
    Returns
    -------
    pd.DataFrame
        A merged table with columns:
        ["community","response","alpha_hat","coef_mom","se_mom","p_mom","q_mom",
         "coef_fix","se_fix","p_fix","q_fix","se_ratio_mom_over_fix",
         "agree_significant","disagree"]
 
    Actionable guidance
    -------------------
    • Inspect `alpha_hat`—if many 0’s, Poisson suffices.
    • `se_ratio_mom_over_fix` > 1 implies MoM-α increased uncertainty (more conservative).
    • Use `disagree` to flag communities whose significance depends on α handling.
    """
    a = eff_mom.loc[eff_mom["term"] == term, ["community", "response", "coef_log", "se", "p", "q", "dispersion_alpha"]].copy()
    a = a.rename(columns={"coef_log": "coef_mom", "se": "se_mom", "p": "p_mom", "q": "q_mom", "dispersion_alpha": "alpha_hat"})
    b = eff_fix.loc[eff_fix["term"] == term, ["community", "response", "coef_log", "se", "p", "q"]].copy()
    b = b.rename(columns={"coef_log": "coef_fix", "se": "se_fix", "p": "p_fix", "q": "q_fix"})
    out = a.merge(b, on=["community", "response"], how="outer", validate="one_to_one")
    out["se_ratio_mom_over_fix"] = out["se_mom"] / out["se_fix"]
    out["agree_significant"] = (out["q_mom"] < 0.05) & (out["q_fix"] < 0.05)
    out["disagree"] = (out["q_mom"] < 0.05) ^ (out["q_fix"] < 0.05)
    return out
 