# =================================================================================================
# Pure-Python port of the edgeR NB dispersion used by voomCLR (no R needed)
# =================================================================================================
# voomCLR:::.estimateNBDispersion does
#     d <- calcNormFactors(DGEList(counts));  d <- estimateDisp(d, design);  d$tagwise.dispersion
# This module reproduces that call for the arguments voomCLR uses (TMM, trend.method="locfit",
# robust=FALSE, no observation weights). Ported line by line from the R and C sources of
#     edgeR 4.10.6  (estimateDisp, glmFit, mglmLevenberg, mglmOneWay, adjustedProfileLik,
#                    maximizeInterpolant, aveLogCPM, WLEB, calcNormFactors/TMM)
#     limma 3.68.5  (squeezeVar, fitFDist, fitFDistUnequalDF1, weightedLowess, trigammaInverse)
#     locfit 1.5-9.12 (locfit(deg=0) with the default adaptive-tree evaluation, as in locfitByCol)
#     R stats       (optimize = Brent_fmin, fmm spline, lowess = clowess, approxfun)
# Orientation follows edgeR: genes (communities) in rows, libraries (bags) in columns.
# Entry point: edger_tagwise_dispersion(counts_bags_x_communities, design).
# =================================================================================================
from typing import Tuple
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import lapack, null_space
from scipy.special import digamma, gammaln, polygamma
from scipy.stats import rankdata


# -------------------------------------------------------------------------------------------------
# calcNormFactors(method="TMM")
# -------------------------------------------------------------------------------------------------
def _calc_factor_tmm(obs, ref, n_obs, n_ref, logratio_trim=0.3, sum_trim=0.05, a_cutoff=-1e10):
    with np.errstate(divide="ignore", invalid="ignore"):
        log_r = np.log2((obs / n_obs) / (ref / n_ref))
        abs_e = (np.log2(obs / n_obs) + np.log2(ref / n_ref)) / 2
        v = (n_obs - obs) / n_obs / obs + (n_ref - ref) / n_ref / ref
    fin = np.isfinite(log_r) & np.isfinite(abs_e) & (abs_e > a_cutoff)
    log_r, abs_e, v = log_r[fin], abs_e[fin], v[fin]
    if log_r.size == 0 or np.max(np.abs(log_r)) < 1e-6:
        return 1.0
    n = log_r.size
    lo_l = np.floor(n * logratio_trim) + 1
    hi_l = n + 1 - lo_l
    lo_s = np.floor(n * sum_trim) + 1
    hi_s = n + 1 - lo_s
    rl, re = rankdata(log_r), rankdata(abs_e)
    keep = (rl >= lo_l) & (rl <= hi_l) & (re >= lo_s) & (re <= hi_s)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.nansum(log_r[keep] / v[keep]) / np.nansum(1 / v[keep])
    if np.isnan(f):
        f = 0.0
    return 2.0 ** f


def calc_norm_factors_tmm(y: np.ndarray) -> np.ndarray:
    """edgeR::calcNormFactors(method="TMM") for a genes x libraries count matrix."""
    x = np.asarray(y, dtype=float)
    nsamples = x.shape[1]
    lib_size = x.sum(axis=0)
    x = x[(x > 0).sum(axis=1) > 0]
    if x.shape[0] == 0 or nsamples == 1:
        return np.ones(nsamples)
    f75 = np.quantile(x, 0.75, axis=0) / lib_size
    if np.median(f75) < 1e-20:
        ref = int(np.argmax(np.sqrt(x).sum(axis=0)))
    else:
        ref = int(np.argmin(np.abs(f75 - f75.mean())))
    f = np.array([_calc_factor_tmm(x[:, i], x[:, ref], lib_size[i], lib_size[ref])
                  for i in range(nsamples)])
    return f / np.exp(np.mean(np.log(f)))


# -------------------------------------------------------------------------------------------------
# NB GLM fitting (glmFit -> mglmOneWay / mglmLevenberg)
# -------------------------------------------------------------------------------------------------
def _nb_unit_deviance(y, mu, phi):
    """compute_unit_nb_deviance() from compute_nbdev.c (vectorised)."""
    y = y + 1e-8
    mu = mu + 1e-8
    phi = np.broadcast_to(phi, y.shape)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        resid = y - mu
        # C evaluates 2/3 in integer arithmetic, i.e. 0; kept for exactness.
        pois = 2 * (y * np.log(y / mu) - resid - 0.5 * resid * resid * phi * (1 + phi * (0 * resid - y)))
        product = mu * phi
        gam = 2 * ((y - mu) / mu - np.log(y / mu)) * mu / (1 + product)
        inv = 1 / phi
        nb = 2 * (y * np.log(y / mu) + (y + inv) * np.log((mu + inv) / (y + inv)))
    out = np.where(phi < 1e-4, pois, np.where(product > 1e6, gam, nb))
    return np.maximum(out, 0.0)


def _fit_one_group(y, offset, disp, maxit=50, tol=1e-10, start=np.nan):
    """glm_one_group_vec() from glm.c, unit weights."""
    pos = y > 1e-10
    if not pos.any():
        return -np.inf
    if np.isnan(start):
        b = np.log(np.sum(y[pos] / np.exp(offset[pos])) / y.size)
    else:
        b = float(start)
    for _ in range(maxit):
        mu = np.exp(b + offset)
        den = 1 + mu * disp
        step = np.sum((y - mu) / den) / np.sum(mu / den)
        b += step
        if abs(step) < tol:
            break
    # (on non-convergence edgeR leaves the coefficient unset; the last iterate is returned here)
    return b


def _levenberg_start(y, offset, disp, X):
    """get_leven_start(use_null=TRUE) from glm.c."""
    n_lib = np.exp(offset)
    cw = n_lib / (1 + disp * n_lib)
    with np.errstate(divide="ignore"):
        level = np.log(np.sum(y * cw / n_lib) / np.sum(cw))
    return np.linalg.lstsq(X, np.full(X.shape[0], level), rcond=None)[0]


def _fit_levenberg_one(y, offset, disp, X, beta, tol=1e-6, maxit=250):
    """fit_leven_vec() from glm.c, unit weights. Returns (beta, mu, deviance)."""
    p = X.shape[1]
    ymax = y.max()
    if ymax < 1e-10:
        return np.full(p, np.nan), np.zeros_like(y), 0.0
    beta = np.array(beta, dtype=float)
    mu = np.exp(offset + X @ beta)
    dev = float(np.sum(_nb_unit_deviance(y, mu, disp)))
    max_info, lam = -1.0, 0.0
    it = 0
    while it < maxit:
        it += 1
        denom = 1 + mu * disp
        xtwx = X.T @ ((mu / denom)[:, None] * X)
        dl = X.T @ ((y - mu) / denom)
        max_info = max(max_info, float(np.max(np.diag(xtwx))))
        if it == 1:
            lam = max(max_info * 1e-6, 1e-13)
        lev, low_dev, failed = 0, False, False
        while True:
            lev += 1
            while True:
                chol, info = lapack.dpotrf(xtwx + lam * np.eye(p), lower=0)
                if info == 0:
                    break
                lam *= 10
                if lam <= 0:
                    lam = 1e-100
            db, _ = lapack.dpotrs(chol, dl, lower=0)
            nbeta = beta + db
            nmu = np.exp(offset + X @ nbeta)
            ndev = float(np.sum(_nb_unit_deviance(y, nmu, disp)))
            if ndev / ymax < 1e-13:
                low_dev = True
            if ndev <= dev or low_dev:
                beta, mu, dev = nbeta, nmu, ndev
                break
            lam *= 2
            if lam <= 0:
                lam = 1e-100
            if lam / max_info > 1e13:
                failed = True
                break
        if failed or low_dev or float(dl @ db) < tol:
            break
        if lev == 1:
            lam /= 10
    return beta, mu, dev


def _design_as_factor(X):
    """edgeR::designAsFactor: one level per distinct design row (factor() matches on 15 digits)."""
    z = (np.e + np.pi) / 5
    key = np.mean(X * z ** np.arange(X.shape[1]), axis=1)
    key = np.array([float(f"{k:.15g}") for k in key])
    levels, group = np.unique(key, return_inverse=True)
    return group, len(levels)


def _glm_fit(y, X, offset, disp, start=None):
    """glmFit(prior.count=0) for unit weights. Returns (coefficients, fitted, deviance)."""
    G, n = y.shape
    p = X.shape[1]
    disp = np.broadcast_to(np.asarray(disp, dtype=float).reshape(-1, 1), (G, 1))[:, 0]
    group, ngroups = _design_as_factor(X)
    mu = np.empty_like(y)
    if ngroups == p:
        # mglmOneWay
        first = np.array([np.flatnonzero(group == g)[0] for g in range(ngroups)])
        dunique = X[first]
        gstart = None if start is None else start @ dunique.T
        gbeta = np.empty((G, ngroups))
        for g in range(ngroups):
            j = group == g
            for i in range(G):
                s = np.nan if gstart is None else gstart[i, g]
                gbeta[i, g] = _fit_one_group(y[i, j], offset[j], disp[i], start=s)
        gbeta = np.maximum(gbeta, -1e8)
        mu = np.exp(offset[None, :] + gbeta[:, group])
        coef = np.linalg.solve(dunique, gbeta.T).T
        dev = np.sum(_nb_unit_deviance(y, mu, disp[:, None]), axis=1)
        return coef, mu, dev
    coef = np.empty((G, p))
    dev = np.empty(G)
    for i in range(G):
        b0 = _levenberg_start(y[i], offset, disp[i], X) if start is None else start[i]
        coef[i], mu[i], dev[i] = _fit_levenberg_one(y[i], offset, disp[i], X, b0)
    return coef, mu, dev


def _qr_keep_cols(X, tol=1e-7):
    """Columns kept by R's qr() (LINPACK dqrdc2: near-dependent columns pivoted to the end)."""
    keep, basis = [], np.zeros((X.shape[0], 0))
    for j in range(X.shape[1]):
        v = X[:, j]
        norm0 = np.linalg.norm(v)
        r = v - basis @ (basis.T @ v)
        if norm0 > 0 and np.linalg.norm(r) >= tol * norm0:
            keep.append(j)
            basis = np.column_stack([basis, r / np.linalg.norm(r)])
    return keep


def _adjusted_profile_lik(disp, y, X, offset, start=None):
    """adjustedProfileLik(adjust=TRUE, get.coef=TRUE) via compute_adj_profile_ll() in compute_apl.c."""
    coef, mu, _ = _glm_fit(y, X, offset, disp, start)
    G = y.shape[0]
    p = X.shape[1]
    apl = np.zeros(G)
    r = 1 / disp
    for i in range(G):
        u, yy = mu[i], y[i]
        nz = u != 0
        un, yn = u[nz], yy[nz]
        logmur = np.log(un + r)
        ll = yn * np.log(un) - yn * logmur + r * np.log(r) - r * logmur \
            + gammaln(yn + r) - gammaln(yn + 1) - gammaln(r)
        w = np.where(nz, u / (1 + disp * u), 0.0)
        if p == 1:
            adj = np.log(abs(np.sum(w))) / 2
        else:
            ldl, _, info = lapack.dsytrf(X.T @ (w[:, None] * X), lower=0)
            dg = np.diag(ldl)
            with np.errstate(divide="ignore", invalid="ignore"):
                adj = np.sum(np.where(dg < 1e-10, np.log(1e-10), 0.5 * np.log(dg)))
        apl[i] = np.sum(ll) - adj
    return apl, coef


def _resid_df(zero, X):
    """edgeR:::.residDF"""
    nlib, ncoef = zero.shape[1], X.shape[1]
    nzero = zero.sum(axis=1)
    df = np.full(zero.shape[0], nlib - ncoef, dtype=float)
    df[nzero == nlib] = 0
    for i in np.flatnonzero((nzero > 0) & (nzero < nlib)):
        df[i] = max(nlib - nzero[i] - len(_qr_keep_cols(X[~zero[i]])), 0)
    return df


# -------------------------------------------------------------------------------------------------
# maximizeInterpolant (fmm spline from R's splines.c + find_max from interpolator.c)
# -------------------------------------------------------------------------------------------------
def _fmm_spline(x, y):
    n = len(x)
    b, c, d = np.zeros(n), np.zeros(n), np.zeros(n)
    if n < 3:
        b[0] = b[1] = (y[1] - y[0]) / (x[1] - x[0])
        return b, c, d
    # 0-based translation of the 1-based C code
    d[0] = x[1] - x[0]
    c[1] = (y[1] - y[0]) / d[0]
    for i in range(1, n - 1):
        d[i] = x[i + 1] - x[i]
        b[i] = 2.0 * (d[i - 1] + d[i])
        c[i + 1] = (y[i + 1] - y[i]) / d[i]
        c[i] = c[i + 1] - c[i]
    b[0] = -d[0]
    b[n - 1] = -d[n - 2]
    c[0] = c[n - 1] = 0.0
    if n > 3:
        c[0] = c[2] / (x[3] - x[1]) - c[1] / (x[2] - x[0])
        c[n - 1] = c[n - 2] / (x[n - 1] - x[n - 3]) - c[n - 3] / (x[n - 2] - x[n - 4])
        c[0] = c[0] * d[0] * d[0] / (x[3] - x[0])
        c[n - 1] = -c[n - 1] * d[n - 2] * d[n - 2] / (x[n - 1] - x[n - 4])
    for i in range(1, n):
        t = d[i - 1] / b[i - 1]
        b[i] = b[i] - t * d[i - 1]
        c[i] = c[i] - t * c[i - 1]
    c[n - 1] = c[n - 1] / b[n - 1]
    for i in range(n - 2, -1, -1):
        c[i] = (c[i] - d[i] * c[i + 1]) / b[i]
    b[n - 1] = (y[n - 1] - y[n - 2]) / d[n - 2] + d[n - 2] * (c[n - 2] + 2.0 * c[n - 1])
    for i in range(n - 1):
        b[i] = (y[i + 1] - y[i]) / d[i] - d[i] * (c[i + 1] + 2.0 * c[i])
        d[i] = (c[i + 1] - c[i]) / d[i]
        c[i] = 3.0 * c[i]
    c[n - 1] = 3.0 * c[n - 1]
    d[n - 1] = d[n - 2]
    return b, c, d


def _find_max(x, y):
    m = int(np.argmax(y))
    maxed, x_max = y[m], x[m]
    b, c, d = _fmm_spline(x, y)
    for seg in ((m - 1,) if m > 0 else ()) + ((m,) if m < len(x) - 1 else ()):
        delta = c[seg] ** 2 - 3 * d[seg] * b[seg]
        if delta < 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            sol = (-c[seg] - np.sqrt(delta)) / (3 * d[seg])
        if 0 < sol < x[seg + 1] - x[seg]:
            val = ((d[seg] * sol + c[seg]) * sol + b[seg]) * sol + y[seg]
            if val > maxed:
                maxed, x_max = val, sol + x[seg]
    return x_max


def maximize_interpolant(x: np.ndarray, ll: np.ndarray) -> np.ndarray:
    """edgeR::maximizeInterpolant: spline maximum of each row of ll over the grid x."""
    x = np.asarray(x, dtype=float)
    return np.array([_find_max(x, row) for row in np.atleast_2d(np.asarray(ll, dtype=float))])


# -------------------------------------------------------------------------------------------------
# aveLogCPM (ave_log_cpm in compute_cpm.c, prior.count=2)
# -------------------------------------------------------------------------------------------------
def ave_log_cpm(y, lib_size, dispersion, prior_count=2.0):
    y = np.asarray(y, dtype=float)
    lib = np.asarray(lib_size, dtype=float)
    prior = prior_count * lib / lib.mean()
    offset = np.log(lib + 2 * prior)
    disp = np.broadcast_to(np.asarray(dispersion, dtype=float), (y.shape[0],))
    out = np.empty(y.shape[0])
    for i in range(y.shape[0]):
        b = _fit_one_group(y[i] + prior, offset, disp[i], maxit=50, tol=1e-10)
        out[i] = (b + np.log(1e6)) / np.log(2)
    return out


# -------------------------------------------------------------------------------------------------
# locfitByCol(degree=0): locfit(y ~ x, alpha=span, deg=0), ev=rbox() adaptive tree, tricube kernel
# -------------------------------------------------------------------------------------------------
def _locfit_deg0_fitted(y: np.ndarray, x: np.ndarray, span: float, cut: float = 0.8) -> np.ndarray:
    """fitted(locfit(y ~ x, alpha=span, deg=0)) for every column of y (all columns share x)."""
    n = len(x)
    nn = int(n * span + 1e-12)
    lo, hi = float(x.min()), float(x.max())

    def fit_at(v):
        di = np.abs(x - v)
        if nn <= 0:
            h = 0.0
        elif nn < n:
            h = float(np.sort(di)[nn - 1])          # kordstat: nn-th smallest distance
        else:
            h = float(di.max() * np.exp(np.log(nn / n)))
        if h == 0:
            w = (di == 0).astype(float)
        else:
            u = di / h
            w = np.where(u > 1, 0.0, (1 - u ** 3) ** 3)
        with np.errstate(invalid="ignore"):  # all-zero weights: locfit LF_NOPT, vertex unused at data points
            return (w @ y) / w.sum(), h

    # Adaptive tree (ev_atree.c, d=1): vertices at the data range ends, cells split at the midpoint
    # while length/min(vertex bandwidth) > cut. No derivatives at deg=0, so linear interpolation.
    xs, vals, hs = [], [], []

    def add_vertex(v):
        f, h = fit_at(v)
        xs.append(v), vals.append(f), hs.append(h)
        if len(xs) > int((5 * (1 / span if span <= 1 else 1) / min(cut, 1.0) + 1) * 2):
            raise RuntimeError("locfit: newsplit: out of vertex space")
        return len(xs) - 1

    def split(i0, i1):
        le = xs[i1] - xs[i0]
        hpos = [h for h in (hs[i0], hs[i1]) if h > 0]
        score = 2 * le / (hi - lo) if not hpos else le / min(hpos)
        return cut < score

    def grow(i0, i1):
        if not split(i0, i1):
            return (i0, i1, None)
        m = add_vertex((xs[i0] + xs[i1]) / 2)
        return (i0, i1, (grow(i0, m), grow(m, i1)))

    root = grow(add_vertex(lo), add_vertex(hi))
    fitted = np.empty_like(y, dtype=float)
    for k in range(n):
        node = root
        while node[2] is not None:
            i0, i1, kids = node
            node = kids[0] if 2 * (x[k] - xs[i0]) < (xs[i1] - xs[i0]) else kids[1]
        i0, i1, _ = node
        dlen = xs[i1] - xs[i0]
        fitted[k] = vals[i0] if dlen == 0 else ((dlen - (x[k] - xs[i0])) * vals[i0] + (x[k] - xs[i0]) * vals[i1]) / dlen
    return fitted


def locfit_by_col(y: np.ndarray, x: np.ndarray, span: float) -> np.ndarray:
    """edgeR::locfitByCol(y, x, weights=1, span, degree=0)."""
    y = np.asarray(y, dtype=float)
    ntags = y.shape[0]
    if span * ntags < 2 or ntags <= 1:
        return y.copy()
    return _locfit_deg0_fitted(y, np.asarray(x, dtype=float), span)


def choose_lowess_span(n, small_n=50, min_span=0.3, power=1 / 3):
    """limma::chooseLowessSpan"""
    return min(min_span + (1 - min_span) * (small_n / n) ** power, 1.0)


# -------------------------------------------------------------------------------------------------
# limma: squeezeVar -> fitFDist / fitFDistUnequalDF1 (df.prior only)
# -------------------------------------------------------------------------------------------------
def _logmdigamma(x):
    return np.log(x) - digamma(x)


def trigamma_inverse(x: float) -> float:
    """limma::trigammaInverse (Newton iteration, scalar)."""
    if x > 1e7:
        return 1 / np.sqrt(x)
    if x < 1e-6:
        return 1 / x
    y = 0.5 + 1 / x
    for _ in range(51):
        tri = polygamma(1, y)
        dif = tri * (1 - tri / x) / polygamma(2, y)
        y += dif
        if -dif / y < 1e-8:
            break
    return float(y)


def _brent_fmin(f, ax, bx, tol=np.finfo(float).eps ** 0.25):
    """R's optimize() (Brent_fmin in src/library/stats/src/optimize.c)."""
    c = (3.0 - np.sqrt(5.0)) * 0.5
    eps = np.sqrt(np.finfo(float).eps)
    a, b = ax, bx
    v = w = x = a + c * (b - a)
    d = e = 0.0
    fv = fw = fx = f(x)
    tol3 = tol / 3.0
    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0
        if abs(x - xm) <= t2 - (b - a) * 0.5:
            break
        p = q = r = 0.0
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2.0
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d
        if abs(p) >= abs(q * 0.5 * r) or p <= q * (a - x) or p >= q * (b - x):
            e = (b - x) if x < xm else (a - x)
            d = c * e
        else:
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = tol1 if x < xm else -tol1
        if abs(d) >= tol1:
            u = x + d
        elif d > 0.0:
            u = x + tol1
        else:
            u = x - tol1
        fu = f(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv, w, fw = w, fw, u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x


def _ns_basis(x, df):
    """A basis spanning splines::ns(x, df=df, intercept=TRUE) (only its column space is needed)."""
    n_iknots = df - 2
    probs = np.linspace(0, 1, n_iknots + 2)[1:-1]
    knots = np.quantile(x, probs) if n_iknots > 0 else np.array([])
    bk = np.array([x.min(), x.max()])
    t = np.sort(np.r_[np.repeat(bk[0], 4), knots, np.repeat(bk[1], 4)])
    nb = len(t) - 4
    basis = np.column_stack([BSpline(t, np.eye(nb)[j], 3, extrapolate=True)(x) for j in range(nb)])
    const = np.vstack([BSpline(t, np.eye(nb)[j], 3).derivative(2)(bk) for j in range(nb)]).T
    return basis @ null_space(const)


def _fit_f_dist(x, df1, covariate):
    """limma::fitFDist -> df2 (prior df)."""
    n = len(x)
    if n == 0:
        return np.nan
    if n == 1:
        return 0.0
    ok = np.isfinite(df1) & (df1 > 1e-15) & np.isfinite(x) & (x > -1e-15)
    nok = int(ok.sum())
    if nok == 1:
        return 0.0
    x, df1 = x[ok], df1[ok]
    splinedf = 1
    if covariate is not None:
        cov = covariate[ok]
        splinedf = 1 + (nok >= 3) + (nok >= 6) + (nok >= 30)
        splinedf = min(splinedf, len(np.unique(cov)))
        if splinedf < 2:
            return _fit_f_dist(x, df1, None)
    x = np.maximum(x, 0)
    m = np.median(x)
    if m == 0:
        m = 1.0
    x = np.maximum(x, 1e-5 * m)
    e = np.log(x) + _logmdigamma(df1 / 2)
    if covariate is None:
        evar = np.sum((e - e.mean()) ** 2) / (nok - 1)
    else:
        design = _ns_basis(cov, splinedf)
        coef, _, rank, _ = np.linalg.lstsq(design, e, rcond=None)
        rss = np.sum((e - design @ coef) ** 2)
        evar = rss / (nok - rank)
    evar = evar - np.mean(polygamma(1, df1 / 2))
    return 2 * trigamma_inverse(evar) if evar > 0 else np.inf


def _weighted_lowess_fitted(x, y, w, span):
    """limma::weightedLowess(iterations=1, npts=200) fitted values (weighted_lowess.c)."""
    o = np.argsort(x, kind="stable")
    xs, ys, ws = x[o], y[o], w[o]
    npts = len(xs)
    if 200 >= npts:
        delta = 0.0
    else:
        dx = np.sort(np.diff(xs))
        cum = np.cumsum(dx)
        k = np.arange(200)
        delta = float(np.min(cum[len(dx) - 1 - k] / (200 - k)))
    seeds = [0]
    for pt in range(1, npts - 1):
        if xs[pt] - xs[seeds[-1]] > delta:
            seeds.append(pt)
    seeds.append(npts - 1)
    spanweight = ws.sum() * span
    subrange = (xs[-1] - xs[0]) / npts
    fit = np.empty(npts)

    def local(cur):
        left = right = cur
        curw = ws[cur]
        ende, ends = cur == npts - 1, cur == 0
        mdist = 0.0
        while curw < spanweight and (not ende or not ends):
            if ende:
                left -= 1
                curw += ws[left]
                ends = ends or left == 0
                mdist = max(mdist, xs[cur] - xs[left])
            elif ends:
                right += 1
                curw += ws[right]
                ende = ende or right == npts - 1
                mdist = max(mdist, xs[right] - xs[cur])
            else:
                ldist, rdist = xs[cur] - xs[left - 1], xs[right + 1] - xs[cur]
                if ldist < rdist:
                    left -= 1
                    curw += ws[left]
                    ends = ends or left == 0
                    mdist = max(mdist, ldist)
                else:
                    right += 1
                    curw += ws[right]
                    ende = ende or right == npts - 1
                    mdist = max(mdist, rdist)
        while left > 0 and xs[left] == xs[left - 1]:
            left -= 1
        while right < npts - 1 and xs[right] == xs[right + 1]:
            right += 1
        sl = slice(left, right + 1)
        if mdist < 1e-7:
            return np.sum(ys[sl] * ws[sl]) / np.sum(ws[sl])
        wk = (1 - (np.abs(xs[cur] - xs[sl]) / mdist) ** 3) ** 3 * ws[sl]
        xm, ym = np.sum(wk * xs[sl]) / wk.sum(), np.sum(wk * ys[sl]) / wk.sum()
        var = np.sum((xs[sl] - xm) ** 2 * wk)
        if var < 1e-7:
            return ym
        slope = np.sum((xs[sl] - xm) * (ys[sl] - ym) * wk) / var
        return slope * xs[cur] + ym - slope * xm

    last = 0
    fit[0] = local(0)
    for pt in seeds[1:]:
        fit[pt] = local(pt)
        if pt - last > 1:
            cur = xs[pt] - xs[last]
            if cur > 1e-7 * subrange:
                slope = (fit[pt] - fit[last]) / cur
                fit[last + 1:pt] = slope * xs[last + 1:pt] + fit[pt] - slope * xs[pt]
            else:
                fit[last + 1:pt] = 0.5 * (fit[pt] + fit[last])
        last = pt
    out = np.empty(npts)
    out[o] = fit
    return out


def _loess_fit(y, x, weights, span, min_weight=1e-8, max_weight=100.0):
    """limma::loessFit(iterations=1) fitted values for finite inputs with weights."""
    nobs = len(y)
    if span < 1 / nobs:
        return y.copy()
    w = np.minimum(np.maximum(weights, min_weight), max_weight)
    if w.max() - w.min() < 1e-15:
        # loessFit would use stats::lowess here; unreachable from fitFDistUnequalDF1 (df1 differ)
        w = np.ones_like(w)
    if nobs < 4 + 1 / span:
        if nobs == 1:
            return y.copy()
        A = np.column_stack([np.ones(nobs), x]) * np.sqrt(w)[:, None]
        coef = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)[0]
        return coef[0] + coef[1] * x
    return _weighted_lowess_fitted(x, y, w, span)


def _fit_f_dist_unequal_df1(x, df1, covariate):
    """limma::fitFDistUnequalDF1(robust=FALSE) -> df2 (prior df)."""
    x = np.array(x, dtype=float)
    df1 = np.array(df1, dtype=float)
    n = len(x)
    prior_w = None
    if np.any(np.isnan(x)):
        i = np.isnan(x)
        prior_w = (~i).astype(float)
        x[i] = 0
    if df1.min() < 0.01:
        i = df1 < 0.01
        prior_w = (~i).astype(float) if prior_w is None else np.where(i, 0.0, prior_w)
        df1[i] = 1
    inf = x > 0
    if prior_w is not None:
        inf[prior_w == 0] = False
    n_inf = int(inf.sum())
    if n_inf < 2:
        return np.nan
    if n_inf == 2:
        covariate, prior_w = None, None
    m = np.median(x[inf])
    xpos = np.maximum(x, 1e-12 * m)
    d1 = df1 / 2
    e = np.log(xpos) + _logmdigamma(d1)
    w = 1 / polygamma(1, d1)
    if prior_w is not None:
        w = w * prior_w
    if covariate is None:
        emean = np.sum(w * e) / np.sum(w)
    else:
        span = choose_lowess_span(n, small_n=500)
        emean = _loess_fit(e, covariate, w / np.quantile(w, 0.75), span)
    d1x = d1 * xpos
    pw = 1.0 if prior_w is None else prior_w

    def m2ll(par):
        d2 = par / (1 - par)
        d2s20 = d2 * np.exp(emean - _logmdigamma(d2))
        return -2 * np.sum(pw * (-(d1 + d2) * np.log1p(d1x / d2s20) - d1 * np.log(d2s20)
                                 + gammaln(d1 + d2) - gammaln(d2)))

    par = _brent_fmin(m2ll, 0.5, 0.9998)
    return 2 * par / (1 - par)


def squeeze_var_df_prior(var, df, covariate=None) -> float:
    """limma::squeezeVar(robust=FALSE)$df.prior"""
    var = np.array(var, dtype=float)
    df = np.asarray(df, dtype=float)
    if len(var) < 3:
        return 0.0
    var[df == 0] = 0
    dfp = df[df > 0]
    legacy = dfp.size > 0 and dfp.min() == dfp.max()
    if legacy:
        df_prior = _fit_f_dist(var, df, covariate)
    else:
        df_prior = _fit_f_dist_unequal_df1(var, df, covariate)
    if np.isnan(df_prior):
        raise RuntimeError("Could not estimate prior df")
    return float(df_prior)


# -------------------------------------------------------------------------------------------------
# R stats::lowess (clowess/lowest from R's lowess.c) and approxfun(rule=2, ties=mean)
# -------------------------------------------------------------------------------------------------
def _lowest(x, y, n, xs, nleft, nright, w, it, rw):
    """lowest() from lowess.c; arrays are 1-based (index 0 unused). Returns (ys, ok)."""
    rng = x[n] - x[1]
    h = max(xs - x[nleft], x[nright] - xs)
    h9, h1 = 0.999 * h, 0.001 * h
    a = 0.0
    j = nleft
    while j <= n:
        w[j] = 0.0
        r = abs(x[j] - xs)
        if r <= h9:
            w[j] = 1.0 if r <= h1 else (1.0 - (r / h) ** 3) ** 3
            if it > 1:
                w[j] *= rw[j]
            a += w[j]
        elif x[j] > xs:
            break
        j += 1
    nrt = j - 1
    if a <= 0.0:
        return 0.0, False
    for j in range(nleft, nrt + 1):
        w[j] /= a
    if h > 0.0:
        a = sum(w[j] * x[j] for j in range(nleft, nrt + 1))
        b = xs - a
        c = sum(w[j] * (x[j] - a) ** 2 for j in range(nleft, nrt + 1))
        if np.sqrt(c) > 0.001 * rng:
            b /= c
            for j in range(nleft, nrt + 1):
                w[j] *= b * (x[j] - a) + 1.0
    return sum(w[j] * y[j] for j in range(nleft, nrt + 1)), True


def r_lowess(x, y, f=2 / 3, iters=3, delta=None) -> Tuple[np.ndarray, np.ndarray]:
    """stats::lowess(x, y, f, iter, delta = 0.01 * diff(range(x))). Returns (sorted x, fitted y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    o = np.argsort(x, kind="stable")
    xo, yo = x[o], y[o]
    n = len(xo)
    if delta is None:
        delta = 0.01 * (xo[-1] - xo[0])
    if n < 2:
        return xo, yo.copy()
    X = np.r_[0.0, xo]
    Y = np.r_[0.0, yo]
    ys = np.zeros(n + 1)
    rw = np.zeros(n + 1)
    w = np.zeros(n + 1)
    ns = max(2, min(n, int(f * n + 1e-7)))
    it = 1
    while it <= iters + 1:
        nleft, nright, last, i = 1, ns, 0, 1
        while True:
            if nright < n:
                d1 = X[i] - X[nleft]
                d2 = X[nright + 1] - X[i]
                if d1 > d2:
                    nleft += 1
                    nright += 1
                    continue
            val, ok = _lowest(X, Y, n, X[i], nleft, nright, w, it, rw)
            ys[i] = val if ok else Y[i]
            if last < i - 1:
                denom = X[i] - X[last]
                for j in range(last + 1, i):
                    alpha = (X[j] - X[last]) / denom
                    ys[j] = alpha * ys[i] + (1.0 - alpha) * ys[last]
            last = i
            cut = X[last] + delta
            i = last + 1
            while i <= n:
                if X[i] > cut:
                    break
                if X[i] == X[last]:
                    ys[i] = ys[last]
                    last = i
                i += 1
            i = max(last + 1, i - 1)
            if last >= n:
                break
        res = Y[1:] - ys[1:]
        sc = np.mean(np.abs(res))
        if it > iters:
            break
        ar = np.sort(np.abs(res))
        m1 = n // 2
        cmad = 3.0 * (ar[m1] + ar[n - m1 - 1]) if n % 2 == 0 else 6.0 * ar[m1]
        if cmad < 1e-7 * sc:
            break
        r = np.abs(res)
        with np.errstate(divide="ignore", invalid="ignore"):  # cmad == 0: every |res| <= c1 -> weight 1
            rw[1:] = np.where(r <= 0.001 * cmad, 1.0,
                              np.where(r <= 0.999 * cmad, (1.0 - (r / cmad) ** 2) ** 2, 0.0))
        it += 1
    return xo, ys[1:]


def r_approxfun_rule2(x, y):
    """stats::approxfun(x, y, rule=2, ties=list("ordered", mean)) for sorted x."""
    ux, inv = np.unique(np.asarray(x, dtype=float), return_inverse=True)
    uy = np.bincount(inv, weights=np.asarray(y, dtype=float)) / np.bincount(inv)
    if len(ux) == 1:
        return lambda v: np.full(np.shape(v), uy[0])
    return lambda v: np.interp(v, ux, uy)


# -------------------------------------------------------------------------------------------------
# estimateDisp (design given, trend.method="locfit", tagwise=TRUE, robust=FALSE)
# -------------------------------------------------------------------------------------------------
def estimate_disp(y, design, lib_size, min_row_sum=5, grid_length=21, grid_range=(-10, 10)) -> dict:
    """edgeR::estimateDisp.default for a genes x libraries matrix with offset=log(lib_size)."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(design, dtype=float)
    ntags, nlibs = y.shape
    offset = np.log(np.asarray(lib_size, dtype=float))
    if X.shape[1] >= nlibs:
        nan = np.full(ntags, np.nan)
        return {"common.dispersion": np.nan, "trended.dispersion": nan, "tagwise.dispersion": nan}
    sel = y.sum(axis=1) >= min_row_sum
    if not sel.any():
        raise ValueError(f"No community has a total count >= {min_row_sum}.")
    sy = y[sel]
    spline_pts = np.linspace(grid_range[0], grid_range[1], grid_length)
    spline_disp = 0.1 * 2.0 ** spline_pts
    l0 = np.zeros((sy.shape[0], grid_length))

    _, mu0, _ = _glm_fit(sy, X, offset, 0.05)
    zerofit = (sy < 1e-4) & (mu0 < 1e-4)
    patterns, pat_idx = np.unique(zerofit, axis=0, return_inverse=True)
    for u in range(patterns.shape[0]):
        subg = np.flatnonzero(pat_idx.ravel() == u)
        nz = ~patterns[u]
        if not nz.any():
            continue
        if nz.all():
            redesign = X
        else:
            redesign = X[nz][:, _qr_keep_cols(X[nz])]
            if redesign.shape[0] == redesign.shape[1]:
                continue
        cury, curo = sy[np.ix_(subg, nz)], offset[nz]
        last_beta = None
        for i in range(grid_length):
            l0[subg, i], last_beta = _adjusted_profile_lik(spline_disp[i], cury, redesign, curo, last_beta)

    overall = maximize_interpolant(spline_pts, l0.sum(axis=0)[None, :])[0]
    common = 0.1 * 2 ** overall

    ave = ave_log_cpm(y, lib_size, common)
    cov = ave[sel]
    span = choose_lowess_span(sy.shape[0])
    m0 = locfit_by_col(l0, cov, span)
    disp_trend = 0.1 * 2 ** maximize_interpolant(spline_pts, m0)
    trended = np.full(ntags, disp_trend[np.argmin(cov)])
    trended[sel] = disp_trend

    _, mu, dev = _glm_fit(sy, X, offset, disp_trend)
    df_resid = _resid_df((sy < 1e-4) & (mu < 1e-4), X)
    with np.errstate(divide="ignore", invalid="ignore"):
        s2 = dev / df_resid
    s2[df_resid == 0] = 0
    s2 = np.maximum(s2, 0)
    prior_df = squeeze_var_df_prior(s2, df_resid, cov)
    prior_n = prior_df / (nlibs - X.shape[1])

    tagwise = trended.copy()
    if not prior_n > 1e6:
        tagwise[sel] = 0.1 * 2 ** maximize_interpolant(spline_pts, l0 + prior_n * m0)
    return {"common.dispersion": common, "trended.dispersion": trended,
            "tagwise.dispersion": tagwise, "span": span, "prior.df": prior_df, "prior.n": prior_n}


def edger_tagwise_dispersion(counts: np.ndarray, design: np.ndarray) -> np.ndarray:
    """
    Python equivalent of
        d <- calcNormFactors(DGEList(t(counts))); estimateDisp(d, design)$tagwise.dispersion
    counts: bags x communities; design: bags x coefficients. Returns one dispersion per community.
    """
    y = np.asarray(counts, dtype=float).T
    lib_size = y.sum(axis=0) * calc_norm_factors_tmm(y)
    return estimate_disp(y, design, lib_size)["tagwise.dispersion"]
