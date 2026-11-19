# voomCLR backend (`backend="voomclr"`)

`fit_ca1_multi_targets(..., backend="voomclr")` is a pure-Python port of R
[voomCLR](https://github.com/koenvandenberge/voomCLR) 0.99.41
([Takele Assefa, Verbist & Van den Berge, *Bioinformatics* 2025](https://doi.org/10.1093/bioinformatics/btaf637)).
No R installation is needed. Result rows are tagged `coding="voomclr"`.

## Pipeline

CLR transform → unweighted fit for fitted counts → voom precision weights → weighted fit per community →
subtract the modal bias → limma empirical-Bayes variance moderation.

1. **NB dispersion** (for the precision weights), as `voomCLR:::.estimateNBDispersion`: TMM-normalised library
   sizes, Cox-Reid adjusted profile likelihood over a dispersion grid, a locfit abundance trend, then empirical-Bayes
   shrinkage of each community's dispersion towards that trend (edgeR `estimateDisp`, tagwise).
   Weight = 1 / [((K-1)/K)^2 · (1/mu + phi)].
2. **Empirical mean-variance trend** (only with `voom_var_calc="empirical"`): R `stats::lowess` (clowess,
   3 robustness iterations, delta = 0.01 · range(x)) of sqrt(residual SD) against mean log abundance, leaving out
   communities with zero total, read off with `approxfun(rule=2, ties=mean)`. Weight = 1 / trend^4.
3. **Bias (mode) correction**, as `voomCLR:::.getMode`: `modeest::mlv(method="meanshift")` on sqrt(n)·beta,
   starting at the shorth and climbing to the nearest Gaussian-KDE peak with bandwidth `bw.nrd0`. The mode is
   subtracted from every community's coefficient for that term.
4. **Variance moderation**, as limma `squeezeVar` / `fitFDist`: s² floored at 1e-5 · median, s2_prior includes the
   `logmdigamma(df_prior/2)` term, s2_prior = mean(s²) when df_prior = Inf, and df_total is capped at the pooled
   residual df.

## Ported sources (`src/dcafa/_edger.py`)

Line-by-line ports, for the arguments voomCLR uses, of edgeR 4.10.6 (estimateDisp, glmFit, mglmLevenberg,
mglmOneWay, adjustedProfileLik, maximizeInterpolant, aveLogCPM, WLEB, calcNormFactors/TMM), limma 3.68.5
(squeezeVar, fitFDist, fitFDistUnequalDF1, weightedLowess, trigammaInverse), locfit 1.5-9.12 (`deg=0`, adaptive
tree) and R stats (optimize, fmm spline, lowess, approxfun).

## Validation against R

- Full pipeline vs the R voomCLR package on 30 random datasets: coefficients, t and p agree to within about 1e-7,
  for both `voom_var_calc="analytical"` and `"empirical"`. A real cohort reproduced R's output to every printed
  digit for both settings, with and without covariates.
- Tagwise dispersions vs R edgeR on 46 synthetic datasets (continuous and one-way designs, all-zero groups,
  2-120 communities): within 4e-12 relative error.
- End-to-end `fit_ca1_multi_targets(backend="voomclr")` agrees with an Rscript-based implementation to 4e-14.
