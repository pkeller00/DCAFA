# dcafa

Statistical tests for patch-level tissue communities in whole-slide images:

- **CA1 (community abundance)**: does a patient-level target (e.g. response) change how much of each community a
  patient has? One negative-binomial GLM per community, `count ~ target + covariates` with offset `log(n patches)`,
  method-of-moments dispersion, robust (HC) standard errors and BH FDR across communities. Because community shares
  sum to one, `handle_collinearity="auto"` reports deviation-coded effects (centred across communities).
  An alternative `backend="voomclr"` is a pure-Python port of R
  [voomCLR](https://github.com/koenvandenberge/voomCLR) 0.99.41 (matches R to ~1e-7, no R needed; edgeR
  dispersion, limma squeezeVar, locfit and R lowess ported in `dcafa/_edger.py`; see `docs/`).
- **FA1 (feature association)**: univariate GLMs of each target on each patch-level feature, with clustered SEs by
  patient (bag) and BH FDR across features. It shows which features define or separate communities.

## Install

```bash
pip install git+https://github.com/pkeller00/DCAFA.git@v0.1.0
# or, for development
pip install -e ".[test]"
```

## Usage

```python
from dcafa import fit_ca1_multi_targets, fit_fa1_univariate_multi_targets

# counts: one row per patient, community count columns c_1..c_K, total n, target and covariates
res = fit_ca1_multi_targets(
    df=counts, K=10, target_cols=["Response"], covariates=["age", "male"],
    community_cols=[f"c_{i}" for i in range(1, 11)], offset_col="n",
    cov_type="HC1", handle_collinearity="auto", ci_level=0.90, fdr_alpha=0.10,
    backend="nb", outdir=None, plot=False,
)["Response"]
# rows with term == "Response": percent_change (rate ratio - 1), 90% CI, p, BH q per community
```

## Tests

`pytest` checks the package against reference outputs of the original research implementation on synthetic data
(`tests/synthetic.py`, `tests/make_reference.py`; no patient data in this repository).

## License

[PolyForm Noncommercial 1.0.0](LICENSE.md): free for research, personal and other non-commercial use; commercial use is
not permitted.

Required Notice: Copyright Piotr Keller (https://github.com/pkeller00/DCAFA)
