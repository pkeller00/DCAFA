# dcafa

**DCAFA (Differential Community Abundance and Feature Analysis)**: statistical tests for patch-level tissue
communities in whole-slide images.

Paper: [DCAFA: Differential Community Abundance and Feature Analysis for Histological Images (bioRxiv)](https://www.biorxiv.org/content/10.64898/2026.04.28.721329v1).
Original research code: [wgrgwrght/DCAFA](https://github.com/wgrgwrght/DCAFA).

|                    | bag-level target (e.g. patient response)       | instance-level target (e.g. patch label) |
|--------------------|------------------------------------------------|------------------------------------------|
| **Community abundance** | CA1 `fit_ca1_multi_targets`              | CA2 `fit_ca2_memberships`                |
| **Feature association** | FA2 `fit_fa2_aggregate_multi_targets`    | FA1 `fit_fa1_univariate_multi_targets`   |

- **CA1 (community abundance, bag-level targets)**: does a patient-level target (e.g. response) change how much of
  each community a patient has? One negative-binomial GLM per community, `count ~ target + covariates` with offset
  `log(n patches)`, method-of-moments dispersion, robust (HC) standard errors and BH FDR across communities. Because
  community shares sum to one, `handle_collinearity="auto"` reports deviation-coded effects (centred across
  communities). An alternative `backend="voomclr"` is a pure-Python port of
  [voomCLR](https://github.com/koenvandenberge/voomCLR) 0.99.41
  ([Takele Assefa, Verbist & Van den Berge, *Bioinformatics* 2025](https://doi.org/10.1093/bioinformatics/btaf637)).
  It matches R to ~1e-7 without needing R; edgeR dispersion, limma squeezeVar, locfit and R lowess are ported in
  `dcafa/_edger.py` (see [docs/voomclr_port.md](docs/voomclr_port.md)).
- **CA2 (community abundance, instance-level targets)**: does a patch's community predict its own outcome? One GLM
  `t ~ m_1 + ... + m_K + covariates` on hard or soft memberships, bag-clustered SEs, BH FDR across communities. For
  exclusive memberships the last community is the reference.
- **FA1 (feature association, instance-level targets)**: univariate GLMs of each target on each patch-level
  feature, with clustered SEs by patient (bag) and BH FDR across features. It shows which features define or separate
  communities.
- **FA2 (feature association, bag-level targets)**: patch features are averaged per bag, either globally (`xg_*`) or
  within each community (`xk{k}_*`, membership-weighted), by `aggregate_bag_features`. A multivariable GLM per
  target, `y ~ aggregated features + covariates`, is then fitted with HC SEs and BH FDR within the global and
  community-local feature sets.

## Install

```bash
pip install git+https://github.com/pkeller00/DCAFA.git@v0.2.0
# or, for development
pip install -e ".[test]"
```

## Usage

```python
from dcafa import (fit_ca1_multi_targets, fit_ca2_memberships,
                   fit_fa1_univariate_multi_targets, aggregate_bag_features, fit_fa2_aggregate_multi_targets)

# CA1 -- counts: one row per patient, community count columns c_1..c_K, total n, target and covariates
res = fit_ca1_multi_targets(
    df=counts, K=10, target_cols=["Response"], covariates=["age", "male"],
    community_cols=[f"c_{i}" for i in range(1, 11)], offset_col="n",
    cov_type="HC1", handle_collinearity="auto", ci_level=0.90, fdr_alpha=0.10,
    backend="nb", outdir=None, plot=False,
)["Response"]
# rows with term == "Response": percent_change (rate ratio - 1), 90% CI, p, BH q per community

# patches: one row per patch with features x_1..x_d, memberships m_1..m_K, bag_id and targets
# CA2 -- instance-level target on community memberships
ca2 = fit_ca2_memberships(patches, target_col="t", membership_cols=[f"m_{k}" for k in range(1, 11)],
                          bag_id_col="bag_id", family="binomial")

# FA1 -- each instance-level target on each patch feature
fa1 = fit_fa1_univariate_multi_targets(patches, target_cols=["t"], bag_id_col="bag_id",
                                       families={"t": "binomial"})

# FA2 -- bag-level target on bag-aggregated features
design = aggregate_bag_features(patches, membership_cols=[f"m_{k}" for k in range(1, 11)], bag_id_col="bag_id")
fa2 = fit_fa2_aggregate_multi_targets(design, bag_table[["bag_id", "Response", "age"]], target_cols=["Response"],
                                      covariate_cols=["age"], use_global_means=True, use_community_means=False,
                                      family_by_target={"Response": "binomial"}, bag_id_col="bag_id")
```

`forest_plot_publication`, `plot_ca2_forest` and `plot_fa2_forest` draw forest plots of the results.

## Tests

`pytest` checks the package against reference outputs of the original research implementations on synthetic data
(`tests/synthetic.py`, `tests/make_reference.py`; no patient data in this repository). CA2 and FA2 are checked
against `DCAFA.py` from [wgrgwrght/DCAFA](https://github.com/wgrgwrght/DCAFA).

## Citation

```
Wright, G., Keller, P., Muter, J., Brosens, J., Tejpar, S., & Minhas, F. (2026).
DCAFA: Differential Community Abundance and Feature Analysis for Histological Images.
bioRxiv. https://www.biorxiv.org/content/10.64898/2026.04.28.721329v1
```

If you use `backend="voomclr"`, please also cite voomCLR:

```
Takele Assefa, A., Verbist, B., & Van den Berge, K. (2025).
Assessing differential cell composition in single-cell studies using voomCLR.
Bioinformatics. https://doi.org/10.1093/bioinformatics/btaf637
```

## License

[PolyForm Noncommercial 1.0.0](LICENSE.md): free for research, personal and other non-commercial use; commercial use is
not permitted.

Required Notice: Copyright Piotr Keller (https://github.com/pkeller00/DCAFA)

CA2 and FA2 (`src/dcafa/ca2.py`, `src/dcafa/fa2.py`) are ported from the original DCAFA code, which is MIT-licensed;
its notice is in [LICENSE-MIT-upstream.md](LICENSE-MIT-upstream.md).
