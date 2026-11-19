"""DCAFA: Differential Community Abundance and Feature Analysis of patch-level tissue communities.

CA1  fit_ca1_multi_targets              bag-level targets: per-community negative-binomial GLM
                                        (count ~ target + covariates, offset log n), or the voomCLR backend
                                        (pure-Python port of R voomCLR 0.99.41)
CA2  fit_ca2_memberships                instance-level target ~ community memberships, bag-clustered SEs
FA1  fit_fa1_univariate_multi_targets   univariate GLMs of each target on each patch-level feature
FA2  fit_fa2_aggregate_multi_targets    bag-level targets ~ bag-aggregated (global / community-local) features;
                                        build the design with aggregate_bag_features
"""
from .ca1 import fit_ca1_multi_targets, forest_plot_publication
from .ca2 import fit_ca2_memberships, plot_ca2_forest
from .fa1 import fit_fa1_univariate_multi_targets
from .fa2 import aggregate_bag_features, fit_fa2_aggregate_multi_targets, plot_fa2_forest

__all__ = [
    "fit_ca1_multi_targets", "forest_plot_publication",
    "fit_ca2_memberships", "plot_ca2_forest",
    "fit_fa1_univariate_multi_targets",
    "aggregate_bag_features", "fit_fa2_aggregate_multi_targets", "plot_fa2_forest",
]
__version__ = "0.2.0"
