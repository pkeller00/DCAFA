"""DCAFA: community-abundance (CA1) and feature-association (FA1) analysis of patch-level tissue communities.

CA1  fit_ca1_multi_targets     per-community negative-binomial GLM (count ~ target + covariates, offset log n),
                               or the voomCLR backend (pure-Python port of R voomCLR 0.99.41)
FA1  fit_fa1_univariate_multi_targets   univariate GLMs of each target on each patch-level feature
"""
from .ca1 import fit_ca1_multi_targets, forest_plot_publication
from .fa1 import fit_fa1_univariate_multi_targets

__all__ = ["fit_ca1_multi_targets", "forest_plot_publication", "fit_fa1_univariate_multi_targets"]
__version__ = "0.1.0"
