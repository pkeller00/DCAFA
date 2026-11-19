"""Deterministic synthetic inputs for the regression tests (no patient data)."""
import numpy as np
import pandas as pd

K = 6
COMM = [f"c_{i}" for i in range(1, K + 1)]


def community_counts(seed: int = 0, n_patients: int = 60) -> pd.DataFrame:
    """Per-patient community counts with a binary response, an age and a sex covariate.

    Communities c_2 and c_5 are enriched / depleted in responders, so CA1 has real effects to find.
    """
    rng = np.random.default_rng(seed)
    resp = rng.integers(0, 2, n_patients)
    age = rng.normal(62, 9, n_patients).round(0)
    male = rng.integers(0, 2, n_patients).astype(float)
    base = rng.dirichlet(np.full(K, 2.0), n_patients)
    base[:, 1] *= np.where(resp == 1, 1.8, 1.0)
    base[:, 4] *= np.where(resp == 1, 0.5, 1.0)
    base /= base.sum(1, keepdims=True)
    n = rng.integers(150, 2000, n_patients)
    counts = np.vstack([rng.multinomial(n_i, p) for n_i, p in zip(n, base)])
    df = pd.DataFrame(counts, columns=COMM, index=[f"P{i:03d}" for i in range(n_patients)])
    df["n"] = df[COMM].sum(1)
    df["Response"] = resp
    df["age"] = age
    df["male"] = male
    return df


def patch_features(seed: int = 1, n_bags: int = 40, per_bag: int = 25, d: int = 8) -> pd.DataFrame:
    """Patch-level features x_1..x_d in bags (patients) with a bag-level binary target t_r1."""
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, n_bags)
    rows = []
    for b in range(n_bags):
        X = rng.normal(0, 1, (per_bag, d))
        X[:, 0] += 0.6 * t[b]  # x_1 associated with the target
        for x in X:
            rows.append({"bag_id": f"B{b:03d}", "t_r1": t[b], **{f"x_{j + 1}": v for j, v in enumerate(x)}})
    return pd.DataFrame(rows)


CASES = {
    "nb_unadjusted": dict(covariates=[], backend="nb"),
    "nb_age_sex": dict(covariates=["age", "male"], backend="nb"),
    "voomclr_unadjusted": dict(covariates=[], backend="voomclr"),
}


def patch_memberships(seed: int = 2, n_bags: int = 40, per_bag: int = 30, d: int = 4, k: int = 4) -> pd.DataFrame:
    """Patches with features x_1..x_d, exclusive community memberships m_1..m_k, a binary instance-level target
    t (more likely in m_2, less in m_4), a bag-level continuous target y (driven by x_1) and a bag covariate age."""
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 1, n_bags)
    age = rng.normal(60, 10, n_bags).round(0)
    gamma = np.array([0.0, 1.0, 0.0, -1.0])[:k]
    rows = []
    for b in range(n_bags):
        comm = rng.integers(0, k, per_bag)
        X = rng.normal(0, 1, (per_bag, d))
        X[:, 0] += 0.8 * y[b]
        X[:, 1] += 0.5 * (comm == 1)  # x_2 differs by community, for the community means
        eta = -0.3 + gamma[comm] + rng.normal(0, 0.5)
        t = rng.random(per_bag) < 1 / (1 + np.exp(-eta))
        for i in range(per_bag):
            rows.append({"bag_id": f"B{b:03d}", "t": int(t[i]), "y": y[b], "age": age[b],
                         **{f"x_{j + 1}": X[i, j] for j in range(d)},
                         **{f"m_{c + 1}": float(comm[i] == c) for c in range(k)}})
    return pd.DataFrame(rows)
