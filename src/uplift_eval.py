"""Shared evaluation code for the Team 1B uplift project.

Every ranking method (random baseline, response model, T-learner,
transformed-outcome model) should be scored with these functions so that
results stay comparable, as the project brief requires.

Conventions
-----------
* Higher score = ranked higher = targeted first.
* Group 1 is the highest-ranked group.
* Ties in scores are broken by row order (stable sort), so results are
  deterministic for a given dataset file.
* Uplift = treatment visit rate - control visit rate (intention-to-treat).
* 95% confidence interval = uplift +/- 1.96 * standard error, using the
  difference-in-proportions formula from the project brief.
* Incremental visits = group size x uplift.

Nothing in this file uses `exposure`, and no outcome is used to build scores.
"""

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [f"f{i}" for i in range(12)]
SPECIAL_COLUMNS = ["treatment", "exposure", "visit", "conversion"]
Z_95 = 1.96


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def find_data_dir():
    """Return the data folder whether the notebook runs from the repo root
    or from the notebooks folder."""
    for candidate in (Path("data"), Path("../data")):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find the data folder. Run from the repository root "
        "or from the notebooks folder."
    )


def load_splits(data_dir=None):
    """Load train, validation, and test as a dict of DataFrames."""
    data_dir = Path(data_dir) if data_dir is not None else find_data_dir()
    return {
        name: pd.read_csv(data_dir / f"{name}.csv.gz")
        for name in ("train", "validation", "test")
    }


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def diff_in_proportions(visits_t, n_t, visits_c, n_c, z=Z_95):
    """Uplift, standard error, and CI for treatment rate minus control rate.

    Works on scalars or NumPy arrays.
    """
    visits_t = np.asarray(visits_t, dtype=float)
    visits_c = np.asarray(visits_c, dtype=float)
    n_t = np.asarray(n_t, dtype=float)
    n_c = np.asarray(n_c, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_t = visits_t / n_t
        rate_c = visits_c / n_c
        uplift = rate_t - rate_c
        se = np.sqrt(rate_t * (1 - rate_t) / n_t + rate_c * (1 - rate_c) / n_c)
    return rate_t, rate_c, uplift, se, uplift - z * se, uplift + z * se


def overall_itt(treatment, outcome):
    """Overall intention-to-treat effect for one split, as a one-row dict."""
    t = np.asarray(treatment).astype(bool)
    y = np.asarray(outcome).astype(float)
    n_t, n_c = t.sum(), (~t).sum()
    v_t, v_c = y[t].sum(), y[~t].sum()
    rate_t, rate_c, uplift, se, lo, hi = diff_in_proportions(v_t, n_t, v_c, n_c)
    n = len(y)
    return {
        "n": n,
        "n_treatment": int(n_t),
        "n_control": int(n_c),
        "visits_treatment": int(v_t),
        "visits_control": int(v_c),
        "treatment_rate": float(rate_t),
        "control_rate": float(rate_c),
        "uplift": float(uplift),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "incremental_visits": float(n * uplift),
        "incremental_ci_low": float(n * lo),
        "incremental_ci_high": float(n * hi),
    }


# ---------------------------------------------------------------------------
# Ranking into groups
# ---------------------------------------------------------------------------

def _order_by_score(scores):
    """Row positions sorted from highest to lowest score, ties by row order."""
    scores = np.asarray(scores, dtype=float)
    if np.isnan(scores).any():
        raise ValueError("Scores contain NaN values.")
    return np.argsort(-scores, kind="stable")


def assign_groups(scores, n_groups=10):
    """Group label (1 = top) for each row, in the original row order.

    Groups are as equal in size as possible.
    """
    order = _order_by_score(scores)
    n = len(order)
    groups = np.empty(n, dtype=int)
    groups[order] = (np.arange(n) * n_groups) // n + 1
    return groups


def uplift_by_group(scores, treatment, outcome, n_groups=10):
    """Uplift table by ranked group, with every column the brief asks for."""
    t = np.asarray(treatment).astype(bool)
    y = np.asarray(outcome).astype(float)
    groups = assign_groups(scores, n_groups)

    size = np.bincount(groups, minlength=n_groups + 1)[1:]
    n_t = np.bincount(groups, weights=t, minlength=n_groups + 1)[1:]
    v_t = np.bincount(groups, weights=y * t, minlength=n_groups + 1)[1:]
    n_c = size - n_t
    v_c = np.bincount(groups, weights=y * ~t, minlength=n_groups + 1)[1:]

    rate_t, rate_c, uplift, se, lo, hi = diff_in_proportions(v_t, n_t, v_c, n_c)
    return pd.DataFrame({
        "group": np.arange(1, n_groups + 1),
        "n": size,
        "n_treatment": n_t.astype(int),
        "n_control": n_c.astype(int),
        "visits_treatment": v_t.astype(int),
        "visits_control": v_c.astype(int),
        "treatment_rate": rate_t,
        "control_rate": rate_c,
        "uplift": uplift,
        "ci_low": lo,
        "ci_high": hi,
        "incremental_visits": size * uplift,
        "incremental_ci_low": size * lo,
        "incremental_ci_high": size * hi,
    })


# ---------------------------------------------------------------------------
# Cumulative uplift curve
# ---------------------------------------------------------------------------

def cumulative_uplift(scores, treatment, outcome, n_points=100):
    """Cumulative incremental visits when targeting the top k% by score.

    For each cutoff, uplift is estimated inside the targeted audience and
    multiplied by the audience size. `random_expected` is the straight line
    a ranking with no information would follow on average:
    fraction targeted x overall incremental visits.
    """
    t = np.asarray(treatment).astype(bool)
    y = np.asarray(outcome).astype(float)
    order = _order_by_score(scores)
    t_sorted, y_sorted = t[order], y[order]

    cum_n_t = np.cumsum(t_sorted)
    cum_v_t = np.cumsum(y_sorted * t_sorted)
    cum_n_c = np.cumsum(~t_sorted)
    cum_v_c = np.cumsum(y_sorted * ~t_sorted)

    n = len(y)
    cutoffs = np.unique(np.linspace(n / n_points, n, n_points).round().astype(int))
    idx = cutoffs - 1
    rate_t, rate_c, uplift, se, lo, hi = diff_in_proportions(
        cum_v_t[idx], cum_n_t[idx], cum_v_c[idx], cum_n_c[idx]
    )
    total_incremental = n * uplift[-1]
    fraction = cutoffs / n
    return pd.DataFrame({
        "fraction_targeted": fraction,
        "n_targeted": cutoffs,
        "uplift": uplift,
        "ci_low": lo,
        "ci_high": hi,
        "incremental_visits": cutoffs * uplift,
        "incremental_ci_low": cutoffs * lo,
        "incremental_ci_high": cutoffs * hi,
        "random_expected": fraction * total_incremental,
    })


def area_above_random(curve):
    """Average gap between the cumulative curve and the random line,
    divided by total incremental visits.

    0 means no better than random on average; positive means the ranking
    concentrates incremental visits early. This is a simple AUUC-style
    summary used to compare rankings, not an official Qini implementation.
    """
    total = curve["random_expected"].iloc[-1]
    gap = curve["incremental_visits"] - curve["random_expected"]
    return float(gap.mean() / total)


def incremental_in_top(curve, fraction=0.3):
    """Incremental visits captured by targeting the top `fraction` of the audience."""
    row = curve.iloc[(curve["fraction_targeted"] - fraction).abs().argmin()]
    return float(row["incremental_visits"])


# ---------------------------------------------------------------------------
# Random-ranking null distribution
# ---------------------------------------------------------------------------

def random_ranking_null(treatment, outcome, n_seeds=200, n_groups=10,
                        top_fraction=0.3, first_seed=0):
    """Score the same data with many random rankings.

    Shows how much a ranking with no information can vary by chance.
    Later models should be compared against this spread, not against zero.
    Only the random scores change between seeds; the data stays fixed, so
    this does not capture sampling variability of the data itself.
    """
    n = len(np.asarray(outcome))
    rows = []
    for seed in range(first_seed, first_seed + n_seeds):
        scores = np.random.default_rng(seed).random(n)
        table = uplift_by_group(scores, treatment, outcome, n_groups)
        curve = cumulative_uplift(scores, treatment, outcome)
        rows.append({
            "seed": seed,
            "top_group_uplift": table["uplift"].iloc[0],
            "best_minus_worst_group": table["uplift"].max() - table["uplift"].min(),
            f"incremental_top_{int(top_fraction * 100)}pct": incremental_in_top(curve, top_fraction),
            "area_above_random": area_above_random(curve),
        })
    return pd.DataFrame(rows)


def summarize_null(null_df, lower=2.5, upper=97.5):
    """Median and percentile range for each column of random_ranking_null."""
    cols = [c for c in null_df.columns if c != "seed"]
    return pd.DataFrame({
        "median": null_df[cols].median(),
        f"p{lower:g}": null_df[cols].quantile(lower / 100),
        f"p{upper:g}": null_df[cols].quantile(upper / 100),
    })


def percent_view(table, decimals=3):
    """Copy of a results table with rates and uplift shown in percent
    (percentage points for uplift) and incremental visits rounded."""
    view = table.copy()
    for col in ("treatment_rate", "control_rate", "uplift", "ci_low", "ci_high"):
        if col in view:
            view[col] = (view[col] * 100).round(decimals)
    for col in ("incremental_visits", "incremental_ci_low", "incremental_ci_high"):
        if col in view:
            view[col] = view[col].round(1)
    return view


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_uplift_by_group(table, itt=None, title="Uplift by group", ax=None):
    """Uplift per group with 95% CI error bars and an optional overall-ITT line."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    pct = 100
    ax.errorbar(
        table["group"], table["uplift"] * pct,
        yerr=[(table["uplift"] - table["ci_low"]) * pct,
              (table["ci_high"] - table["uplift"]) * pct],
        fmt="o-", capsize=4, label="Group uplift (95% CI)",
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    if itt is not None:
        ax.axhline(itt["uplift"] * pct, color="tab:orange", linewidth=1.5,
                   label=f"Overall ITT ({itt['uplift'] * pct:.2f} pp)")
    ax.set_xticks(table["group"])
    ax.set_xlabel("Group (1 = highest score)")
    ax.set_ylabel("Uplift (percentage points)")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_cumulative(curves, title="Cumulative incremental visits", ax=None,
                    show_ci=True):
    """Cumulative incremental visits for one or more rankings.

    `curves` is a dict of {label: cumulative_uplift DataFrame}. The random
    line is taken from the first curve.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    first = next(iter(curves.values()))
    for label, curve in curves.items():
        x = curve["fraction_targeted"] * 100
        ax.plot(x, curve["incremental_visits"], label=label)
        if show_ci:
            ax.fill_between(x, curve["incremental_ci_low"],
                            curve["incremental_ci_high"], alpha=0.15)
    ax.plot(first["fraction_targeted"] * 100, first["random_expected"],
            color="gray", linestyle="--", label="Random expectation")
    ax.set_xlabel("Share of audience targeted, highest scores first (%)")
    ax.set_ylabel("Estimated incremental visits")
    ax.set_title(title)
    ax.legend()
    return ax
