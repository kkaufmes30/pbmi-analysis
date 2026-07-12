# -*- coding: utf-8 -*-
"""Generate the figures for the PBMI systematic review on sex analysis of
lung-cancer machine-learning datasets.

Reads ``analyse_schema_final.csv`` (expected next to this file) and writes
five figures, each as PNG/SVG/PDF, into ``figures/`` (with on-figure titles)
and ``figures/no_title/`` (same figures without titles, for slides or captions):

    fig1_per_study_split    sex split per study, sorted by % male (landscape)
    fig1_..._portrait       the same, as a tall portrait chart
    fig3_sex_handling       how sex is used across the ML pipeline
    fig5_deviation_boxplot  % female per split, one study traced across splits
    fig6_reporting_by_split how many studies report sex numbers per split
    fig7_balance_by_split   % female per split (plain boxplot)

It also prints data-quality ("reconciliation") warnings for any study whose
total-cohort counts or percentages do not add up.

Usage:
    python generate_figures.py
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless backend: render straight to file
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "analyse_schema_final.csv")
OUT_DIR = os.path.join(HERE, "figures")

# Column order of the raw CSV (semicolon-separated, cp1252-encoded, with
# German-style decimal commas and "male;female" pairs inside quoted cells).
COLUMNS = [
    "paper_id", "cancer_type", "task", "total_n",
    "total_abs_mf", "total_pct_mf",
    "train_abs_mf", "train_pct_mf",
    "val_abs_mf", "val_pct_mf",
    "test_abs_mf", "test_pct_mf",
    "ext_abs_mf", "ext_pct_mf",
    "sex_specific_model", "sex_in_dev", "sex_in_eval", "notes",
]

# The three binary "how was sex handled" flags.
FLAG_COLUMNS = ["sex_specific_model", "sex_in_dev", "sex_in_eval"]

# Data splits, as (label, absolute-counts column, percentage column).
SPLITS = [
    ("Total", "total_abs_mf", "total_pct_mf"),
    ("Training", "train_abs_mf", "train_pct_mf"),
    ("Validation", "val_abs_mf", "val_pct_mf"),
    ("Test", "test_abs_mf", "test_pct_mf"),
    ("External", "ext_abs_mf", "ext_pct_mf"),
]

# In the per-split boxplot, a point is highlighted when its female share differs
# from that study's own total cohort by more than this many percentage points.
DRIFT_THRESHOLD = 10

# The single study traced across splits in the per-split boxplot: balanced in its
# cohort, then collapses to male-heavy in the external set.
HIGHLIGHT_STUDY = "10.1200/CCI.24.00133"

# Every figure is written in these formats. PNG for quick preview, SVG for
# PowerPoint (scales without blurring), PDF for the manuscript.
OUTPUT_FORMATS = ("png", "svg", "pdf")

# Output state, reassigned per render pass in main(): titled figures go to
# OUT_DIR, an untitled copy (for separate captions/slide titles) to a subfolder.
SHOW_TITLES = True

# Clean, colorblind-friendly palette, shared across all figures.
COL_MALE = "#4C78A8"       # calm blue
COL_FEMALE = "#E45756"     # warm red
COL_GREY = "#9AA0A6"
COL_DIM = "#C2CCD6"  # splits that track their cohort (small drift)
COL_PIPELINE = ["#4C78A8", "#72B7B2", "#B279A2"]

PLOT_STYLE = {
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def to_float(value):
    """Parse a single number, tolerating German decimals and blank markers.

    Returns ``np.nan`` for empty cells and the placeholders "-" and "NR".
    """
    if value is None:
        return np.nan
    text = str(value).strip()
    if text in ("", "-", "nan", "NR"):
        return np.nan
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return np.nan


def split_pair(value):
    """Split a quoted ``"male;female"`` cell into a ``(male, female)`` tuple.

    Returns ``(nan, nan)`` when the cell is missing or has fewer than two
    parts.
    """
    if value is None:
        return (np.nan, np.nan)
    text = str(value).strip()
    if text in ("", "-", "nan"):
        return (np.nan, np.nan)
    parts = [p for p in re.split(r";", text) if p.strip() != ""]
    if len(parts) < 2:
        return (np.nan, np.nan)
    return (to_float(parts[0]), to_float(parts[1]))


# ---------------------------------------------------------------------------
# Data loading & preparation
# ---------------------------------------------------------------------------
def load_studies():
    """Read the CSV and return one row per study (URL-keyed rows only)."""
    raw = pd.read_csv(
        SRC, sep=";", quotechar='"', engine="python",
        header=None, skiprows=1, dtype=str, encoding="cp1252",
    )
    df = raw.iloc[:, : len(COLUMNS)].copy()
    df.columns = COLUMNS
    is_study = df["paper_id"].notna() & df["paper_id"].str.startswith("http", na=False)
    return df[is_study].reset_index(drop=True)


def add_derived_columns(df):
    """Add the helper columns the figures rely on."""
    df["short"] = df["paper_id"].str.replace(r"https?://dx.doi.org/", "", regex=True)

    df["pct_male"] = df.apply(_total_pct_male, axis=1)
    df["pct_female"] = 100 - df["pct_male"]

    for col in FLAG_COLUMNS:
        df[col + "_i"] = df[col].apply(to_float)
    return df


def _total_pct_male(row):
    """% male in the total cohort: trust the reported %, else derive from counts."""
    male_pct, _ = split_pair(row["total_pct_mf"])
    if not np.isnan(male_pct):
        return male_pct
    male_abs, female_abs = split_pair(row["total_abs_mf"])
    if not np.isnan(male_abs) and not np.isnan(female_abs) and (male_abs + female_abs) > 0:
        return 100 * male_abs / (male_abs + female_abs)
    return np.nan


def pct_female(row, abs_col, pct_col):
    """% female for one split: trust the reported %, else derive from counts.

    Returns ``np.nan`` when the study did not report that split at all.
    """
    male_pct, female_pct = split_pair(row[pct_col])
    if not np.isnan(female_pct):
        return female_pct
    if not np.isnan(male_pct):
        return 100 - male_pct
    male_abs, female_abs = split_pair(row[abs_col])
    if not np.isnan(male_abs) and not np.isnan(female_abs) and (male_abs + female_abs) > 0:
        return 100 * female_abs / (male_abs + female_abs)
    return np.nan


def split_female_table(df):
    """Build a studies x splits table of % female (NaN where a split is absent)."""
    table = pd.DataFrame({"short": df["short"]})
    for label, abs_col, pct_col in SPLITS:
        table[label] = df.apply(lambda r: pct_female(r, abs_col, pct_col), axis=1)
    return table


def reconciliation_warnings(df):
    """Return ``(study, message)`` pairs where the reported numbers don't add up."""
    warnings = []
    for _, row in df.iterrows():
        male_abs, female_abs = split_pair(row["total_abs_mf"])
        male_pct, female_pct = split_pair(row["total_pct_mf"])
        total_n = to_float(row["total_n"])

        problems = []
        if not any(np.isnan([male_abs, female_abs, total_n])) and abs((male_abs + female_abs) - total_n) > 1:
            problems.append(f"abs sum {male_abs + female_abs:.0f} != total {total_n:.0f}")
        if not any(np.isnan([male_pct, female_pct])) and abs((male_pct + female_pct) - 100) > 1.5:
            problems.append(f"pct sum {male_pct + female_pct:.1f} != 100")
        if problems:
            warnings.append((row["short"], "; ".join(problems)))
    return warnings


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _save(fig, name):
    """Write the figure to OUT_DIR in every format in OUTPUT_FORMATS."""
    stem = os.path.splitext(name)[0]
    for ext in OUTPUT_FORMATS:
        fig.savefig(os.path.join(OUT_DIR, f"{stem}.{ext}"))
    plt.close(fig)


def _title(ax, text, **kwargs):
    """Set the axes title unless titles are switched off for this render pass."""
    if SHOW_TITLES:
        ax.set_title(text, **kwargs)


def fig_per_study_split(df):
    """Fig 1 — vertical M/F bars per study (landscape), sorted most to least male."""
    data = (df.dropna(subset=["pct_male"])
              .sort_values("pct_male", ascending=False)
              .reset_index(drop=True))
    x = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x, data["pct_male"], color=COL_MALE, label="Male")
    ax.bar(x, data["pct_female"], bottom=data["pct_male"], color=COL_FEMALE, label="Female")
    ax.axhline(50, color="black", lw=1.2, ls="--")

    ax.set_xticks(x)
    ax.set_xticklabels(data["short"], rotation=90, fontsize=6.5)
    ax.set_xlim(-0.6, len(data) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of cohort (%)")
    _title(ax, f"Sex distribution per study (total cohort), n={len(data)} lung-cancer ML studies")

    # % male printed inside each male segment, read bottom-to-top.
    for xi, pct in zip(x, data["pct_male"]):
        ax.text(xi, pct / 2, f"{pct:.0f}", ha="center", va="center",
                rotation=90, fontsize=6, color="white")
    ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.5), framealpha=0.9)
    _save(fig, "fig1_per_study_split.png")


def fig_per_study_split_portrait(df):
    """Fig 1 (portrait) — horizontal M/F bars per study, most male at top."""
    data = (df.dropna(subset=["pct_male"])
              .sort_values("pct_male")
              .reset_index(drop=True))
    y = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(9, 0.34 * len(data) + 1.2))
    ax.barh(y, data["pct_male"], color=COL_MALE, label="Male")
    ax.barh(y, data["pct_female"], left=data["pct_male"], color=COL_FEMALE, label="Female")
    ax.axvline(50, color="black", lw=1.2, ls="--")

    ax.set_yticks(y)
    ax.set_yticklabels(data["short"], fontsize=7)
    ax.set_ylim(-0.6, len(data) - 0.4)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of cohort (%)")
    _title(ax, f"Sex distribution per study (total cohort), n={len(data)} lung-cancer ML studies")

    # % male printed inside each male segment.
    for yi, pct in zip(y, data["pct_male"]):
        ax.text(pct, yi, f" {pct:.0f}", va="center",
                ha="left" if pct < 88 else "right", fontsize=6.5, color="white")
    ax.legend(loc="lower right", framealpha=0.9)
    _save(fig, "fig1_per_study_split_portrait.png")


def fig_sex_handling(df):
    """Fig 3 — count of studies using sex at each stage of the ML pipeline."""
    n = len(df)
    labels = ["Sex used in\nmodel development",
              "Sex used in\nmodel evaluation",
              "Sex-specific\nmodel trained"]
    counts = [int(np.nansum(df["sex_in_dev_i"])),
              int(np.nansum(df["sex_in_eval_i"])),
              int(np.nansum(df["sex_specific_model_i"]))]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, counts, color=COL_PIPELINE, width=0.6)
    ax.axhline(n, color=COL_GREY, ls="--", lw=1)
    ax.text(2.4, n, f"all studies (n={n})", va="bottom", ha="right",
            fontsize=8, color=COL_GREY)

    ax.set_ylim(0, n + 2)
    ax.set_ylabel("Number of studies")
    _title(ax, "How sex is handled in the ML pipeline")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 0.3,
                f"{count}/{n}\n({100 * count / n:.0f}%)",
                ha="center", va="bottom", fontsize=9)
    _save(fig, "fig3_sex_handling.png")
    return counts


def _split_boxplot(ax, table, labels, series, positions):
    """Draw the shared per-split boxplot: boxes, parity line, points on center."""
    ax.axhline(50, color="#333333", lw=1.2, ls="--", label="parity (50%)")

    bp = ax.boxplot(series, positions=positions, widths=0.55, patch_artist=True,
                    showfliers=False, medianprops=dict(color="#222222", lw=1.6))
    for box in bp["boxes"]:
        box.set(facecolor=COL_MALE, alpha=0.18, edgecolor=COL_MALE)
    for part in ("whiskers", "caps"):
        for line in bp[part]:
            line.set_color(COL_MALE)

    for pos, label in zip(positions, labels):
        values = table[label].dropna().to_numpy()
        ax.scatter(np.full(len(values), pos), values, s=26, color=COL_MALE,
                   alpha=0.40, edgecolor="white", linewidth=0.4, zorder=2)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{label}\n(n={len(v)})" for label, v in zip(labels, series)])
    ax.set_ylim(0, 100)
    ax.set_ylabel("% female in split")


def fig_balance_by_split(df):
    """Fig 7 — plain boxplot of % female per split (descriptive; small, unequal n)."""
    table = split_female_table(df)
    labels = [label for label, _, _ in SPLITS]
    series = [table[label].dropna().to_numpy() for label in labels]
    positions = np.arange(1, len(labels) + 1)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    _split_boxplot(ax, table, labels, series, positions)
    _title(ax, "Sex balance by data split")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    _save(fig, "fig7_balance_by_split.png")


def fig_deviation_boxplot(df):
    """Fig 5 — per-split boxplot with one study traced across splits.

    Descriptive only (small, unequal n). HIGHLIGHT_STUDY is drawn as a red line
    over its reported splits to show a cohort that is balanced overall but
    collapses to male-heavy in the external set.
    """
    table = split_female_table(df)
    labels = [label for label, _, _ in SPLITS]
    series = [table[label].dropna().to_numpy() for label in labels]
    positions = np.arange(1, len(labels) + 1)
    pos_of = {label: pos for label, pos in zip(labels, positions)}

    fig, ax = plt.subplots(figsize=(8, 4.6))
    _split_boxplot(ax, table, labels, series, positions)

    # Trace the one highlighted study across the splits it reports.
    match = table.loc[table["short"] == HIGHLIGHT_STUDY]
    if not match.empty:
        row = match.iloc[0]
        total = row["Total"]
        xs, ys, drifts = [], [], []
        for label in labels:
            value = row[label]
            if not np.isnan(value):
                xs.append(pos_of[label])
                ys.append(value)
                drifts.append(value - total)
        ax.plot(xs, ys, color=COL_FEMALE, lw=2.0, zorder=4,
                label=HIGHLIGHT_STUDY.split("/")[-1])
        ax.scatter(xs, ys, color=COL_FEMALE, s=55, edgecolor="white", linewidth=0.8,
                   zorder=5)
        worst = int(np.argmax(np.abs(drifts)))
        ax.annotate(f"{drifts[worst]:+.0f} pts vs cohort", (xs[worst], ys[worst]),
                    textcoords="offset points", xytext=(0, -14), ha="center", va="top",
                    fontsize=8, color=COL_FEMALE, zorder=6)

    _title(ax, "Sex balance by data split")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    _save(fig, "fig5_deviation_boxplot.png")


def fig_reporting_by_split(df):
    """Fig 6 — how many studies report sex numbers for each data split."""
    table = split_female_table(df)
    labels = [label for label, _, _ in SPLITS]
    counts = [int(table[label].notna().sum()) for label in labels]
    n = len(df)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bars = ax.bar(x, counts, color=COL_MALE, width=0.62)
    ax.axhline(n, color=COL_GREY, ls="--", lw=1)
    ax.text(len(labels) - 0.6, n, f"all studies (n={n})", va="bottom", ha="right",
            fontsize=8, color=COL_GREY)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, n + 5)
    ax.set_ylabel("Number of studies")
    _title(ax, "Studies reporting sex numbers, by data split", pad=12)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 0.3,
                f"{count}\n({100 * count / n:.0f}%)",
                ha="center", va="bottom", fontsize=9)
    _save(fig, "fig6_reporting_by_split.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def render_all(df):
    """Draw and save every figure into the current OUT_DIR / SHOW_TITLES state."""
    os.makedirs(OUT_DIR, exist_ok=True)
    fig_per_study_split(df)
    fig_per_study_split_portrait(df)
    counts = fig_sex_handling(df)
    fig_deviation_boxplot(df)
    fig_reporting_by_split(df)
    fig_balance_by_split(df)
    return counts


def main():
    global OUT_DIR, SHOW_TITLES
    df = add_derived_columns(load_studies())

    print("=== reconciliation flags ===")
    for study, message in reconciliation_warnings(df):
        print(" ", study, "->", message)
    fem = df["pct_female"].dropna()
    print(f"\nN studies: {len(df)} | with usable sex data: {len(fem)}")
    print(f"Total-cohort female share: median {fem.median():.1f}% "
          f"(range {fem.min():.1f}-{fem.max():.1f}; "
          f"mean {fem.mean():.1f}%, SD {fem.std():.1f})")

    plt.rcParams.update(PLOT_STYLE)
    base_dir = OUT_DIR

    # Pass 1: titled figures in figures/. Pass 2: untitled copies in figures/no_title/.
    OUT_DIR, SHOW_TITLES = base_dir, True
    counts = render_all(df)
    OUT_DIR, SHOW_TITLES = os.path.join(base_dir, "no_title"), False
    render_all(df)

    print("\nFlags:", counts, "(dev, eval, sex-specific) of", len(df))
    print("Saved titled figures to:", base_dir)
    print("Saved untitled figures to:", os.path.join(base_dir, "no_title"))


if __name__ == "__main__":
    main()
