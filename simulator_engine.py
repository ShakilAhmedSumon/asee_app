import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator


def coerce_numeric(df, cols):
    df = df.copy()

    for col in cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def simulate_osu_latest_year(
    df_prepared,
    discipline,
    latest_year=2024,
    delta_ttt=0,
    delta_ntt=0,
    delta_d=0.0,
    delta_freshman_ft=0,
    delta_sophomore_ft=0,
    delta_junior_ft=0,
    delta_senior_ft=0,
    delta_freshman_pt=0,
    delta_sophomore_pt=0,
    delta_junior_pt=0,
    delta_senior_pt=0,
    delta_ms_ft=0,
    delta_ms_pt=0,
):
    """
    Simulates only the OSU latest-year row for a selected discipline.

    Important:
    - Peer schools are unchanged.
    - AAU benchmarks are unchanged.
    - Only OSU's latest-year R, STAR, D, and related fields are recalculated.
    - TTT faculty changes preserve the program's D ratio by assigning the
      program-average doctoral mentoring load to the changed faculty count.
    """

    df_sim = df_prepared.copy()

    mask = (
        (df_sim["School Name"] == "Oregon State University")
        & (df_sim["Discipline Name"] == discipline)
        & (df_sim["Year"] == latest_year)
    )

    if mask.sum() == 0:
        raise ValueError(
            f"No OSU row found for {discipline} in {latest_year}."
        )

    if mask.sum() > 1:
        # If multiple OSU rows exist, update all matching rows consistently.
        pass

    row_idx = df_sim.index[mask]

    numeric_cols = [
        "Doctoral Full Time Sum",
        "Total Tenured/Tenure-Track Faculty",
        "Full Time Teaching Faculty (Non-Tenure)",
        "FTE of Part Time Teaching Faculty (Non-Tenure)",
        "Freshman Full Time Sum",
        "Sophomore Full Time Sum",
        "Junior Full Time Sum",
        "Senior Full Time Sum",
        "Freshman Part Time Sum",
        "Sophomore Part Time Sum",
        "Junior Part Time Sum",
        "Senior Part Time Sum",
        "Masters Full Time Sum",
        "Masters Part Time Sum",
        "D_AAU",
        "alpha",
        "STAR_AAU",
        "S_AAU",
        "R_AAU",
        "Rs_AAU_SCH",
    ]

    df_sim = coerce_numeric(df_sim, numeric_cols)

    # ------------------------------------------------------------
    # Apply scenario changes
    # ------------------------------------------------------------
    baseline_d = pd.to_numeric(
        df_sim.loc[row_idx, "D"], errors="coerce"
    ).fillna(0.0)
    baseline_alpha = df_sim.loc[row_idx, "alpha"].copy()
    target_d = (baseline_d + float(delta_d)).clip(lower=0.0)

    df_sim.loc[row_idx, "Total Tenured/Tenure-Track Faculty"] += delta_ttt

    # Keep D independent of faculty changes. New TTT faculty are assumed to
    # mentor the degree program's average number of full-time doctoral students.
    df_sim.loc[row_idx, "Doctoral Full Time Sum"] = (
        target_d
        * df_sim.loc[row_idx, "Total Tenured/Tenure-Track Faculty"].clip(lower=0)
    )

    # Treat NTT increase as full-time NTT by default
    df_sim.loc[row_idx, "Full Time Teaching Faculty (Non-Tenure)"] += delta_ntt

    df_sim.loc[row_idx, "Freshman Full Time Sum"] += delta_freshman_ft
    df_sim.loc[row_idx, "Sophomore Full Time Sum"] += delta_sophomore_ft
    df_sim.loc[row_idx, "Junior Full Time Sum"] += delta_junior_ft
    df_sim.loc[row_idx, "Senior Full Time Sum"] += delta_senior_ft

    df_sim.loc[row_idx, "Freshman Part Time Sum"] += delta_freshman_pt
    df_sim.loc[row_idx, "Sophomore Part Time Sum"] += delta_sophomore_pt
    df_sim.loc[row_idx, "Junior Part Time Sum"] += delta_junior_pt
    df_sim.loc[row_idx, "Senior Part Time Sum"] += delta_senior_pt

    df_sim.loc[row_idx, "Masters Full Time Sum"] += delta_ms_ft
    df_sim.loc[row_idx, "Masters Part Time Sum"] += delta_ms_pt

    # Prevent negative simulated values
    nonnegative_cols = [
        "Doctoral Full Time Sum",
        "Total Tenured/Tenure-Track Faculty",
        "Full Time Teaching Faculty (Non-Tenure)",
        "FTE of Part Time Teaching Faculty (Non-Tenure)",
        "Freshman Full Time Sum",
        "Sophomore Full Time Sum",
        "Junior Full Time Sum",
        "Senior Full Time Sum",
        "Freshman Part Time Sum",
        "Sophomore Part Time Sum",
        "Junior Part Time Sum",
        "Senior Part Time Sum",
        "Masters Full Time Sum",
        "Masters Part Time Sum",
    ]

    for col in nonnegative_cols:
        df_sim.loc[row_idx, col] = df_sim.loc[row_idx, col].clip(lower=0)

    # ------------------------------------------------------------
    # Recalculate OSU latest-year row only
    # ------------------------------------------------------------
    # Preserve the requested D even if TTT is reduced to zero. This keeps the
    # scenario control authoritative and avoids a faculty-only alpha change.
    df_sim.loc[row_idx, "D"] = target_d.to_numpy()

    df_sim.loc[row_idx, "D_int"] = (
        df_sim.loc[row_idx, "D"]
        .round()
        .astype("Int64")
    )

    d_aau = df_sim.loc[row_idx, "D_AAU"]

    if np.isclose(float(delta_d), 0.0):
        # A faculty-only scenario must leave alpha exactly as prepared.
        df_sim.loc[row_idx, "alpha"] = baseline_alpha.to_numpy()
    else:
        df_sim.loc[row_idx, "alpha"] = 0.375

        alpha_mask = (
            (d_aau > 0)
            & (df_sim.loc[row_idx, "D"] < d_aau)
        )

        affected_alpha_idx = row_idx[alpha_mask.to_numpy()]

        df_sim.loc[affected_alpha_idx, "alpha"] = (
            0.75
            - 0.375
            * (
                df_sim.loc[affected_alpha_idx, "D"]
                / df_sim.loc[affected_alpha_idx, "D_AAU"]
            )
        )

    df_sim.loc[row_idx, "alpha"] = (
        df_sim.loc[row_idx, "alpha"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.375)
        .round(3)
    )

    # R view: all UG years + MS
    df_sim.loc[row_idx, "UG_FT_R"] = (
        df_sim.loc[row_idx, "Freshman Full Time Sum"]
        + df_sim.loc[row_idx, "Sophomore Full Time Sum"]
        + df_sim.loc[row_idx, "Junior Full Time Sum"]
        + df_sim.loc[row_idx, "Senior Full Time Sum"]
    )

    df_sim.loc[row_idx, "UG_PT_R"] = (
        df_sim.loc[row_idx, "Freshman Part Time Sum"]
        + df_sim.loc[row_idx, "Sophomore Part Time Sum"]
        + df_sim.loc[row_idx, "Junior Part Time Sum"]
        + df_sim.loc[row_idx, "Senior Part Time Sum"]
    )

    # STAR view: junior/senior only
    df_sim.loc[row_idx, "UG_FT_S"] = (
        df_sim.loc[row_idx, "Junior Full Time Sum"]
        + df_sim.loc[row_idx, "Senior Full Time Sum"]
    )

    df_sim.loc[row_idx, "UG_PT_S"] = (
        df_sim.loc[row_idx, "Junior Part Time Sum"]
        + df_sim.loc[row_idx, "Senior Part Time Sum"]
    )

    df_sim.loc[row_idx, "MS_FT"] = df_sim.loc[row_idx, "Masters Full Time Sum"]
    df_sim.loc[row_idx, "MS_PT"] = df_sim.loc[row_idx, "Masters Part Time Sum"]

    df_sim.loc[row_idx, "TTT"] = df_sim.loc[row_idx, "Total Tenured/Tenure-Track Faculty"]

    df_sim.loc[row_idx, "NTT"] = (
        df_sim.loc[row_idx, "Full Time Teaching Faculty (Non-Tenure)"]
        + df_sim.loc[row_idx, "FTE of Part Time Teaching Faculty (Non-Tenure)"]
    )

    # R recalculation
    df_sim.loc[row_idx, "R_raw_num"] = (
        df_sim.loc[row_idx, "UG_FT_R"]
        + df_sim.loc[row_idx, "UG_PT_R"]
        + df_sim.loc[row_idx, "MS_FT"]
        + df_sim.loc[row_idx, "MS_PT"]
    )

    df_sim.loc[row_idx, "R_raw_den"] = (
        df_sim.loc[row_idx, "TTT"]
        + df_sim.loc[row_idx, "NTT"]
    )

    df_sim.loc[row_idx, "R_raw"] = np.where(
        df_sim.loc[row_idx, "R_raw_den"] > 0,
        df_sim.loc[row_idx, "R_raw_num"] / df_sim.loc[row_idx, "R_raw_den"],
        np.nan
    )

    df_sim.loc[row_idx, "R_raw"] = (
        df_sim.loc[row_idx, "R_raw"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .round(3)
    )

    df_sim.loc[row_idx, "R_raw_int"] = (
        df_sim.loc[row_idx, "R_raw"]
        .round()
        .astype("Int64")
    )

    # STAR recalculation
    df_sim.loc[row_idx, "STAR_num"] = (
        df_sim.loc[row_idx, "UG_FT_S"] * 40
        + df_sim.loc[row_idx, "UG_PT_S"] * 20
        + df_sim.loc[row_idx, "MS_FT"] * 32
        + df_sim.loc[row_idx, "MS_PT"] * 21
    )
    df_sim.loc[row_idx, "SCH"] = df_sim.loc[row_idx, "STAR_num"]

    df_sim.loc[row_idx, "STAR_den"] = (
        df_sim.loc[row_idx, "alpha"] * df_sim.loc[row_idx, "TTT"]
        + 0.9 * df_sim.loc[row_idx, "NTT"]
    )

    df_sim.loc[row_idx, "STAR"] = np.where(
        df_sim.loc[row_idx, "STAR_den"] > 0,
        df_sim.loc[row_idx, "STAR_num"] / df_sim.loc[row_idx, "STAR_den"],
        np.nan
    )

    df_sim.loc[row_idx, "STAR"] = (
        df_sim.loc[row_idx, "STAR"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .round(3)
    )
    df_sim.loc[row_idx, "S_den"] = df_sim.loc[row_idx, "STAR_den"]
    df_sim.loc[row_idx, "S"] = df_sim.loc[row_idx, "STAR"]

    # Backward-compatible aliases
    df_sim.loc[row_idx, "Rs_den"] = df_sim.loc[row_idx, "S_den"]
    df_sim.loc[row_idx, "Rs_SCH"] = df_sim.loc[row_idx, "S"]

    df_sim.loc[row_idx, "Rs_num"] = df_sim.loc[row_idx, "R_raw_num"]
    df_sim.loc[row_idx, "Rs_den_ww"] = df_sim.loc[row_idx, "R_raw_den"]
    df_sim.loc[row_idx, "R_s"] = df_sim.loc[row_idx, "R_raw"]
    df_sim.loc[row_idx, "R_s_int"] = df_sim.loc[row_idx, "R_raw_int"]

    df_sim.loc[row_idx, "UG_FT"] = df_sim.loc[row_idx, "UG_FT_S"]
    df_sim.loc[row_idx, "UG_PT"] = df_sim.loc[row_idx, "UG_PT_S"]

    return df_sim

def plot_simulation(
    disc,
    df_osu_base,
    df_osu_sim,
    df_other,
    latest_peer_year=2024,
    latest_year=2024,
    show_simulation=True
):
    years = [2020, 2021, 2022, 2023, 2024]
    present_year = 2024

    # ------------------------------------------------------------------
    # Filter OSU baseline rows
    # ------------------------------------------------------------------
    df_disc_base = df_osu_base[
        (df_osu_base["Discipline Name"] == disc)
        & (df_osu_base["Year"].isin(years))
    ].copy()

    # ------------------------------------------------------------------
    # Filter OSU simulated rows
    # ------------------------------------------------------------------
    df_disc_sim = df_osu_sim[
        (df_osu_sim["Discipline Name"] == disc)
        & (df_osu_sim["Year"].isin(years))
    ].copy()

    if df_disc_base.empty:
        raise ValueError(f"No OSU rows found for {disc} in {years}.")

    # ------------------------------------------------------------------
    # Filter peer schools
    # ------------------------------------------------------------------
    df_peer = df_other[
        (df_other["Discipline Name"] == disc)
        & (df_other["Year"] == latest_peer_year)
    ].copy()

    # ------------------------------------------------------------------
    # Compatibility with old/new column names
    # ------------------------------------------------------------------
    star_col = "STAR" if "STAR" in df_disc_base.columns else "S" if "S" in df_disc_base.columns else "Rs_SCH"
    star_benchmark_col = (
        "STAR_AAU"
        if "STAR_AAU" in df_disc_base.columns
        else "S_AAU"
        if "S_AAU" in df_disc_base.columns
        else "Rs_AAU_SCH"
    )

    r_col = "R_raw" if "R_raw" in df_disc_base.columns else "R_s"
    r_benchmark_col = "R_AAU" if "R_AAU" in df_disc_base.columns else None

    required_cols = [
        star_col,
        star_benchmark_col,
        r_col,
        "D",
        "D_AAU"
    ]

    if r_benchmark_col is not None:
        required_cols.append(r_benchmark_col)

    for frame in [df_disc_base, df_disc_sim, df_peer]:
        for c in required_cols:
            if c in frame.columns:
                frame[c] = pd.to_numeric(frame[c], errors="coerce")
                frame[c] = frame[c].replace([np.inf, -np.inf], np.nan)

    df_disc_base = df_disc_base.dropna(subset=[star_col, r_col, "D"])
    df_disc_sim = df_disc_sim.dropna(subset=[star_col, r_col, "D"])

    if not df_peer.empty:
        df_peer = df_peer.dropna(subset=[star_col, r_col, "D"])

    if df_disc_base.empty:
        raise ValueError(f"No valid OSU rows found for {disc} after cleaning.")

    # ------------------------------------------------------------------
    # Helper to summarize OSU rows by year
    # ------------------------------------------------------------------
    def build_plot_df(df_disc):
        rows = []

        for yr in years:
            sub = df_disc[df_disc["Year"] == yr].copy()

            if sub.empty:
                continue

            STAR_yr = float(sub[star_col].median())
            R_yr = float(sub[r_col].median())
            D_yr = float(sub["D"].median())

            D_AAU_vals = sub["D_AAU"].dropna().unique()
            D_AAU_yr = (
                float(D_AAU_vals[0])
                if len(D_AAU_vals)
                else float(sub["D"].median())
            )

            STAR_AAU_vals = sub[star_benchmark_col].dropna().unique()
            STAR_AAU_yr = (
                float(STAR_AAU_vals[0])
                if len(STAR_AAU_vals)
                else float(sub[star_col].median())
            )

            if r_benchmark_col is not None and r_benchmark_col in sub.columns:
                R_AAU_vals = sub[r_benchmark_col].dropna().unique()
                R_AAU_yr = (
                    float(R_AAU_vals[0])
                    if len(R_AAU_vals)
                    else float(sub[r_col].median())
                )
            else:
                R_AAU_yr = float(sub[r_col].median())

            rows.append({
                "Year": yr,
                "STAR": STAR_yr,
                "S": STAR_yr,
                "R": R_yr,
                "D": D_yr,
                "D_AAU": D_AAU_yr,
                "STAR_AAU": STAR_AAU_yr,
                "S_AAU": STAR_AAU_yr,
                "R_AAU": R_AAU_yr
            })

        return pd.DataFrame(rows).sort_values("Year")

    plot_df = build_plot_df(df_disc_base)
    plot_df_sim = build_plot_df(df_disc_sim)

    if plot_df.empty:
        raise ValueError(f"No year-level OSU summaries could be created for {disc}.")

    # ------------------------------------------------------------------
    # Baseline and simulated latest rows
    # ------------------------------------------------------------------
    baseline_latest = plot_df[plot_df["Year"] == latest_year]
    simulated_latest = plot_df_sim[plot_df_sim["Year"] == latest_year]

    has_simulated_latest = (
        show_simulation
        and not baseline_latest.empty
        and not simulated_latest.empty
    )

    if has_simulated_latest:
        baseline_latest = baseline_latest.iloc[0]
        simulated_latest = simulated_latest.iloc[0]

    # ------------------------------------------------------------------
    # Shared D benchmarks
    # ------------------------------------------------------------------
    D_min = float(plot_df["D_AAU"].min())
    D_max = float(plot_df["D_AAU"].max())

    latest_row = plot_df.loc[plot_df["Year"].idxmax()]
    D_latest = float(latest_row["D_AAU"])

    r_present = plot_df[plot_df["Year"] == present_year]

    if r_present.empty:
        present_year = int(plot_df["Year"].max())
        r_present = plot_df[plot_df["Year"] == present_year]

    D_intervention_present = 0.5 * float(r_present["D_AAU"].iloc[0])

    # ------------------------------------------------------------------
    # Colors
    # ------------------------------------------------------------------
    cmap = mpl.colormaps.get_cmap("Oranges")
    norm = mcolors.Normalize(vmin=min(years), vmax=max(years))
    year_to_color = {yr: cmap(norm(yr)) for yr in years}

    s_color = "#1b9e77"
    r_color = "#377eb8"

    d_color = "#1f4e79"
    d_band_color = "#cfe8ff"

    intervention_color = "gray"
    simulation_color = "black"

    # ------------------------------------------------------------------
    # Create side-by-side figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(18, 8.5),
        sharey=True
    )

    # ------------------------------------------------------------------
    # Helper draw function
    # ------------------------------------------------------------------
    def draw_panel(
        ax,
        x_col,
        x_benchmark_col,
        x_label,
        x_benchmark_label,
        benchmark_color,
        panel_title,
        peer_x_col,
        latest_x_benchmark,
        latest_d_benchmark
    ):
        if not df_peer.empty and df_peer[peer_x_col].notna().any():
            x_q25 = float(np.nanpercentile(df_peer[peer_x_col], 25))
            x_q75 = float(np.nanpercentile(df_peer[peer_x_col], 75))
        else:
            x_q25 = float(np.nanpercentile(plot_df[x_col], 25))
            x_q75 = float(np.nanpercentile(plot_df[x_col], 75))

        if not df_peer.empty and df_peer["D"].notna().any():
            D_q25 = float(np.nanpercentile(df_peer["D"], 25))
            D_q75 = float(np.nanpercentile(df_peer["D"], 75))
        else:
            D_q25 = float(np.nanpercentile(plot_df["D"], 25))
            D_q75 = float(np.nanpercentile(plot_df["D"], 75))

        if not df_peer.empty:
            peer_x_max = df_peer[peer_x_col].max()
            peer_d_max = df_peer["D"].max()
        else:
            peer_x_max = 0
            peer_d_max = 0

        sim_x_max = 0
        sim_d_max = 0

        if has_simulated_latest:
            sim_x_max = float(simulated_latest[x_col])
            sim_d_max = float(simulated_latest["D"])

        x_data_max = max(
            plot_df[x_col].max(),
            peer_x_max,
            x_q75,
            sim_x_max
        )

        x_axis_max = x_data_max + max(1, x_data_max * 0.10)

        # Peer points
        if not df_peer.empty:
            ax.scatter(
                df_peer[peer_x_col],
                df_peer["D"],
                s=60,
                color="gray",
                alpha=0.22,
                edgecolor="none",
                zorder=2
            )

            if "Peer Short Name" in df_peer.columns:
                peer_labels = df_peer.dropna(subset=["Peer Short Name"])

                for i, (_, r) in enumerate(peer_labels.iterrows()):
                    label = str(r["Peer Short Name"]).strip()

                    if not label:
                        continue

                    x_offset = 4 if i % 2 == 0 else -4
                    ha = "left" if x_offset > 0 else "right"

                    ax.annotate(
                        label,
                        xy=(r[peer_x_col], r["D"]),
                        xytext=(x_offset, 3),
                        textcoords="offset points",
                        ha=ha,
                        va="bottom",
                        fontsize=7,
                        color="#4a4a4a",
                        alpha=0.85,
                        clip_on=True,
                        zorder=4
                    )

        # Baseline OSU points
        for _, r in plot_df.iterrows():
            yr = int(r["Year"])
            c = year_to_color[yr]

            ax.scatter(
                r[x_col],
                r["D"],
                s=140,
                color=c,
                edgecolor="black",
                linewidth=0.4,
                zorder=5
            )

        # Simulation arrow and point
        if has_simulated_latest:
            x0 = float(baseline_latest[x_col])
            y0 = float(baseline_latest["D"])
            x1 = float(simulated_latest[x_col])
            y1 = float(simulated_latest["D"])

            # Draw arrow only if there is actual movement
            if not (np.isclose(x0, x1) and np.isclose(y0, y1)):
                ax.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=simulation_color,
                        lw=2.2,
                        shrinkA=8,
                        shrinkB=8
                    ),
                    zorder=8
                )

            ax.scatter(
                x1,
                y1,
                s=230,
                facecolor="none",
                edgecolor=simulation_color,
                linewidth=2.5,
                zorder=9
            )

            ax.scatter(
                x1,
                y1,
                s=70,
                color=year_to_color[latest_year],
                edgecolor=simulation_color,
                linewidth=0.7,
                zorder=10
            )

            ax.annotate(
                f"Modified {latest_year}",
                xy=(x1, y1),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color=simulation_color,
                zorder=11
            )

        # Highlight the 2024 OSU STAR point with an overlay only on the STAR panel.
        if x_col == "STAR" and not baseline_latest.empty:
            x_star = float(baseline_latest[x_col])
            y_star = float(baseline_latest["D"])

            ax.scatter(
                x_star,
                y_star,
                s=340,
                facecolor="none",
                edgecolor="#d62728",
                linewidth=2,
                alpha=0.75,
                zorder=9
            )

            coord_label = f"({x_star:.0f}, {y_star:.1f})"
            ax.annotate(
                coord_label,
                xy=(x_star, y_star),
                xytext=(18, -20),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color="#732020",
                bbox=dict(boxstyle="round,pad=0.35", fc="#feeceb", ec="#732020", alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="#732020", lw=1.3),
                zorder=12
            )

        # Axis limits
        ax.set_xlim(0, x_axis_max)

        y_axis_max = max(
            5,
            plot_df["D"].max() + 0.5,
            peer_d_max + 0.5,
            D_max + 0.5,
            D_latest + 0.5,
            D_intervention_present + 0.5,
            sim_d_max + 0.5
        )

        ax.set_ylim(0, y_axis_max)

        if x_axis_max <= 30:
            ax.xaxis.set_major_locator(MultipleLocator(5))
        elif x_axis_max <= 100:
            ax.xaxis.set_major_locator(MultipleLocator(10))
        else:
            ax.xaxis.set_major_locator(MultipleLocator(100))

        ax.yaxis.set_major_locator(MultipleLocator(1))

        # Benchmark quartile ranges
        ax.axvspan(
            x_q25,
            x_q75,
            facecolor="green",
            alpha=0.15,
            zorder=1
        )

        ax.axhspan(
            D_q25,
            D_q75,
            facecolor=d_band_color,
            alpha=0.35,
            zorder=1
        )

        # Benchmark lines
        ax.axvline(
            latest_x_benchmark,
            linestyle="-",
            linewidth=2.2,
            color=benchmark_color,
            alpha=0.9,
            zorder=3
        )

        ax.axhline(
            latest_d_benchmark,
            linestyle="-",
            linewidth=2.2,
            color=d_color,
            alpha=0.9,
            zorder=3
        )

        # Intervention zone
        ax.axhline(
            D_intervention_present,
            linestyle="-",
            linewidth=2,
            color=intervention_color,
            alpha=0.8,
            zorder=2
        )

        ax.axhspan(
            0,
            D_intervention_present,
            facecolor=intervention_color,
            alpha=0.05,
            zorder=0
        )

        ax.axhspan(
            0,
            D_intervention_present,
            facecolor="none",
            edgecolor=intervention_color,
            hatch="///",
            linewidth=0.0,
            alpha=0.30,
            zorder=0
        )

        # Zone labels
        y_axis_max = ax.get_ylim()[1]

        ax.text(
            x_q75 + 0.5 * max(x_axis_max - x_q75, 0.1),
            D_max + 0.5 * max(y_axis_max - D_max, 0.2),
            "Zone I",
            fontsize=12,
            alpha=0.55,
            ha="center",
            va="center"
        )

        ax.text(
            0.5 * x_q25,
            D_max + 0.5 * max(y_axis_max - D_max, 0.2),
            "Zone II",
            fontsize=12,
            alpha=0.55,
            ha="center",
            va="center"
        )

        ax.text(
            0.5 * x_q25,
            D_intervention_present + 0.5 * max(D_max - D_intervention_present, 0.2),
            "Zone III",
            fontsize=12,
            alpha=0.55,
            ha="center",
            va="center"
        )

        ax.text(
            x_q75 + 0.5 * max(x_axis_max - x_q75, 0.1),
            D_intervention_present + 0.5 * max(D_max - D_intervention_present, 0.2),
            "Zone IV",
            fontsize=12,
            alpha=0.55,
            ha="center",
            va="center"
        )

        # Benchmark annotations
        x_axis_end = ax.get_xlim()[1]
        y_axis_top = ax.get_ylim()[1]

        ax.annotate(
            f"Q1 = {x_q25:.0f}",
            xy=(x_q25, y_axis_top),
            xytext=(-12, -12),
            textcoords="offset points",
            ha="right",
            va="top",
            rotation=90,
            fontsize=9,
            color=benchmark_color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
        )

        ax.annotate(
            f"Q2 = {latest_x_benchmark:.0f}",
            xy=(latest_x_benchmark, y_axis_top),
            xytext=(6, -12),
            textcoords="offset points",
            ha="left",
            va="top",
            rotation=90,
            fontsize=9,
            color=benchmark_color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
        )

        ax.annotate(
            f"Q3 = {x_q75:.0f}",
            xy=(x_q75, y_axis_top),
            xytext=(10, -12),
            textcoords="offset points",
            ha="left",
            va="top",
            rotation=90,
            fontsize=9,
            color=benchmark_color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
        )

        ax.annotate(
            f"Q1 = {D_q25:.1f}",
            xy=(x_axis_end, D_q25),
            xytext=(-10, 6),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=9,
            color=d_color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
        )

        ax.annotate(
            f"Q2 = {latest_d_benchmark:.1f}",
            xy=(x_axis_end, latest_d_benchmark),
            xytext=(-10, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=9,
            color=d_color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
        )

        ax.annotate(
            f"Q3 = {D_q75:.1f}",
            xy=(x_axis_end, D_q75),
            xytext=(-10, -6),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=9,
            color=d_color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
        )

        ax.set_xlabel(x_label, fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.set_title(panel_title, fontsize=13, fontweight="bold")

    # Left panel: R vs D
    draw_panel(
        ax=axes[0],
        x_col="R",
        x_benchmark_col="R_AAU",
        x_label="Raw Student to Faculty Ratio (R)",
        x_benchmark_label="R_AAU",
        benchmark_color=r_color,
        panel_title="R to D",
        peer_x_col=r_col,
        latest_x_benchmark=float(latest_row["R_AAU"]),
        latest_d_benchmark=float(latest_row["D_AAU"])
    )

    # Right panel: STAR vs D
    draw_panel(
        ax=axes[1],
        x_col="STAR",
        x_benchmark_col="STAR_AAU",
        x_label="SCH to Teaching Assignment Ratio (STAR)",
        x_benchmark_label="STAR_AAU",
        benchmark_color=r_color,
        panel_title="STAR to D",
        peer_x_col=star_col,
        latest_x_benchmark=float(latest_row[star_benchmark_col]),
        latest_d_benchmark=float(latest_row["D_AAU"])
    )

    axes[0].set_ylabel(
        "Full-Time Doctoral Student to TTT Faculty Ratio (D)",
        fontsize=13,
    )

    # Legend
    year_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=9,
            markeredgecolor="black",
            markeredgewidth=0.6,
            color=year_to_color[int(yr)],
            label=f"OSU {int(yr)}"
        )
        for yr in plot_df["Year"]
    ]

    peer_handle = plt.Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        markersize=8,
        markeredgecolor="none",
        color="gray",
        alpha=0.35,
        label=f"Peer institutions {latest_peer_year}"
    )

    sim_handle = plt.Line2D(
        [0],
        [0],
        marker="o",
        linestyle="-",
        markersize=10,
        markerfacecolor="none",
        markeredgecolor="black",
        color="black",
        linewidth=2,
        label=f"Modified OSU {latest_year}"
    )

    year_handles.append(peer_handle)

    if show_simulation:
        year_handles.append(sim_handle)

    fig.legend(
        handles=year_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=min(len(year_handles), 7),
        frameon=False,
        fontsize=11,
    )

    fig.suptitle(
        f"{disc}",
        fontsize=15,
        fontweight="bold",
        y=1.04
    )

    fig.subplots_adjust(
        top=0.86,
        bottom=0.12,
        left=0.07,
        right=0.98,
        wspace=0.15
    )

    return fig
