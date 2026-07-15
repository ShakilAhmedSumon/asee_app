import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from simulator_engine import simulate_osu_latest_year, plot_simulation

# Import your existing data-prep function
# Rename it if possible from prepapre_data to prepare_data
from data_prep import prepapre_data


PROGRAM_THRESHOLDS = {
    # Larger programs
    "Civil/Environmental Engineering": {"ug_ft": 199, "phd_ft": 9, "ttt": 9},
    "Computer Science (inside engineering)": {"ug_ft": 199, "phd_ft": 9, "ttt": 9},
    "Electrical/Computer Engineering": {"ug_ft": 199, "phd_ft": 9, "ttt": 9},
    "Chemical Engineering": {"ug_ft": 199, "phd_ft": 9, "ttt": 9},
    "Mechanical Engineering": {"ug_ft": 199, "phd_ft": 9, "ttt": 9},
    "Biomedical Engineering": {"ug_ft": 199, "phd_ft": 9, "ttt": 9},
    "Industrial/Manufacturing/Systems Engineering": {"ug_ft": 199, "phd_ft": 9, "ttt": 9},

    # Smaller programs
    "Nuclear Engineering": {"ug_ft": 49, "phd_ft": 4, "ttt": 4},
    "Metallurgical and Matrls. Engineering": {"ug_ft": 49, "phd_ft": 4, "ttt": 4},
    "Biological Engr. and Agricultural Engr.": {"ug_ft": 49, "phd_ft": 4, "ttt": 4},
}

DEFAULT_PROGRAM_THRESHOLDS = {"ug_ft": 199, "phd_ft": 9, "ttt": 9}


def add_peer_short_names(df, short_names_path):
    short_names = pd.read_excel(short_names_path)
    short_names = short_names[["Short Name", "University", "IPEDS"]].copy()

    df = df.copy()

    df["IPEDS Unit ID"] = df["IPEDS Unit ID"].astype(str).str.strip()
    short_names["IPEDS"] = short_names["IPEDS"].astype(str).str.strip()

    df = df.merge(
        short_names[["IPEDS", "Short Name"]],
        left_on="IPEDS Unit ID",
        right_on="IPEDS",
        how="left"
    )

    missing_short_name = df["Short Name"].isna()

    if missing_short_name.any():
        short_names_by_school = short_names.rename(
            columns={"University": "School Name", "Short Name": "Short Name by School"}
        )

        df = df.merge(
            short_names_by_school[["School Name", "Short Name by School"]],
            on="School Name",
            how="left"
        )

        df["Short Name"] = df["Short Name"].fillna(df["Short Name by School"])
        df = df.drop(columns=["Short Name by School"])

    df["Peer Short Name"] = df["Short Name"]
    df = df.drop(columns=["IPEDS", "Short Name"])

    return df


st.set_page_config(
    page_title="D vs. STAR peer benchmarking",
    layout="wide"
)

st.title("D vs. STAR peer benchmarking")
st.markdown(
    """
    This simulator changes only the latest OSU point for the selected discipline.
    Peer schools, benchmark bands, and historical OSU points remain fixed.
    """
)

raw_data_path = 'data/raw_data_v3.xlsx'
short_names_path = 'data/AAU Publics 1.xlsx'

# -------------------------------------------------------------------
# Read data directly from local folder
# -------------------------------------------------------------------
raw_df = pd.read_excel(raw_data_path)

# st.caption(f"Loaded data from: `{DATA_PATH.name}`")
# -------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------
excluded_disciplines = {
    "Architectural Engineering",
    "Engineering (General)",
}
disciplines = sorted(
    set(
        raw_df.loc[
            raw_df["School Name"] == "Oregon State University",
            "Discipline Name",
        ].dropna()
    )
    - excluded_disciplines
)

if not disciplines:
    st.error("No OSU disciplines found in the source data.")
    st.stop()

st.sidebar.header("Scenario")

selected_disc = st.sidebar.selectbox(
    "Select discipline",
    disciplines,
)

simulation_years = sorted(
    pd.to_numeric(
        raw_df.loc[
            (raw_df["School Name"] == "Oregon State University")
            & (raw_df["Discipline Name"] == selected_disc),
            "Year",
        ],
        errors="coerce",
    ).dropna().astype(int).unique(),
    reverse=True,
)

latest_year = int(
    st.sidebar.selectbox(
        "Simulation year",
        simulation_years,
        index=0,
        key=f"simulation_year_{selected_disc}",
    )
)

st.sidebar.header("Benchmark filtering rules")

threshold_defaults = PROGRAM_THRESHOLDS.get(
    selected_disc,
    DEFAULT_PROGRAM_THRESHOLDS,
)

ug_ft_threshold = st.sidebar.number_input(
    "Minimum UG full-time headcount for peer schools",
    min_value=0,
    value=threshold_defaults["ug_ft"],
    step=1,
    key=f"ug_ft_threshold_{selected_disc}",
)

phd_ft_threshold = st.sidebar.number_input(
    "Minimum full-time PhD headcount for peer schools",
    min_value=0,
    value=threshold_defaults["phd_ft"],
    step=1,
    key=f"phd_ft_threshold_{selected_disc}",
)

ttt_threshold = st.sidebar.number_input(
    "Minimum TTT faculty for peer schools",
    min_value=0,
    value=threshold_defaults["ttt"],
    step=1,
    key=f"ttt_threshold_{selected_disc}",
)

# -------------------------------------------------------------------
# Prepare baseline data
# -------------------------------------------------------------------
df_prepared = prepapre_data(
    raw_df,
    ug_ft=ug_ft_threshold,
    phd_ft=phd_ft_threshold,
    ttt=ttt_threshold
)

df_prepared = add_peer_short_names(df_prepared, short_names_path)

df_osu_base = df_prepared[
    df_prepared["School Name"] == "Oregon State University"
].copy()

df_other = df_prepared[
    df_prepared["School Name"] != "Oregon State University"
].copy()

if selected_disc not in set(df_osu_base["Discipline Name"].dropna()):
    st.error(f"No prepared OSU data found for {selected_disc}.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Faculty changes")

delta_ttt = st.sidebar.number_input(
    "Change in TTT faculty",
    value=0,
    step=1
)

delta_ntt = st.sidebar.number_input(
    "Change in NTT faculty",
    value=0,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.subheader("Doctoral ratio change")

delta_d = st.sidebar.number_input(
    "Increase in D (FT PhD / TTT)",
    value=0.0,
    step=0.1,
    format="%.2f",
    help=(
        "Changes the full-time doctoral-student-to-TTT-faculty ratio directly. "
        "Changing TTT faculty alone keeps D and alpha fixed."
    ),
)

st.sidebar.markdown("---")
st.sidebar.subheader("Undergraduate enrollment changes")

with st.sidebar.expander("Full-time undergraduate changes"):
    delta_freshman_ft = st.number_input("Freshman FT change", value=0, step=1)
    delta_sophomore_ft = st.number_input("Sophomore FT change", value=0, step=1)
    delta_junior_ft = st.number_input("Junior FT change", value=0, step=1)
    delta_senior_ft = st.number_input("Senior FT change", value=0, step=1)

with st.sidebar.expander("Part-time undergraduate changes"):
    delta_freshman_pt = st.number_input("Freshman PT change", value=0, step=1)
    delta_sophomore_pt = st.number_input("Sophomore PT change", value=0, step=1)
    delta_junior_pt = st.number_input("Junior PT change", value=0, step=1)
    delta_senior_pt = st.number_input("Senior PT change", value=0, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Master's enrollment changes")

delta_ms_ft = st.sidebar.number_input(
    "Master's FT change",
    value=0,
    step=1
)

delta_ms_pt = st.sidebar.number_input(
    "Master's PT change",
    value=0,
    step=1
)

# -------------------------------------------------------------------
# Simulate
# -------------------------------------------------------------------
df_simulated = simulate_osu_latest_year(
    df_prepared=df_prepared,
    discipline=selected_disc,
    latest_year=latest_year,
    delta_ttt=delta_ttt,
    delta_ntt=delta_ntt,
    delta_d=delta_d,
    delta_freshman_ft=delta_freshman_ft,
    delta_sophomore_ft=delta_sophomore_ft,
    delta_junior_ft=delta_junior_ft,
    delta_senior_ft=delta_senior_ft,
    delta_freshman_pt=delta_freshman_pt,
    delta_sophomore_pt=delta_sophomore_pt,
    delta_junior_pt=delta_junior_pt,
    delta_senior_pt=delta_senior_pt,
    delta_ms_ft=delta_ms_ft,
    delta_ms_pt=delta_ms_pt,
)

df_osu_sim = df_simulated[
    df_simulated["School Name"] == "Oregon State University"
].copy()

# -------------------------------------------------------------------
# Metric comparison
# -------------------------------------------------------------------
base_row = df_osu_base[
    (df_osu_base["Discipline Name"] == selected_disc)
    & (df_osu_base["Year"] == latest_year)
].copy()

sim_row = df_osu_sim[
    (df_osu_sim["Discipline Name"] == selected_disc)
    & (df_osu_sim["Year"] == latest_year)
].copy()

if base_row.empty or sim_row.empty:
    st.error("Could not find baseline or modified OSU row.")
    st.stop()

base_row = base_row.iloc[0]
sim_row = sim_row.iloc[0]

st.subheader(f"{selected_disc} — Peer Percentiles")

peer_rows = df_other[
    (df_other["Discipline Name"] == selected_disc)
    & (df_other["Year"] == latest_year)
].copy()

peer_metric_cols = {
    "D": ("D", "D_AAU"),
    "R": ("R_raw", "R_AAU"),
    "STAR": (
        "STAR" if "STAR" in peer_rows.columns else "S",
        "STAR_AAU" if "STAR_AAU" in df_osu_base.columns else "S_AAU",
    ),
}

# Match plot_simulation exactly: Q1/Q3 come from peers in the selected peer
# year, while Q2 is the latest OSU-year AAU benchmark displayed by the plot.
plot_years = [2020, 2021, 2022, 2023, 2024]
osu_plot_rows = df_osu_base[
    (df_osu_base["Discipline Name"] == selected_disc)
    & (df_osu_base["Year"].isin(plot_years))
].copy()
plot_benchmark_row = osu_plot_rows.loc[osu_plot_rows["Year"].idxmax()]


def plot_quartile(metric_col, percentile):
    peer_values = pd.to_numeric(peer_rows[metric_col], errors="coerce").dropna()

    if not peer_values.empty:
        return peer_values.quantile(percentile / 100)

    fallback_values = (
        osu_plot_rows.groupby("Year")[metric_col]
        .median()
        .dropna()
    )
    return fallback_values.quantile(percentile / 100)


def format_up_to_four_decimals(value):
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}".rstrip("0").rstrip(".")

peer_percentiles = pd.DataFrame(
    {
        "Metric": list(peer_metric_cols),
        "25th percentile": [
            plot_quartile(metric_col, 25)
            for metric_col, _ in peer_metric_cols.values()
        ],
        "50th percentile": [
            pd.to_numeric(
                pd.Series([plot_benchmark_row[benchmark_col]]),
                errors="coerce",
            ).iloc[0]
            for _, benchmark_col in peer_metric_cols.values()
        ],
        "75th percentile": [
            plot_quartile(metric_col, 75)
            for metric_col, _ in peer_metric_cols.values()
        ],
    }
)

peer_percentiles_display = peer_percentiles.copy()
for column in ["25th percentile", "50th percentile", "75th percentile"]:
    peer_percentiles_display[column] = peer_percentiles_display[column].map(
        format_up_to_four_decimals
    )

st.dataframe(peer_percentiles_display, hide_index=True, use_container_width=True)

st.subheader(f"{selected_disc} — Scenario Summary")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "D",
    f"{sim_row['D']:.2f}",
    delta=f"{sim_row['D'] - base_row['D']:.2f}"
)

m2.metric(
    "R",
    f"{sim_row['R_raw']:.2f}",
    delta=f"{sim_row['R_raw'] - base_row['R_raw']:.2f}"
)

m3.metric(
    "STAR",
    f"{sim_row['STAR']:.2f}",
    delta=f"{sim_row['STAR'] - base_row['STAR']:.2f}"
)

m4.metric(
    "Alpha",
    f"{sim_row['alpha']:.3f}",
    delta=f"{sim_row['alpha'] - base_row['alpha']:.3f}"
)

# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------
fig = plot_simulation(
    disc=selected_disc,
    df_osu_base=df_osu_base,
    df_osu_sim=df_osu_sim,
    df_other=df_other,
    latest_peer_year=latest_year,
    latest_year=latest_year,
    show_simulation=True
)

st.pyplot(fig, clear_figure=True)

# -------------------------------------------------------------------
# Before/after table
# -------------------------------------------------------------------
st.subheader("Baseline vs Modified OSU Values")

comparison_metrics = [
    ("TTT Faculty", "Total Tenured/Tenure-Track Faculty"),
    ("FT NTT Faculty", "Full Time Teaching Faculty (Non-Tenure)"),
    ("FTE of PT NTT Faculty", "FTE of Part Time Teaching Faculty (Non-Tenure)"),
    ("Freshman FT", "Freshman Full Time Sum"),
    ("Sophomore FT", "Sophomore Full Time Sum"),
    ("Junior FT", "Junior Full Time Sum"),
    ("Senior FT", "Senior Full Time Sum"),
    ("FT Undergrads", "UG_FT_R"),
    ("FT Undergrads exc. 1st and 2nd year", "UG_FT_S"),
    ("Master's FT", "Masters Full Time Sum"),
    ("Master's PT", "Masters Part Time Sum"),
    ("Doctoral FT", "Doctoral Full Time Sum"),
    ("D", "D"),
    ("Alpha", "alpha"),
    ("R", "R_raw"),
    ("STAR", "STAR"),
]

comparison = pd.DataFrame({
    "Metric": [label for label, _ in comparison_metrics],
    "Baseline": [base_row.get(col, None) for _, col in comparison_metrics],
    "Modified": [sim_row.get(col, None) for _, col in comparison_metrics],
})

comparison["Change"] = comparison["Modified"] - comparison["Baseline"]

comparison_display = comparison.copy()
for column in ["Baseline", "Modified", "Change"]:
    comparison_display[column] = comparison_display[column].map(
        format_up_to_four_decimals
    )

st.markdown(
    comparison_display.to_html(index=False),
    unsafe_allow_html=True
)
