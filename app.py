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
    page_title="OSU ASEE Scenario Simulator",
    layout="wide"
)

st.title("OSU ASEE Scenario Simulator")
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
st.sidebar.header("Benchmark filtering rules")

ug_ft_threshold = st.sidebar.number_input(
    "Minimum UG full-time headcount for peer schools",
    min_value=0,
    value=199,
    step=1
)

phd_ft_threshold = st.sidebar.number_input(
    "Minimum full-time PhD headcount for peer schools",
    min_value=0,
    value=10,
    step=1
)

ttt_threshold = st.sidebar.number_input(
    "Minimum TTT faculty for peer schools",
    min_value=0,
    value=10,
    step=1
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

disciplines = sorted(df_osu_base["Discipline Name"].dropna().unique())

if len(disciplines) == 0:
    st.error("No OSU disciplines found after preparing the data.")
    st.stop()

st.sidebar.header("Scenario")

selected_disc = st.sidebar.selectbox(
    "Select discipline",
    disciplines
)

latest_year = int(
    st.sidebar.selectbox(
        "Simulation year",
        sorted(df_osu_base["Year"].dropna().unique(), reverse=True),
        index=0
    )
)

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
st.sidebar.subheader("Doctoral student changes")

delta_phd_ft = st.sidebar.number_input(
    "Change in full-time PhD students",
    value=0,
    step=1
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
    delta_phd_ft=delta_phd_ft,
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
    st.error("Could not find baseline or simulated OSU row.")
    st.stop()

base_row = base_row.iloc[0]
sim_row = sim_row.iloc[0]

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
st.subheader("Baseline vs Simulated OSU Values")

comparison_cols = [
    "Doctoral Full Time Sum",
    "Total Tenured/Tenure-Track Faculty",
    "Full Time Teaching Faculty (Non-Tenure)",
    "FTE of Part Time Teaching Faculty (Non-Tenure)",
    "Freshman Full Time Sum",
    "Sophomore Full Time Sum",
    "Junior Full Time Sum",
    "Senior Full Time Sum",
    "Masters Full Time Sum",
    "Masters Part Time Sum",
    "D",
    "alpha",
    "R_raw",
    "STAR_num",
    "STAR_den",
    "STAR",
]

comparison = pd.DataFrame({
    "Metric": comparison_cols,
    "Baseline": [base_row.get(c, None) for c in comparison_cols],
    "Simulated": [sim_row.get(c, None) for c in comparison_cols],
})

comparison["Change"] = comparison["Simulated"] - comparison["Baseline"]

st.markdown(
    comparison.to_html(index=False),
    unsafe_allow_html=True
)
