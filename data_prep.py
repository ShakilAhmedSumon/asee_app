
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.table import Table

def prepapre_data(
    df,
    ug_ft,
    phd_ft,
    ttt,
    latest_osu_year=2025,
    peer_benchmark_year=2024,
):
    df = df.copy()
    df = df.fillna(0)

    excluded_disciplines = {
        "Architectural Engineering",
        "Engineering (General)",
    }
    df = df[~df["Discipline Name"].isin(excluded_disciplines)]

    # ------------------------------------------------------------------
    # 1. Clean columns needed for doctoral ratio D
    # ------------------------------------------------------------------
    cols_needed_for_D = [
        "Doctoral Full Time Sum",
        "Doctoral Part Time Sum",
        "Total Tenured/Tenure-Track Faculty"
    ]

    df = df[((df["School Name"] == "Oregon State University") | (df["Doctoral Full Time Sum"] > phd_ft))]
    df = df[((df["School Name"] == "Oregon State University") | (df["Total Tenured/Tenure-Track Faculty"] > ttt))]
    
    for col in cols_needed_for_D:
        df[col] = (
            df[col].astype(str)
              .str.replace(",", "", regex=False)
              .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------------------
    # 2. Compute doctoral student to faculty ratio D
    # D = full-time doctoral students / TTT faculty
    # ------------------------------------------------------------------
    den = df["Total Tenured/Tenure-Track Faculty"]
    num = df["Doctoral Full Time Sum"]

    df["D"] = (num / den).where(den > 0, np.nan)
    df["D_int"] = df["D"].round().astype("Int64")

    # ------------------------------------------------------------------
    # 3. Year-aware D_AAU benchmark
    # ------------------------------------------------------------------
    d_aau = (
        df.groupby(["Discipline Name", "Year"], as_index=False)["D"]
          .median()
          .rename(columns={"D": "D_AAU"})
    )

    df = df.merge(d_aau, on=["Discipline Name", "Year"], how="left")

    df["D"] = pd.to_numeric(df["D"], errors="coerce")
    df["D_AAU"] = pd.to_numeric(df["D_AAU"], errors="coerce")

    # OSU has 2025 observations but the peer dataset ends in 2024.
    # Treat the 2025 OSU point as the new actual point while carrying forward
    # the discipline's 2024 peer-dependent benchmark.
    d_benchmark_by_discipline = (
        d_aau[d_aau["Year"] == peer_benchmark_year]
        .set_index("Discipline Name")["D_AAU"]
    )
    latest_osu_mask = (
        (df["School Name"] == "Oregon State University")
        & (df["Year"] == latest_osu_year)
    )
    df.loc[latest_osu_mask, "D_AAU"] = (
        df.loc[latest_osu_mask, "Discipline Name"]
        .map(d_benchmark_by_discipline)
        .fillna(df.loc[latest_osu_mask, "D_AAU"])
    )

    df = df.fillna(0)

    # ------------------------------------------------------------------
    # 4. Compute alpha
    # alpha = 0.375 if D >= D_AAU
    # alpha = 0.75 - 0.375 * D / D_AAU if D < D_AAU
    # ------------------------------------------------------------------
    mask = (df["D_AAU"] > 0) & (df["D"] < df["D_AAU"])

    df["alpha"] = 0.375
    df.loc[mask, "alpha"] = 0.75 - 0.375 * (df.loc[mask, "D"] / df.loc[mask, "D_AAU"])

    df["alpha"] = df["alpha"].replace([np.inf, -np.inf], np.nan).fillna(0.375)
    df["alpha"] = df["alpha"].round(3)

    # ------------------------------------------------------------------
    # 5. Clean numeric columns used in student and faculty calculations
    # ------------------------------------------------------------------
    num_cols = [
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
        "Total Tenured/Tenure-Track Faculty",
        "Full Time Teaching Faculty (Non-Tenure)",
        "FTE of Part Time Teaching Faculty (Non-Tenure)",
        "alpha",
    ]

    for col in num_cols:
        df[col] = (
            df[col].astype(str)
                  .str.replace(",", "", regex=False)
                  .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # ------------------------------------------------------------------
    # 6. Build student terms
    #
    # R = traditional student-to-faculty view.
    #     Includes 1st-, 2nd-, 3rd-, and 4th-year undergraduates.
    #
    # STAR = SCH-to-teaching-assignment view.
    #     Excludes 1st- and 2nd-year undergraduates.
    # ------------------------------------------------------------------

    # For R: include all undergraduate years
    df["UG_FT_R"] = (
        df["Freshman Full Time Sum"]
        + df["Sophomore Full Time Sum"]
        + df["Junior Full Time Sum"]
        + df["Senior Full Time Sum"]
    )

    df["UG_PT_R"] = (
        df["Freshman Part Time Sum"]
        + df["Sophomore Part Time Sum"]
        + df["Junior Part Time Sum"]
        + df["Senior Part Time Sum"]
    )

    # For STAR: exclude 1st- and 2nd-year undergraduates
    df["UG_FT_S"] = (
        df["Junior Full Time Sum"]
        + df["Senior Full Time Sum"]
    )

    df["UG_PT_S"] = (
        df["Junior Part Time Sum"]
        + df["Senior Part Time Sum"]
    )

    df["MS_FT"] = df["Masters Full Time Sum"]
    df["MS_PT"] = df["Masters Part Time Sum"]

    # ------------------------------------------------------------------
    # 7. Faculty terms
    # ------------------------------------------------------------------
    df["TTT"] = df["Total Tenured/Tenure-Track Faculty"]

    df["NTT"] = (
        df["Full Time Teaching Faculty (Non-Tenure)"]
        + df["FTE of Part Time Teaching Faculty (Non-Tenure)"]
    )

    ## filter to remove low-enrolled programs
    df = df[((df["School Name"] == "Oregon State University") | (df["UG_FT_R"] > ug_ft))]
    
    # ------------------------------------------------------------------
    # 8. Raw student-to-faculty ratio R
    #
    # R = all UG students + MS students / TTT + NTT
    # ------------------------------------------------------------------
    df["R_raw_num"] = (
        df["UG_FT_R"]
        + df["UG_PT_R"]
        + df["MS_FT"]
        + df["MS_PT"]
    )

    df["R_raw_den"] = df["TTT"] + df["NTT"]

    df["R_raw"] = np.where(
        df["R_raw_den"] == 0,
        np.nan,
        df["R_raw_num"] / df["R_raw_den"]
    )

    df["R_raw"] = df["R_raw"].replace([np.inf, -np.inf], np.nan)
    df["R_raw"] = df["R_raw"].fillna(0).round(3)
    df["R_raw_int"] = df["R_raw"].round().astype("Int64")

    # ------------------------------------------------------------------
    # 9. SCH numerator for STAR
    #
    # STAR numerator = 40UG_FT_S + 20UG_PT_S + 32MS_FT + 21MS_PT
    # ------------------------------------------------------------------
    df["STAR_num"] = (
        df["UG_FT_S"] * 40
        + df["UG_PT_S"] * 20
        + df["MS_FT"] * 32
        + df["MS_PT"] * 21
    )
    df["SCH"] = df["STAR_num"]

    # ------------------------------------------------------------------
    # 10. Teaching assignment denominator for STAR
    #
    # STAR denominator = alpha * TTT + 0.9NTT
    # ------------------------------------------------------------------
    df["STAR_den"] = (df["alpha"] * df["TTT"]) + (0.9 * df["NTT"])

    df["STAR"] = np.where(
        df["STAR_den"] == 0,
        np.nan,
        df["STAR_num"] / df["STAR_den"]
    )

    df["STAR"] = df["STAR"].replace([np.inf, -np.inf], np.nan)
    df["STAR"] = df["STAR"].fillna(0).round(3)
    df["S_den"] = df["STAR_den"]
    df["S"] = df["STAR"]

    # ------------------------------------------------------------------
    # 11. Backward-compatible aliases
    # ------------------------------------------------------------------
    df["Rs_den"] = df["S_den"]
    df["Rs_SCH"] = df["S"]

    df["Rs_num"] = df["R_raw_num"]
    df["Rs_den_ww"] = df["R_raw_den"]
    df["R_s"] = df["R_raw"]
    df["R_s_int"] = df["R_raw_int"]

    # These aliases keep old filters from breaking, but they now refer to
    # the STAR-definition undergraduate counts.
    df["UG_FT"] = df["UG_FT_S"]
    df["UG_PT"] = df["UG_PT_S"]

    # ------------------------------------------------------------------
    # 12. Year-aware benchmark for raw R
    # ------------------------------------------------------------------
    r_aau = (
        df.groupby(["Discipline Name", "Year"], as_index=False)["R_raw"]
          .median()
          .rename(columns={"R_raw": "R_AAU"})
    )

    df = df.merge(
        r_aau,
        on=["Discipline Name", "Year"],
        how="left"
    )

    # ------------------------------------------------------------------
    # 13. Year-aware benchmark for STAR
    # ------------------------------------------------------------------
    star_aau = (
        df.groupby(["Discipline Name", "Year"], as_index=False)["STAR"]
          .median()
          .rename(columns={"STAR": "STAR_AAU"})
    )

    df = df.merge(
        star_aau,
        on=["Discipline Name", "Year"],
        how="left"
    )

    r_benchmark_by_discipline = (
        r_aau[r_aau["Year"] == peer_benchmark_year]
        .set_index("Discipline Name")["R_AAU"]
    )
    star_benchmark_by_discipline = (
        star_aau[star_aau["Year"] == peer_benchmark_year]
        .set_index("Discipline Name")["STAR_AAU"]
    )
    latest_osu_mask = (
        (df["School Name"] == "Oregon State University")
        & (df["Year"] == latest_osu_year)
    )
    df.loc[latest_osu_mask, "R_AAU"] = (
        df.loc[latest_osu_mask, "Discipline Name"]
        .map(r_benchmark_by_discipline)
        .fillna(df.loc[latest_osu_mask, "R_AAU"])
    )
    df.loc[latest_osu_mask, "STAR_AAU"] = (
        df.loc[latest_osu_mask, "Discipline Name"]
        .map(star_benchmark_by_discipline)
        .fillna(df.loc[latest_osu_mask, "STAR_AAU"])
    )

    # Backward-compatible benchmark alias
    df["S_AAU"] = df["STAR_AAU"]
    df["Rs_AAU_SCH"] = df["S_AAU"]

    # ------------------------------------------------------------------
    # 14. Optional filtering
    #
    # Keep OSU out of benchmark comparison set.
    # Apply size threshold only to non-OSU institutions.
    #
    # NOTE:
    # If you re-enable this, decide whether the threshold should be based on:
    #   UG_FT_R = traditional all-UG size
    #   UG_FT_S = S-definition junior/senior-only size
    # ------------------------------------------------------------------
    # if mode == 199:
    #     df = df[
    #         ((df["School Name"] == "Oregon State University") | (df["UG_FT_R"] > 199))
    #     ]
    # else:
    #     df = df[
    #         ((df["School Name"] == "Oregon State University") | (df["UG_FT_R"] > 49))
    #     ]

    return df
