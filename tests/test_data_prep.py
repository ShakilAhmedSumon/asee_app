import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data_prep import prepapre_data


def _raw_row(school, year, doctoral_ft, ttt, junior_ft, senior_ft):
    return {
        "School Name": school,
        "Discipline Name": "Test Disc",
        "Year": year,
        "Doctoral Full Time Sum": doctoral_ft,
        "Doctoral Part Time Sum": 0,
        "Total Tenured/Tenure-Track Faculty": ttt,
        "Freshman Full Time Sum": 100,
        "Sophomore Full Time Sum": 80,
        "Junior Full Time Sum": junior_ft,
        "Senior Full Time Sum": senior_ft,
        "Freshman Part Time Sum": 10,
        "Sophomore Part Time Sum": 8,
        "Junior Part Time Sum": 6,
        "Senior Part Time Sum": 4,
        "Masters Full Time Sum": 30,
        "Masters Part Time Sum": 5,
        "Full Time Teaching Faculty (Non-Tenure)": 2,
        "FTE of Part Time Teaching Faculty (Non-Tenure)": 1,
    }


def test_2025_osu_uses_2024_peer_dependent_benchmarks():
    raw = pd.DataFrame(
        [
            _raw_row("Oregon State University", 2024, 20, 10, 60, 40),
            _raw_row("Peer University", 2024, 40, 10, 80, 70),
            _raw_row("Oregon State University", 2025, 30, 10, 90, 80),
        ]
    )

    prepared = prepapre_data(raw, ug_ft=0, phd_ft=0, ttt=0)
    osu_2024 = prepared[
        (prepared["School Name"] == "Oregon State University")
        & (prepared["Year"] == 2024)
    ].iloc[0]
    osu_2025 = prepared[
        (prepared["School Name"] == "Oregon State University")
        & (prepared["Year"] == 2025)
    ].iloc[0]

    assert np.isclose(osu_2025["D_AAU"], osu_2024["D_AAU"])
    assert np.isclose(osu_2025["R_AAU"], osu_2024["R_AAU"])
    assert np.isclose(osu_2025["STAR_AAU"], osu_2024["STAR_AAU"])
    assert np.isclose(osu_2025["alpha"], 0.375)
    assert not np.isclose(osu_2025["R_raw"], osu_2024["R_raw"])
    assert not np.isclose(osu_2025["STAR"], osu_2024["STAR"])
