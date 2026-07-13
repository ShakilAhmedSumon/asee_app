import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulator_engine import simulate_osu_latest_year


def _prepared_row():
    return pd.DataFrame(
        [
            {
                "School Name": "Oregon State University",
                "Discipline Name": "Test Disc",
                "Year": 2024,
                "Doctoral Full Time Sum": 20.0,
                "Total Tenured/Tenure-Track Faculty": 10.0,
                "Full Time Teaching Faculty (Non-Tenure)": 2.0,
                "FTE of Part Time Teaching Faculty (Non-Tenure)": 1.0,
                "Freshman Full Time Sum": 100.0,
                "Sophomore Full Time Sum": 80.0,
                "Junior Full Time Sum": 60.0,
                "Senior Full Time Sum": 40.0,
                "Freshman Part Time Sum": 10.0,
                "Sophomore Part Time Sum": 8.0,
                "Junior Part Time Sum": 6.0,
                "Senior Part Time Sum": 4.0,
                "Masters Full Time Sum": 30.0,
                "Masters Part Time Sum": 5.0,
                "D": 2.0,
                "D_AAU": 4.0,
                "alpha": 0.562,
                "STAR_AAU": 100.0,
                "S_AAU": 100.0,
                "R_AAU": 10.0,
                "Rs_AAU_SCH": 100.0,
            }
        ]
    )


def test_ttt_change_preserves_d_and_alpha_and_scales_doctoral_students():
    result = simulate_osu_latest_year(
        _prepared_row(),
        discipline="Test Disc",
        delta_ttt=5,
    ).iloc[0]

    assert np.isclose(result["D"], 2.0)
    assert np.isclose(result["alpha"], 0.562)
    assert np.isclose(result["Doctoral Full Time Sum"], 30.0)


def test_delta_d_changes_d_alpha_and_derived_doctoral_students():
    result = simulate_osu_latest_year(
        _prepared_row(),
        discipline="Test Disc",
        delta_ttt=5,
        delta_d=1.0,
    ).iloc[0]

    assert np.isclose(result["D"], 3.0)
    assert np.isclose(result["alpha"], 0.469)
    assert np.isclose(result["Doctoral Full Time Sum"], 45.0)
    assert np.isclose(result["UG_FT_R"], 280.0)
    assert np.isclose(result["UG_FT_S"], 100.0)
