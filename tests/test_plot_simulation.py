import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulator_engine import plot_simulation


def test_plot_simulation_uses_peer_quartiles_for_band_bounds():
    years = [2020, 2021, 2022, 2023, 2024]

    df_osu_base = pd.DataFrame(
        [
            {"School Name": "Oregon State University", "Discipline Name": "Test Disc", "Year": y, "R_raw": 2.0 + y * 0.1, "STAR": 4.0 + y * 0.1, "D": 1.2 + 0.1 * (y - 2020), "D_AAU": 1.5, "R_AAU": 10.0, "STAR_AAU": 10.0}
            for y in years
        ]
    )

    df_osu_sim = df_osu_base.copy()

    df_peer = pd.DataFrame(
        [
            {"School Name": "Peer A", "Discipline Name": "Test Disc", "Year": 2024, "R_raw": 5.0, "STAR": 8.0, "D": 1.0, "D_AAU": 2.0, "R_AAU": 20.0, "STAR_AAU": 20.0},
            {"School Name": "Peer B", "Discipline Name": "Test Disc", "Year": 2024, "R_raw": 10.0, "STAR": 12.0, "D": 2.0, "D_AAU": 2.0, "R_AAU": 30.0, "STAR_AAU": 30.0},
            {"School Name": "Peer C", "Discipline Name": "Test Disc", "Year": 2024, "R_raw": 20.0, "STAR": 20.0, "D": 3.0, "D_AAU": 2.0, "R_AAU": 40.0, "STAR_AAU": 40.0},
            {"School Name": "Peer D", "Discipline Name": "Test Disc", "Year": 2024, "R_raw": 30.0, "STAR": 32.0, "D": 4.0, "D_AAU": 2.0, "R_AAU": 50.0, "STAR_AAU": 50.0},
        ]
    )

    fig = plot_simulation(
        disc="Test Disc",
        df_osu_base=df_osu_base,
        df_osu_sim=df_osu_sim,
        df_other=df_peer,
        latest_peer_year=2024,
        latest_year=2024,
        show_simulation=False,
    )

    ax = fig.axes[0]
    v_patch = ax.patches[0]
    h_patch = ax.patches[1]

    x_q25 = float(np.nanpercentile(df_peer["R_raw"], 25))
    x_q75 = float(np.nanpercentile(df_peer["R_raw"], 75))
    d_q25 = float(np.nanpercentile(df_peer["D"], 25))
    d_q75 = float(np.nanpercentile(df_peer["D"], 75))

    assert np.isclose(v_patch.get_x(), x_q25)
    assert np.isclose(v_patch.get_x() + v_patch.get_width(), x_q75)
    assert np.isclose(h_patch.get_y(), d_q25)
    assert np.isclose(h_patch.get_y() + h_patch.get_height(), d_q75)
