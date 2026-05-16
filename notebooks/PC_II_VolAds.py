# /// script
# [tool.marimo.opengraph]
# title = "PC II: Volumetrische Adsorption"
# description = "Interaktive Auswertung der volumetrischen Adsorption mit Kalibration, Isothermen, Isosteren, Adsorptionsenthalpie und Modellvergleich."
# ///


import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium", app_title="PC II Volumetrische Adsorption")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import math
    from io import StringIO

    R = 8.314462618  # J mol^-1 K^-1

    mbar_to_Pa = 100.0
    mL_to_m3 = 1e-6

    mo.md(
        r"""
        # Volumetrische Adsorption

        Dieses Notebook wertet den Versuch **volumetrische Adsorption** aus.

        Die Auswertung umfasst:

        1. Offset-Korrektur der gemessenen Drücke  
        2. Kalibration der Volumina \(V_M\) und \(V_P\)  
        3. Berechnung der adsorbierten Stoffmenge \(n_\mathrm{ads}\)  
        4. Darstellung der Adsorptionsisothermen  
        5. Konstruktion von Adsorptionsisosteren  
        6. Bestimmung der isosteren Adsorptionsenthalpie  
        7. Vergleich von Langmuir-, Freundlich- und Temkin-Modell  

        Die Berechnungen sind so aufgebaut, dass alle Zwischenschritte nachvollziehbar bleiben.
        """
    )
    return R, StringIO, curve_fit, mL_to_m3, math, mbar_to_Pa, mo, np, pd, plt


@app.cell
def _(StringIO, math, np, pd):
    def parse_csv_text(text: str, required_cols: list[str], name: str) -> pd.DataFrame:
        text = text.strip()
        if not text:
            raise ValueError(f"Keine Daten für {name} eingegeben.")

        df = pd.read_csv(StringIO(text))

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"{name}: Es fehlen Spalten: {missing}. "
                f"Gefordert sind: {required_cols}. Gefunden wurden: {list(df.columns)}"
            )

        return df.copy()


    def p_corr(p_mbar: float, offset_mbar: float) -> float:
        """
        Offset-Korrektur.

        p_korr = p_roh - offset

        Beispiel:
        offset = -0.4 mbar
        p_korr = p_roh - (-0.4) = p_roh + 0.4
        """
        return float(p_mbar) - float(offset_mbar)


    def stats_series(x: pd.Series) -> dict:
        x = x.dropna().astype(float)
        n = int(x.shape[0])
        mean = float(x.mean())
        std = float(x.std(ddof=1)) if n > 1 else np.nan
        sem = std / np.sqrt(n) if n > 1 else np.nan
        return {
            "n": n,
            "mean": mean,
            "std": std,
            "sem": sem,
            "min": float(x.min()),
            "max": float(x.max()),
        }


    def r_squared(y_true, y_fit) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_fit = np.asarray(y_fit, dtype=float)

        ss_res = np.sum((y_true - y_fit) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if np.isclose(ss_tot, 0.0):
            return np.nan

        return float(1.0 - ss_res / ss_tot)


    def _round_to_sig(x: float, sig: int) -> float:
        if x == 0 or not np.isfinite(x):
            return x
        return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


    def format_value_uncert(value: float, uncert: float, sig_uncert: int = 2) -> str:
        if not np.isfinite(value) or not np.isfinite(uncert):
            return f"{value} ± {uncert}"

        if uncert < 0:
            raise ValueError("uncert must be >= 0")

        if uncert == 0:
            return f"{value} ± 0"

        u = _round_to_sig(float(uncert), sig_uncert)

        exp = math.floor(math.log10(abs(u)))
        decimals = max(0, -(exp) + (sig_uncert - 1))

        v = round(float(value), decimals)
        u = round(float(u), decimals)

        fmt = f"{{:.{decimals}f}}"
        return f"{fmt.format(v)} ± {fmt.format(u)}"

    return format_value_uncert, p_corr, parse_csv_text, r_squared, stats_series


@app.cell
def _(mo):
    VE_mL_input = mo.ui.number(
        value=47.6,
        start=0.0,
        step=0.1,
        label="Eichvolumen VE / mL",
    )

    offset_input = mo.ui.number(
        value=-0.4,
        step=0.1,
        label="Druckoffset / mbar",
    )

    m_ads_g_input = mo.ui.number(
        value=0.1175,
        start=0.0,
        step=0.0001,
        label="Masse des trockenen Adsorbens / g",
    )

    T_room_K_input = mo.ui.number(
        value=293.15,
        start=250.0,
        step=0.1,
        label="Raumtemperatur für PV = nRT / K",
    )

    adsorbens_input = mo.ui.text(
        value="",
        label="Adsorbens, falls bekannt",
    )

    adsorptiv_input = mo.ui.text(
        value="",
        label="Probengas / Adsorptiv, falls bekannt",
    )

    T_calibration_C_input = mo.ui.number(
        value=90.0,
        step=0.1,
        label="Temperatur der Kalibration / °C",
    )

    T1_C_input = mo.ui.number(value=90.0, step=0.1, label="Temperatur 1 / °C")
    T2_C_input = mo.ui.number(value=105.0, step=0.1, label="Temperatur 2 / °C")
    T3_C_input = mo.ui.number(value=120.0, step=0.1, label="Temperatur 3 / °C")

    mo.vstack(
        [
            mo.md("## 1. Allgemeine Versuchsdaten"),
            mo.md(
                r"""
                Die volumetrische Stoffmengenbilanz wird mit der Raumtemperatur berechnet, da die Drücke und
                Volumina der Gasphase über die ideale Gasgleichung ausgewertet werden.
                """
            ),
            VE_mL_input,
            offset_input,
            m_ads_g_input,
            T_room_K_input,
            adsorbens_input,
            adsorptiv_input,
            T_calibration_C_input,
        ]
    )
    return (
        T1_C_input,
        T2_C_input,
        T3_C_input,
        T_calibration_C_input,
        T_room_K_input,
        VE_mL_input,
        adsorbens_input,
        adsorptiv_input,
        m_ads_g_input,
        offset_input,
    )


@app.cell
def _(T1_C_input, T2_C_input, T3_C_input, mo):


    T_info = {
        "T1": {"label": f"{T1_C_input.value:g} °C", "T_C": float(T1_C_input.value), "T_K": float(T1_C_input.value) + 273.15},
        "T2": {"label": f"{T2_C_input.value:g} °C", "T_C": float(T2_C_input.value), "T_K": float(T2_C_input.value) + 273.15},
        "T3": {"label": f"{T3_C_input.value:g} °C", "T_C": float(T3_C_input.value), "T_K": float(T3_C_input.value) + 273.15},
    }

    mo.vstack(
        [
            mo.md("## 2. Messtemperaturen"),
            T1_C_input,
            T2_C_input,
            T3_C_input,
        ]
    )
    return (T_info,)


@app.cell
def _(mo):
    calibration_template = (
        "nr,pE_mbar,pM_mbar,pMP_mbar\n"
        "1,596.3,445.6,286.8\n"
        "2,445.6,333.3,214.4\n"
        "3,333.3,249.3,160.2\n"
        "4,249.3,186.5,119.9\n"
        "5,186.5,139.6,89.7\n"
        "6,139.6,104.5,67.2"
    )

    isotherm_template = (
        "step,pM_mbar,pG_mbar\n"
        "1,39.6,6.6\n"
        "2,87.4,35.6\n"
        "3,139.1,79.3\n"
        "4,456.0,262.0\n"
        "5,578.2,427.5\n"
        "6,694.7,569.7\n"
        "7,838.2,718.9\n"
        "8,994.7,872.8"
    )

    calibration_code = "```csv\n" + calibration_template + "\n```"
    isotherm_code = "```csv\n" + isotherm_template + "\n```"

    mo.vstack(
        [
            mo.md("## 3. Datenformat"),
            mo.md("**Kalibrationsdaten** müssen dieses Format haben:"),
            mo.md(calibration_code),
            mo.md("**Isothermendaten** müssen dieses Format haben:"),
            mo.md(isotherm_code),
        ]
    )
    return calibration_template, isotherm_template


@app.cell
def _(
    T_calibration_C_input,
    T_info,
    calibration_template,
    isotherm_template,
    mo,
):
    calibration_text = mo.ui.text_area(
        value=calibration_template,
        label="Kalibrationsdaten als CSV",
        rows=8,
    )

    iso_T1_text = mo.ui.text_area(
        value=isotherm_template,
        label=f"Isotherme bei {T_info['T1']['label']} als CSV",
        rows=10,
    )

    iso_T2_text = mo.ui.text_area(
        value=isotherm_template,
        label=f"Isotherme bei {T_info['T2']['label']} als CSV",
        rows=10,
    )

    iso_T3_text = mo.ui.text_area(
        value=isotherm_template,
        label=f"Isotherme bei {T_info['T3']['label']} als CSV",
        rows=10,
    )

    T_calibration_C = float(T_calibration_C_input.value)
    T_calibration_K = T_calibration_C + 273.15

    mo.vstack(
        [
            mo.md("## 4. Rohdaten eingeben"),
            mo.md(
                "Die Daten können direkt aus einer CSV-Datei kopiert und hier eingefügt werden. "
                "Wichtig sind die korrekten Spaltennamen."
            ),
            calibration_text,
            iso_T1_text,
            iso_T2_text,
            iso_T3_text,
        ]
    )
    return calibration_text, iso_T1_text, iso_T2_text, iso_T3_text


@app.cell
def _(
    T_info,
    calibration_text,
    iso_T1_text,
    iso_T2_text,
    iso_T3_text,
    mo,
    parse_csv_text,
):
    cal_raw = parse_csv_text(
        calibration_text.value,
        required_cols=["nr", "pE_mbar", "pM_mbar", "pMP_mbar"],
        name="Kalibration",
    )

    iso_raw = {
        "T1": parse_csv_text(
            iso_T1_text.value,
            required_cols=["step", "pM_mbar", "pG_mbar"],
            name=f"Isotherme {T_info['T1']['label']}",
        ),
        "T2": parse_csv_text(
            iso_T2_text.value,
            required_cols=["step", "pM_mbar", "pG_mbar"],
            name=f"Isotherme {T_info['T2']['label']}",
        ),
        "T3": parse_csv_text(
            iso_T3_text.value,
            required_cols=["step", "pM_mbar", "pG_mbar"],
            name=f"Isotherme {T_info['T3']['label']}",
        ),
    }

    mo.vstack(
        [
            mo.md("## 5. Eingelesene Rohdaten"),
            mo.md("### Kalibration"),
            mo.ui.table(cal_raw),
            mo.md("### Isotherme T1"),
            mo.ui.table(iso_raw["T1"]),
            mo.md("### Isotherme T2"),
            mo.ui.table(iso_raw["T2"]),
            mo.md("### Isotherme T3"),
            mo.ui.table(iso_raw["T3"]),
        ]
    )
    return cal_raw, iso_raw


@app.cell
def _(cal_raw, iso_raw, mo, offset_input, p_corr):
    offset_mbar = float(offset_input.value)

    cal_corr = cal_raw.copy()
    for col in ["pE_mbar", "pM_mbar", "pMP_mbar"]:
        cal_corr[col.replace("_mbar", "_corr_mbar")] = cal_corr[col].apply(lambda p: p_corr(p, offset_mbar))

    iso_corr = {}
    for key, df_raw in iso_raw.items():
        df = df_raw.copy()
        for col in ["pM_mbar", "pG_mbar"]:
            df[col.replace("_mbar", "_corr_mbar")] = df[col].apply(lambda p: p_corr(p, offset_mbar))
        iso_corr[key] = df

    mo.vstack(
        [
            mo.md("## 6. Offset-Korrektur"),
            mo.md(
                rf"""
                Für alle Drücke wird gerechnet:

                \[
                p_\mathrm{{korr}} = p_\mathrm{{roh}} - p_\mathrm{{offset}}
                \]

                Eingesetzter Offset:

                \[
                p_\mathrm{{offset}} = {offset_mbar:g}\,\mathrm{{mbar}}
                \]
                """
            ),
            mo.md("### Kalibration nach Offset-Korrektur"),
            mo.ui.table(cal_corr),
        ]
    )
    return cal_corr, iso_corr


@app.cell
def _(VE_mL_input, cal_corr, mo, pd, stats_series):
    VE_mL = float(VE_mL_input.value)

    cal_calc = cal_corr.copy()

    cal_calc["VM_mL"] = VE_mL * (
        cal_calc["pE_corr_mbar"] / cal_calc["pM_corr_mbar"] - 1.0
    )

    cal_calc["VP_mL"] = cal_calc["VM_mL"] * (
        cal_calc["pM_corr_mbar"] / cal_calc["pMP_corr_mbar"] - 1.0
    )

    vm_stats = stats_series(cal_calc["VM_mL"])
    vp_stats = stats_series(cal_calc["VP_mL"])

    cal_summary = pd.DataFrame(
        [
            {"quantity": "VM_mL", **vm_stats},
            {"quantity": "VP_mL", **vp_stats},
        ]
    )

    VM_mean_mL = float(cal_summary.loc[cal_summary["quantity"] == "VM_mL", "mean"].iloc[0])
    VP_mean_mL = float(cal_summary.loc[cal_summary["quantity"] == "VP_mL", "mean"].iloc[0])

    VM_sem_mL = float(cal_summary.loc[cal_summary["quantity"] == "VM_mL", "sem"].iloc[0])
    VP_sem_mL = float(cal_summary.loc[cal_summary["quantity"] == "VP_mL", "sem"].iloc[0])

    mo.vstack(
        [
            mo.md(
                r"""
                ## 7. Kalibration von \(V_M\) und \(V_P\)

                Aus der Expansion des Heliums aus dem Eichvolumen \(V_E\) ergeben sich:

                \[
                V_M = V_E\left(\frac{p_E}{p_M}-1\right)
                \]

                \[
                V_P = V_M\left(\frac{p_M}{p_{MP}}-1\right)
                \]

                Die Mittelwerte werden anschließend für die Stoffmengenbilanz verwendet.
                """
            ),
            mo.ui.table(
                cal_calc[
                    [
                        "nr",
                        "pE_mbar",
                        "pM_mbar",
                        "pMP_mbar",
                        "pE_corr_mbar",
                        "pM_corr_mbar",
                        "pMP_corr_mbar",
                        "VM_mL",
                        "VP_mL",
                    ]
                ]
            ),
            mo.md("### Statistik"),
            mo.ui.table(cal_summary),
            mo.md(
                rf"""
                Verwendete Mittelwerte:

                \[
                V_M = {VM_mean_mL:.3f}\,\mathrm{{mL}}
                \]

                \[
                V_P = {VP_mean_mL:.3f}\,\mathrm{{mL}}
                \]
                """
            ),
        ]
    )
    return VM_mean_mL, VP_mean_mL, cal_calc, cal_summary


@app.cell
def _(VM_mean_mL, VP_mean_mL, cal_calc, mo, plt):
    fig_VM, ax_VM = plt.subplots(figsize=(7, 4))
    ax_VM.plot(cal_calc["nr"], cal_calc["VM_mL"], marker="o")
    ax_VM.axhline(VM_mean_mL, linestyle="--", label=f"Mittelwert = {VM_mean_mL:.3f} mL")
    ax_VM.set_xlabel("Messung Nr.")
    ax_VM.set_ylabel(r"$V_M$ / mL")
    ax_VM.set_title(r"Kalibration: Einzelwerte von $V_M$")
    ax_VM.grid(True)
    ax_VM.legend()
    fig_VM.tight_layout()

    fig_VP, ax_VP = plt.subplots(figsize=(7, 4))
    ax_VP.plot(cal_calc["nr"], cal_calc["VP_mL"], marker="o")
    ax_VP.axhline(VP_mean_mL, linestyle="--", label=f"Mittelwert = {VP_mean_mL:.3f} mL")
    ax_VP.set_xlabel("Messung Nr.")
    ax_VP.set_ylabel(r"$V_P$ / mL")
    ax_VP.set_title(r"Kalibration: Einzelwerte von $V_P$")
    ax_VP.grid(True)
    ax_VP.legend()
    fig_VP.tight_layout()

    mo.vstack(
        [
            mo.md("### Kalibrationsplots"),
            fig_VM,
            fig_VP,
        ]
    )
    return


@app.cell
def _(R, mL_to_m3, mbar_to_Pa, pd):
    def build_isotherm_calc(
        df_corr: pd.DataFrame,
        VM_mL: float,
        VP_mL: float,
        m_ads_g: float,
        T_room_K: float,
    ) -> pd.DataFrame:
        df = df_corr.copy()

        VM_m3 = VM_mL * mL_to_m3
        VP_m3 = VP_mL * mL_to_m3
        Vtot_m3 = VM_m3 + VP_m3

        m_ads_kg = m_ads_g / 1000.0

        df["pM_corr_Pa"] = df["pM_corr_mbar"] * mbar_to_Pa
        df["pG_corr_Pa"] = df["pG_corr_mbar"] * mbar_to_Pa

        df["pG_prev_corr_Pa"] = df["pG_corr_Pa"].shift(1).fillna(0.0)

        df["dnM_mol"] = (
            (df["pM_corr_Pa"] - df["pG_prev_corr_Pa"]) * VM_m3 / (R * T_room_K)
        )

        df["nGes_mol"] = df["dnM_mol"].cumsum()

        df["nG_mol"] = df["pG_corr_Pa"] * Vtot_m3 / (R * T_room_K)

        df["nads_mol"] = df["nGes_mol"] - df["nG_mol"]

        df["nads_per_kg_molkg"] = df["nads_mol"] / m_ads_kg
        df["nads_per_g_mmolg"] = df["nads_per_kg_molkg"]

        return df

    return (build_isotherm_calc,)


@app.cell
def _(
    T_info,
    T_room_K_input,
    VM_mean_mL,
    VP_mean_mL,
    build_isotherm_calc,
    iso_corr,
    m_ads_g_input,
    mo,
    pd,
):
    def make_iso_calc_and_checks(
        iso_corr_in,
        T_info_in,
        VM_mean_mL_in,
        VP_mean_mL_in,
        m_ads_g_in,
        T_room_K_in,
    ):
        calculated = {
            temp_key: build_isotherm_calc(
                df_corr=iso_corr_in[temp_key],
                VM_mL=VM_mean_mL_in,
                VP_mL=VP_mean_mL_in,
                m_ads_g=m_ads_g_in,
                T_room_K=T_room_K_in,
            )
            for temp_key in ["T1", "T2", "T3"]
        }

        check_rows = []

        for temp_key in ["T1", "T2", "T3"]:
            current = calculated[temp_key]
            check_rows.append(
                {
                    "Temperatur": T_info_in[temp_key]["label"],
                    "negative n_ads Punkte": int((current["nads_mol"] < 0).sum()),
                    "n_ads monoton nicht-abnehmend": bool(
                        (current["nads_mol"].diff().dropna() >= 0).all()
                    ),
                    "pG_max_mbar": float(current["pG_corr_mbar"].max()),
                    "nads_max_molkg": float(current["nads_per_kg_molkg"].max()),
                }
            )

        return calculated, pd.DataFrame(check_rows)


    m_ads_g = float(m_ads_g_input.value)
    T_room_K = float(T_room_K_input.value)

    iso_calc, checks_df = make_iso_calc_and_checks(
        iso_corr_in=iso_corr,
        T_info_in=T_info,
        VM_mean_mL_in=VM_mean_mL,
        VP_mean_mL_in=VP_mean_mL,
        m_ads_g_in=m_ads_g,
        T_room_K_in=T_room_K,
    )

    balance_cols = [
        "step",
        "pM_mbar",
        "pG_mbar",
        "pM_corr_mbar",
        "pG_corr_mbar",
        "pG_prev_corr_Pa",
        "dnM_mol",
        "nGes_mol",
        "nG_mol",
        "nads_mol",
        "nads_per_kg_molkg",
    ]

    mo.vstack(
        [
            mo.md(
                r"""
                ## 8. Stoffmengenbilanz

                Für jeden Dosierschritt gilt:

                \[
                \Delta n_M^i =
                \frac{(p_M^i-p_G^{i-1})V_M}{RT}
                \]

                \[
                n_\mathrm{gesamt}^i = \sum_{j=1}^{i}\Delta n_M^j
                \]

                \[
                n_G^i =
                \frac{p_G^i(V_M+V_P)}{RT}
                \]

                \[
                n_\mathrm{ads}^i =
                n_\mathrm{gesamt}^i - n_G^i
                \]
                """
            ),
            mo.md("### Plausibilitätschecks"),
            mo.ui.table(checks_df),
            mo.md(f"### Stoffmengenbilanz {T_info['T1']['label']}"),
            mo.ui.table(iso_calc["T1"][balance_cols]),
            mo.md(f"### Stoffmengenbilanz {T_info['T2']['label']}"),
            mo.ui.table(iso_calc["T2"][balance_cols]),
            mo.md(f"### Stoffmengenbilanz {T_info['T3']['label']}"),
            mo.ui.table(iso_calc["T3"][balance_cols]),
        ]
    )
    return checks_df, iso_calc


@app.cell
def _(T_info, iso_calc, mo, plt):
    def make_isotherm_plot(iso_calc_in, T_info_in):
        fig_iso, ax_iso = plt.subplots(figsize=(7, 4.5))

        for temp_key in ["T1", "T2", "T3"]:
            current_iso = iso_calc_in[temp_key]
            ax_iso.plot(
                current_iso["pG_corr_mbar"],
                current_iso["nads_per_kg_molkg"],
                marker="o",
                linestyle="None",
                label=T_info_in[temp_key]["label"],
            )

        ax_iso.set_xlabel(r"$p_G$ / mbar")
        ax_iso.set_ylabel(r"$n_{\mathrm{ads},s}$ / mol kg$^{-1}$")
        ax_iso.set_title("Adsorptionsisothermen")
        ax_iso.grid(True, which="major")
        ax_iso.minorticks_on()
        ax_iso.grid(True, which="minor", alpha=0.3)
        ax_iso.legend()
        fig_iso.tight_layout()

        return fig_iso


    isotherm_figure = make_isotherm_plot(
        iso_calc_in=iso_calc,
        T_info_in=T_info,
    )

    mo.vstack(
        [
            mo.md("## 9. Adsorptionsisothermen"),
            mo.md(
                r"""
                Eine Adsorptionsisotherme beschreibt die adsorbierte Stoffmenge als Funktion des
                Gleichgewichtsdruckes bei konstanter Temperatur:

                \[
                n_{\mathrm{ads},s} = f(p)_T
                \]

                Für die Auswertung wird gegen den Gleichgewichtsdruck \(p_G\) aufgetragen,
                nicht gegen den Dosierdruck \(p_M\).
                """
            ),
            isotherm_figure,
        ]
    )
    return


@app.cell
def _(T_info, iso_calc, mo, np, pd):
    def get_coverage_range(isotherm_calc):
        coverage_values = isotherm_calc["nads_per_kg_molkg"].to_numpy(dtype=float)
        return float(np.min(coverage_values)), float(np.max(coverage_values))


    def make_coverage_ranges(iso_calc_in, T_info_in):
        ranges_local = {
            temp_key: get_coverage_range(iso_calc_in[temp_key])
            for temp_key in ["T1", "T2", "T3"]
        }

        common_min_local = max(local_range[0] for local_range in ranges_local.values())
        common_max_local = min(local_range[1] for local_range in ranges_local.values())

        ranges_table_local = pd.DataFrame(
            [
                {
                    "Temperatur": T_info_in[temp_key]["label"],
                    "min_molkg": ranges_local[temp_key][0],
                    "max_molkg": ranges_local[temp_key][1],
                }
                for temp_key in ["T1", "T2", "T3"]
            ]
        )

        return ranges_local, common_min_local, common_max_local, ranges_table_local


    coverage_ranges, common_coverage_min, common_coverage_max, coverage_ranges_df = make_coverage_ranges(
        iso_calc_in=iso_calc,
        T_info_in=T_info,
    )

    default_isostere_coverages = list(
        np.linspace(common_coverage_min, common_coverage_max, 5)[1:-1]
    )

    cov1_input = mo.ui.number(
        value=float(default_isostere_coverages[0]),
        step=0.01,
        label="Isostere 1: Bedeckung / mol kg⁻¹",
    )

    cov2_input = mo.ui.number(
        value=float(default_isostere_coverages[1]),
        step=0.01,
        label="Isostere 2: Bedeckung / mol kg⁻¹",
    )

    cov3_input = mo.ui.number(
        value=float(default_isostere_coverages[2]),
        step=0.01,
        label="Isostere 3: Bedeckung / mol kg⁻¹",
    )

    mo.vstack(
        [
            mo.md("## 10. Auswahl der Bedeckungen für Isosteren"),
            mo.md(
                rf"""
                Für die Isosteren müssen Bedeckungen gewählt werden, die in allen drei
                Isothermen vorkommen.

                Gemeinsamer Bereich:

                \[
                {common_coverage_min:.4g}
                \le n_{{\mathrm{{ads}},s}}
                \le {common_coverage_max:.4g}
                \quad \mathrm{{mol\,kg^{{-1}}}}
                \]
                """
            ),
            mo.ui.table(coverage_ranges_df),
            cov1_input,
            cov2_input,
            cov3_input,
        ]
    )
    return cov1_input, cov2_input, cov3_input


@app.cell
def _(cov1_input, cov2_input, cov3_input, mo, pd):
    selected_coverages = [
        float(cov1_input.value),
        float(cov2_input.value),
        float(cov3_input.value),
    ]

    selected_coverages_df = pd.DataFrame(
        {
            "Isostere": ["1", "2", "3"],
            "Bedeckung / mol kg^-1": selected_coverages,
        }
    )

    mo.vstack(
        [
            mo.md("### Verwendete Bedeckungen"),
            mo.ui.table(selected_coverages_df),
        ]
    )
    return (selected_coverages,)


@app.cell
def _(T_info, iso_calc, mbar_to_Pa, mo, np, pd, plt, selected_coverages):
    def interpolate_pressure_at_coverage(isotherm_calc, target_coverage):
        interpolation_data = isotherm_calc[
            ["step", "pG_corr_mbar", "nads_per_kg_molkg"]
        ].copy()

        interpolation_data = interpolation_data.sort_values(
            "nads_per_kg_molkg"
        ).reset_index(drop=True)

        coverage_array = interpolation_data["nads_per_kg_molkg"].to_numpy(dtype=float)
        pressure_array = interpolation_data["pG_corr_mbar"].to_numpy(dtype=float)

        coverage_min_local = float(coverage_array.min())
        coverage_max_local = float(coverage_array.max())

        if not (coverage_min_local <= target_coverage <= coverage_max_local):
            raise ValueError(
                f"Bedeckung {target_coverage:.6g} mol/kg liegt außerhalb des Bereichs "
                f"[{coverage_min_local:.6g}, {coverage_max_local:.6g}]"
            )

        lower_index = int(np.searchsorted(coverage_array, target_coverage) - 1)
        lower_index = max(0, min(lower_index, len(coverage_array) - 2))

        coverage_low = float(coverage_array[lower_index])
        coverage_high = float(coverage_array[lower_index + 1])
        pressure_low = float(pressure_array[lower_index])
        pressure_high = float(pressure_array[lower_index + 1])

        if np.isclose(coverage_high, coverage_low):
            raise ValueError(
                "Interpolation nicht möglich, da zwei benachbarte Bedeckungen gleich sind."
            )

        pressure_interpolated = pressure_low + (
            (target_coverage - coverage_low)
            * (pressure_high - pressure_low)
            / (coverage_high - coverage_low)
        )

        return {
            "i_low": int(interpolation_data["step"].iloc[lower_index]),
            "i_high": int(interpolation_data["step"].iloc[lower_index + 1]),
            "p_low_mbar": pressure_low,
            "p_high_mbar": pressure_high,
            "cov_low_molkg": coverage_low,
            "cov_high_molkg": coverage_high,
            "p_interp_mbar": float(pressure_interpolated),
        }


    def build_isostere_points_table(iso_calc_in, T_info_in, coverages_in):
        isostere_rows_local = []

        for target_coverage in coverages_in:
            for temp_key in ["T1", "T2", "T3"]:
                interpolation_result = interpolate_pressure_at_coverage(
                    iso_calc_in[temp_key],
                    target_coverage,
                )

                pressure_interpolated_Pa = (
                    interpolation_result["p_interp_mbar"] * mbar_to_Pa
                )

                isostere_rows_local.append(
                    {
                        "coverage_molkg": float(target_coverage),
                        "Temperatur": T_info_in[temp_key]["label"],
                        "T_C": float(T_info_in[temp_key]["T_C"]),
                        "T_K": float(T_info_in[temp_key]["T_K"]),
                        **{
                            f"bracket_{interpolation_key}": interpolation_value
                            for interpolation_key, interpolation_value
                            in interpolation_result.items()
                        },
                        "p_interp_Pa": float(pressure_interpolated_Pa),
                        "ln_p": float(np.log(pressure_interpolated_Pa)),
                        "invT_Kinv": float(1.0 / T_info_in[temp_key]["T_K"]),
                    }
                )

        return pd.DataFrame(isostere_rows_local).sort_values(
            ["coverage_molkg", "T_K"]
        ).reset_index(drop=True)


    def make_interpolation_plot(iso_calc_in, T_info_in, isostere_points_in, coverages_in):
        fig_interp_local, ax_interp_local = plt.subplots(figsize=(7, 4.5))

        for temp_key in ["T1", "T2", "T3"]:
            current_iso = iso_calc_in[temp_key]
            ax_interp_local.plot(
                current_iso["pG_corr_mbar"],
                current_iso["nads_per_kg_molkg"],
                marker="o",
                linestyle="None",
                label=T_info_in[temp_key]["label"],
            )

        for target_coverage in coverages_in:
            ax_interp_local.axhline(target_coverage, linestyle="--", alpha=0.5)

            coverage_points = isostere_points_in[
                isostere_points_in["coverage_molkg"] == target_coverage
            ]

            ax_interp_local.plot(
                coverage_points["bracket_p_interp_mbar"],
                coverage_points["coverage_molkg"],
                marker="x",
                linestyle="None",
                markersize=8,
            )

        ax_interp_local.set_xlabel(r"$p_G$ / mbar")
        ax_interp_local.set_ylabel(r"$n_{\mathrm{ads},s}$ / mol kg$^{-1}$")
        ax_interp_local.set_title("Isothermen mit interpolierten Isosterenpunkten")
        ax_interp_local.grid(True)
        ax_interp_local.legend()
        fig_interp_local.tight_layout()

        return fig_interp_local


    isostere_points = build_isostere_points_table(
        iso_calc_in=iso_calc,
        T_info_in=T_info,
        coverages_in=selected_coverages,
    )

    interpolation_figure = make_interpolation_plot(
        iso_calc_in=iso_calc,
        T_info_in=T_info,
        isostere_points_in=isostere_points,
        coverages_in=selected_coverages,
    )

    mo.vstack(
        [
            mo.md("## 11. Lineare Interpolation der Isosterenpunkte"),
            mo.md(
                r"""
                Für jede gewählte Bedeckung wird auf jeder Isotherme der zugehörige
                Gleichgewichtsdruck \(p_G\) durch lineare Interpolation zwischen zwei
                benachbarten Messpunkten bestimmt.
                """
            ),
            mo.ui.table(
                isostere_points[
                    [
                        "coverage_molkg",
                        "Temperatur",
                        "T_K",
                        "bracket_i_low",
                        "bracket_i_high",
                        "bracket_cov_low_molkg",
                        "bracket_cov_high_molkg",
                        "bracket_p_low_mbar",
                        "bracket_p_high_mbar",
                        "bracket_p_interp_mbar",
                        "ln_p",
                        "invT_Kinv",
                    ]
                ]
            ),
            interpolation_figure,
        ]
    )
    return (isostere_points,)


@app.cell
def _(curve_fit, isostere_points, mo, np, pd, plt, r_squared):
    def linear_model_for_isostere(inv_temperature, intercept, slope):
        return intercept + slope * inv_temperature


    def fit_isosteres(isostere_points_in, reference_pressure_Pa):
        fit_rows_local = []

        fig_isostere_local, ax_isostere_local = plt.subplots(figsize=(7, 4.5))

        for target_coverage in sorted(isostere_points_in["coverage_molkg"].unique()):
            coverage_subset = isostere_points_in[
                isostere_points_in["coverage_molkg"] == target_coverage
            ].copy()

            inv_temperature_values = coverage_subset["invT_Kinv"].to_numpy(float)
            log_pressure_values = np.log(
                coverage_subset["p_interp_Pa"].to_numpy(float) / reference_pressure_Pa
            )

            fit_parameters, fit_covariance = curve_fit(
                linear_model_for_isostere,
                inv_temperature_values,
                log_pressure_values,
            )

            intercept_fit, slope_fit = fit_parameters
            sigma_intercept, sigma_slope = np.sqrt(np.diag(fit_covariance))

            fitted_log_pressure = linear_model_for_isostere(
                inv_temperature_values,
                intercept_fit,
                slope_fit,
            )

            fit_r_squared = r_squared(log_pressure_values, fitted_log_pressure)

            inv_temperature_plot = np.linspace(
                inv_temperature_values.min(),
                inv_temperature_values.max(),
                100,
            )

            log_pressure_plot = linear_model_for_isostere(
                inv_temperature_plot,
                intercept_fit,
                slope_fit,
            )

            ax_isostere_local.plot(
                inv_temperature_values,
                log_pressure_values,
                marker="o",
                linestyle="None",
                label=f"{target_coverage:.3g} mol/kg",
            )

            ax_isostere_local.plot(inv_temperature_plot, log_pressure_plot)

            fit_rows_local.append(
                {
                    "coverage_molkg": float(target_coverage),
                    "a_intercept": float(intercept_fit),
                    "sigma_a": float(sigma_intercept),
                    "b_slope_K": float(slope_fit),
                    "sigma_b_K": float(sigma_slope),
                    "R_squared": float(fit_r_squared),
                }
            )

        ax_isostere_local.set_xlabel(r"$1/T$ / K$^{-1}$")
        ax_isostere_local.set_ylabel(r"$\ln(p_G/p^0)$")
        ax_isostere_local.set_title("Adsorptionsisosteren mit linearem Fit")
        ax_isostere_local.grid(True)
        ax_isostere_local.legend()
        fig_isostere_local.tight_layout()

        fit_table_local = pd.DataFrame(fit_rows_local).sort_values("coverage_molkg")

        return fit_table_local, fig_isostere_local


    reference_pressure_Pa = 1e5

    isostere_fit_table, isostere_fit_figure = fit_isosteres(
        isostere_points_in=isostere_points,
        reference_pressure_Pa=reference_pressure_Pa,
    )

    mo.vstack(
        [
            mo.md(
                r"""
                ## 12. Isosteren-Fit

                Für konstante Bedeckung wird aufgetragen:

                \[
                \ln\left(\frac{p}{p^0}\right)
                \quad \text{gegen} \quad
                \frac{1}{T}
                \]

                Aus der Steigung \(b\) folgt:

                \[
                \Delta H_\mathrm{ads} = -R \cdot b
                \]
                """
            ),
            isostere_fit_figure,
            mo.ui.table(isostere_fit_table),
        ]
    )
    return (isostere_fit_table,)


@app.cell
def _(R, format_value_uncert, isostere_fit_table, mo, plt):
    def build_adsorption_enthalpy_table(isostere_fit_table_in):
        enthalpy_table_local = isostere_fit_table_in.copy()

        enthalpy_table_local["dH_ads_J_per_mol"] = (
            -R * enthalpy_table_local["b_slope_K"]
        )

        enthalpy_table_local["sigma_dH_J_per_mol"] = (
            R * enthalpy_table_local["sigma_b_K"]
        )

        enthalpy_table_local["dH_ads_kJ_per_mol"] = (
            enthalpy_table_local["dH_ads_J_per_mol"] / 1000.0
        )

        enthalpy_table_local["sigma_dH_kJ_per_mol"] = (
            enthalpy_table_local["sigma_dH_J_per_mol"] / 1000.0
        )

        enthalpy_table_local["DeltaH_ads_formatted"] = [
            format_value_uncert(enthalpy_value, enthalpy_uncertainty)
            for enthalpy_value, enthalpy_uncertainty
            in zip(
                enthalpy_table_local["dH_ads_kJ_per_mol"],
                enthalpy_table_local["sigma_dH_kJ_per_mol"],
            )
        ]

        return enthalpy_table_local[
            [
                "coverage_molkg",
                "b_slope_K",
                "sigma_b_K",
                "R_squared",
                "dH_ads_kJ_per_mol",
                "sigma_dH_kJ_per_mol",
                "DeltaH_ads_formatted",
            ]
        ].sort_values("coverage_molkg")


    def make_adsorption_enthalpy_plot(enthalpy_output_table):
        fig_dH_local, ax_dH_local = plt.subplots(figsize=(7, 4.5))

        ax_dH_local.errorbar(
            enthalpy_output_table["coverage_molkg"],
            enthalpy_output_table["dH_ads_kJ_per_mol"],
            yerr=enthalpy_output_table["sigma_dH_kJ_per_mol"],
            marker="o",
            linestyle="None",
            capsize=4,
        )

        ax_dH_local.axhline(0, linewidth=1)
        ax_dH_local.set_xlabel(r"$n_{\mathrm{ads},s}$ / mol kg$^{-1}$")
        ax_dH_local.set_ylabel(r"$\Delta H_{\mathrm{ads}}$ / kJ mol$^{-1}$")
        ax_dH_local.set_title(
            "Isostere Adsorptionsenthalpie als Funktion der Bedeckung"
        )
        ax_dH_local.grid(True)
        fig_dH_local.tight_layout()

        return fig_dH_local


    dH_out = build_adsorption_enthalpy_table(
        isostere_fit_table_in=isostere_fit_table,
    )

    adsorption_enthalpy_figure = make_adsorption_enthalpy_plot(
        enthalpy_output_table=dH_out,
    )

    mo.vstack(
        [
            mo.md(
                r"""
                ## 13. Adsorptionsenthalpie

                Die Adsorptionsenthalpie wird aus der Steigung der Isosteren bestimmt.

                Negative Werte entsprechen einer exothermen Adsorption. Die Größenordnung kann
                später in der Diskussion zur Einordnung der Wechselwirkung verwendet werden.
                """
            ),
            mo.ui.table(dH_out),
            adsorption_enthalpy_figure,
        ]
    )
    return (dH_out,)


@app.cell
def _(T_info, mo, np):
    def langmuir_model(pressure_mbar, qmax, b_parameter):
        return qmax * b_parameter * pressure_mbar / (1.0 + b_parameter * pressure_mbar)


    def freundlich_model(pressure_mbar, K_parameter, n_parameter):
        return K_parameter * pressure_mbar**n_parameter


    def temkin_model(pressure_mbar, B_parameter, A_parameter):
        return B_parameter * np.log(A_parameter * pressure_mbar)


    temperature_label_to_key = {
        T_info["T1"]["label"]: "T1",
        T_info["T2"]["label"]: "T2",
        T_info["T3"]["label"]: "T3",
    }

    fit_temp_select = mo.ui.dropdown(
        options=list(temperature_label_to_key.keys()),
        value=T_info["T2"]["label"],
        label="Temperatur für Modellvergleich",
    )

    mo.vstack(
        [
            mo.md(
                r"""
                ## 14. Modellvergleich

                Es werden drei Modelle direkt an eine ausgewählte Isotherme gefittet.

                **Langmuir:**

                \[
                n_{\mathrm{ads},s}(p)=
                q_\mathrm{max}\frac{bp}{1+bp}
                \]

                **Freundlich:**

                \[
                n_{\mathrm{ads},s}(p)=Kp^n
                \]

                **Temkin:**

                \[
                n_{\mathrm{ads},s}(p)=B\ln(Ap)
                \]

                Der Fit erfolgt gegen den Gleichgewichtsdruck \(p_G\).
                """
            ),
            fit_temp_select,
        ]
    )
    return (
        fit_temp_select,
        freundlich_model,
        langmuir_model,
        temkin_model,
        temperature_label_to_key,
    )


@app.cell
def _(fit_temp_select, mo, temperature_label_to_key):
    fit_temperature_label = fit_temp_select.value
    fit_temperature_key = temperature_label_to_key[fit_temperature_label]

    mo.md(
        f"""
        Gewählte Temperatur für den Modellvergleich: **{fit_temperature_label}**
        """
    )
    return fit_temperature_key, fit_temperature_label


@app.cell
def _(
    curve_fit,
    fit_temperature_key,
    fit_temperature_label,
    format_value_uncert,
    freundlich_model,
    iso_calc,
    langmuir_model,
    mo,
    np,
    pd,
    plt,
    r_squared,
    temkin_model,
):
    def fit_adsorption_models(isotherm_calc_for_fit, temperature_label):
        fit_input_table = isotherm_calc_for_fit.copy()

        pressure_values_raw = fit_input_table["pG_corr_mbar"].to_numpy(dtype=float)
        coverage_values_raw = fit_input_table["nads_per_kg_molkg"].to_numpy(dtype=float)

        positive_pressure_mask = pressure_values_raw > 0

        pressure_values = pressure_values_raw[positive_pressure_mask]
        coverage_values = coverage_values_raw[positive_pressure_mask]

        qmax_start = float(np.nanmax(coverage_values))
        b_start = float(1.0 / max(1e-12, np.nanmean(pressure_values)))

        langmuir_parameters, langmuir_covariance = curve_fit(
            langmuir_model,
            pressure_values,
            coverage_values,
            p0=[qmax_start, b_start],
            bounds=([0.0, 0.0], [np.inf, np.inf]),
            maxfev=200000,
        )

        qmax_L, b_L = langmuir_parameters
        sigma_qmax_L, sigma_b_L = np.sqrt(np.diag(langmuir_covariance))
        R2_L = r_squared(
            coverage_values,
            langmuir_model(pressure_values, *langmuir_parameters),
        )

        freundlich_K_start = (
            float(coverage_values[0] / (pressure_values[0] ** 0.5))
            if pressure_values[0] > 0
            else 0.1
        )

        freundlich_parameters, freundlich_covariance = curve_fit(
            freundlich_model,
            pressure_values,
            coverage_values,
            p0=[max(freundlich_K_start, 1e-9), 0.5],
            bounds=([0.0, 0.0], [np.inf, np.inf]),
            maxfev=200000,
        )

        K_F, n_F = freundlich_parameters
        sigma_K_F, sigma_n_F = np.sqrt(np.diag(freundlich_covariance))
        R2_F = r_squared(
            coverage_values,
            freundlich_model(pressure_values, *freundlich_parameters),
        )

        temkin_A_start = float(1.0 / max(1e-12, np.nanmean(pressure_values)))

        temkin_parameters, temkin_covariance = curve_fit(
            temkin_model,
            pressure_values,
            coverage_values,
            p0=[0.5, temkin_A_start],
            bounds=([0.0, 0.0], [np.inf, np.inf]),
            maxfev=200000,
        )

        B_T, A_T = temkin_parameters
        sigma_B_T, sigma_A_T = np.sqrt(np.diag(temkin_covariance))
        R2_T = r_squared(
            coverage_values,
            temkin_model(pressure_values, *temkin_parameters),
        )

        pressure_plot_values = np.linspace(
            float(pressure_values.min()),
            float(pressure_values.max()),
            400,
        )

        fig_model_local, ax_model_local = plt.subplots(figsize=(7, 4.5))

        ax_model_local.plot(
            pressure_values,
            coverage_values,
            marker="o",
            linestyle="None",
            label=f"Messpunkte {temperature_label}",
        )

        ax_model_local.plot(
            pressure_plot_values,
            langmuir_model(pressure_plot_values, *langmuir_parameters),
            label="Langmuir",
        )

        ax_model_local.plot(
            pressure_plot_values,
            freundlich_model(pressure_plot_values, *freundlich_parameters),
            label="Freundlich",
        )

        ax_model_local.plot(
            pressure_plot_values,
            temkin_model(pressure_plot_values, *temkin_parameters),
            label="Temkin",
        )

        ax_model_local.set_xlabel(r"$p_G$ / mbar")
        ax_model_local.set_ylabel(r"$n_{\mathrm{ads},s}$ / mol kg$^{-1}$")
        ax_model_local.set_title(f"Modellvergleich bei {temperature_label}")
        ax_model_local.grid(True)
        ax_model_local.legend()
        fig_model_local.tight_layout()

        model_table_local = pd.DataFrame(
            [
                {
                    "Modell": "Langmuir",
                    "Param1_Name": "qmax / mol kg^-1",
                    "Param1": qmax_L,
                    "sigma_Param1": sigma_qmax_L,
                    "Param2_Name": "b / mbar^-1",
                    "Param2": b_L,
                    "sigma_Param2": sigma_b_L,
                    "R_squared": R2_L,
                },
                {
                    "Modell": "Freundlich",
                    "Param1_Name": "K",
                    "Param1": K_F,
                    "sigma_Param1": sigma_K_F,
                    "Param2_Name": "n",
                    "Param2": n_F,
                    "sigma_Param2": sigma_n_F,
                    "R_squared": R2_F,
                },
                {
                    "Modell": "Temkin",
                    "Param1_Name": "B / mol kg^-1",
                    "Param1": B_T,
                    "sigma_Param1": sigma_B_T,
                    "Param2_Name": "A / mbar^-1",
                    "Param2": A_T,
                    "sigma_Param2": sigma_A_T,
                    "R_squared": R2_T,
                },
            ]
        )

        model_table_local["Param1_formatted"] = [
            format_value_uncert(parameter_value, parameter_uncertainty)
            for parameter_value, parameter_uncertainty
            in zip(model_table_local["Param1"], model_table_local["sigma_Param1"])
        ]

        model_table_local["Param2_formatted"] = [
            format_value_uncert(parameter_value, parameter_uncertainty)
            for parameter_value, parameter_uncertainty
            in zip(model_table_local["Param2"], model_table_local["sigma_Param2"])
        ]

        best_model_local = model_table_local.sort_values(
            "R_squared",
            ascending=False,
        ).iloc[0]["Modell"]

        return model_table_local, fig_model_local, best_model_local


    model_table, model_comparison_figure, best_model = fit_adsorption_models(
        isotherm_calc_for_fit=iso_calc[fit_temperature_key],
        temperature_label=fit_temperature_label,
    )

    mo.vstack(
        [
            mo.md(f"## 15. Modellfit bei {fit_temperature_label}"),
            model_comparison_figure,
            mo.ui.table(
                model_table[
                    [
                        "Modell",
                        "Param1_Name",
                        "Param1_formatted",
                        "Param2_Name",
                        "Param2_formatted",
                        "R_squared",
                    ]
                ]
            ),
            mo.md(
                rf"""
                Nach dem Bestimmtheitsmaß \(R^2\) beschreibt in dieser Rechnung das
                **{best_model}-Modell** die gewählte Isotherme am besten.

                Dieser Satz ist nur eine rechnerische Zusammenfassung und ersetzt nicht die
                Diskussion der Modellannahmen.
                """
            ),
        ]
    )
    return best_model, model_table


@app.cell
def _(
    T_info,
    T_room_K_input,
    VE_mL_input,
    adsorbens_input,
    adsorptiv_input,
    best_model,
    cal_summary,
    checks_df,
    dH_out,
    fit_temperature_label,
    m_ads_g_input,
    mo,
    model_table,
    offset_input,
    pd,
    selected_coverages,
):
    def build_results_summary_tables(
        VE_mL_in,
        offset_mbar_in,
        m_ads_g_in,
        T_room_K_in,
        adsorbens_in,
        adsorptiv_in,
        T_info_in,
        cal_summary_in,
        checks_df_in,
        selected_coverages_in,
        dH_out_in,
        model_table_in,
        fit_temperature_label_in,
        best_model_in,
    ):
        general_summary_local = pd.DataFrame(
            [
                {
                    "Größe": "Eichvolumen VE",
                    "Wert": f"{VE_mL_in:.4g}",
                    "Einheit": "mL",
                    "Hinweis": "Vorgegebenes Eichvolumen",
                },
                {
                    "Größe": "Druckoffset",
                    "Wert": f"{offset_mbar_in:.4g}",
                    "Einheit": "mbar",
                    "Hinweis": "Korrektur: p_korr = p_roh - offset",
                },
                {
                    "Größe": "Masse Adsorbens",
                    "Wert": f"{m_ads_g_in:.5g}",
                    "Einheit": "g",
                    "Hinweis": "Trockene Masse für spezifische Adsorbatmenge",
                },
                {
                    "Größe": "Raumtemperatur",
                    "Wert": f"{T_room_K_in:.2f}",
                    "Einheit": "K",
                    "Hinweis": "Für die volumetrische Stoffmengenbilanz",
                },
                {
                    "Größe": "Adsorbens",
                    "Wert": adsorbens_in if adsorbens_in else "nicht angegeben",
                    "Einheit": "-",
                    "Hinweis": "Feststoffprobe",
                },
                {
                    "Größe": "Adsorptiv / Probengas",
                    "Wert": adsorptiv_in if adsorptiv_in else "nicht angegeben",
                    "Einheit": "-",
                    "Hinweis": "Für Rechnung nicht zwingend nötig, für Diskussion wichtig",
                },
            ]
        )

        temperature_summary_local = pd.DataFrame(
            [
                {
                    "Messreihe": temp_key,
                    "Temperatur / °C": T_info_in[temp_key]["T_C"],
                    "Temperatur / K": T_info_in[temp_key]["T_K"],
                }
                for temp_key in ["T1", "T2", "T3"]
            ]
        )

        calibration_accuracy_local = cal_summary_in.copy()
        calibration_accuracy_local["relative_std_percent"] = (
            calibration_accuracy_local["std"]
            / calibration_accuracy_local["mean"]
            * 100.0
        )
        calibration_accuracy_local["relative_sem_percent"] = (
            calibration_accuracy_local["sem"]
            / calibration_accuracy_local["mean"]
            * 100.0
        )

        selected_coverages_local = pd.DataFrame(
            {
                "Isostere": ["1", "2", "3"],
                "Bedeckung / mol kg^-1": selected_coverages_in,
            }
        )

        enthalpy_summary_local = dH_out_in.copy()
        enthalpy_summary_local = enthalpy_summary_local[
            [
                "coverage_molkg",
                "DeltaH_ads_formatted",
                "R_squared",
                "dH_ads_kJ_per_mol",
                "sigma_dH_kJ_per_mol",
            ]
        ].rename(
            columns={
                "coverage_molkg": "Bedeckung / mol kg^-1",
                "DeltaH_ads_formatted": "ΔH_ads / kJ mol^-1",
                "R_squared": "R² Isosteren-Fit",
                "dH_ads_kJ_per_mol": "ΔH_ads roh / kJ mol^-1",
                "sigma_dH_kJ_per_mol": "σ(ΔH_ads) / kJ mol^-1",
            }
        )

        model_summary_local = model_table_in[
            [
                "Modell",
                "Param1_Name",
                "Param1_formatted",
                "Param2_Name",
                "Param2_formatted",
                "R_squared",
            ]
        ].copy()

        accuracy_notes_local = pd.DataFrame(
            [
                {
                    "Punkt": "Druckmessung",
                    "Bedeutung": "Alle Drücke werden um den gemessenen Offset korrigiert.",
                    "Konsequenz": "Ein falsches Vorzeichen beim Offset verschiebt alle Stoffmengen systematisch.",
                },
                {
                    "Punkt": "Volumenkalibration",
                    "Bedeutung": "VM und VP werden aus sechs Einzelmessungen gemittelt.",
                    "Konsequenz": "Die Standardabweichung zeigt die Streuung der Kalibration; der SEM beschreibt die Unsicherheit des Mittelwertes.",
                },
                {
                    "Punkt": "Temperatur in PV = nRT",
                    "Bedeutung": "Für die volumetrische Bilanz wird die Raumtemperatur verwendet.",
                    "Konsequenz": "Die Messtemperaturen werden für die Isosteren und die Adsorptionsenthalpie verwendet.",
                },
                {
                    "Punkt": "Interpolation der Isosteren",
                    "Bedeutung": "Die Gleichgewichtsdrücke bei konstanter Bedeckung werden linear zwischen Messpunkten interpoliert.",
                    "Konsequenz": "Bedeckungen nahe am Rand des gemeinsamen Bereichs sind weniger robust.",
                },
                {
                    "Punkt": "Adsorptionsenthalpie",
                    "Bedeutung": "Die Unsicherheit stammt hier aus dem linearen Fit der Isosteren.",
                    "Konsequenz": "Systematische Fehler aus Druck, Volumen und Temperatur sind darin nicht vollständig enthalten.",
                },
                {
                    "Punkt": "Modellvergleich",
                    "Bedeutung": "Das beste Modell wird rechnerisch nach R² bestimmt.",
                    "Konsequenz": "Die physikalischen Modellannahmen müssen im Protokoll zusätzlich diskutiert werden.",
                },
            ]
        )

        short_text_local = rf"""
**Kurzzusammenfassung der Auswertung**

- Die Kalibration ergab Mittelwerte für \(V_M\) und \(V_P\), die anschließend für alle drei Isothermen verwendet wurden.
- Die Stoffmengenbilanz wurde für alle drei Temperaturen mit der Raumtemperatur in der idealen Gasgleichung berechnet.
- Für die Isosteren wurden drei Bedeckungen im gemeinsamen Bedeckungsbereich der Isothermen gewählt.
- Die Adsorptionsenthalpien wurden aus der Steigung der Auftragung \(\ln(p/p^0)\) gegen \(1/T\) bestimmt.
- Für den Modellvergleich bei **{fit_temperature_label_in}** liefert nach \(R^2\) das **{best_model_in}-Modell** die beste rechnerische Beschreibung.
"""

        return {
            "general": general_summary_local,
            "temperatures": temperature_summary_local,
            "calibration_accuracy": calibration_accuracy_local,
            "checks": checks_df_in.copy(),
            "coverages": selected_coverages_local,
            "enthalpy": enthalpy_summary_local,
            "models": model_summary_local,
            "accuracy_notes": accuracy_notes_local,
            "short_text": short_text_local,
        }


    summary_tables = build_results_summary_tables(
        VE_mL_in=float(VE_mL_input.value),
        offset_mbar_in=float(offset_input.value),
        m_ads_g_in=float(m_ads_g_input.value),
        T_room_K_in=float(T_room_K_input.value),
        adsorbens_in=str(adsorbens_input.value).strip(),
        adsorptiv_in=str(adsorptiv_input.value).strip(),
        T_info_in=T_info,
        cal_summary_in=cal_summary,
        checks_df_in=checks_df,
        selected_coverages_in=selected_coverages,
        dH_out_in=dH_out,
        model_table_in=model_table,
        fit_temperature_label_in=fit_temperature_label,
        best_model_in=best_model,
    )

    mo.vstack(
        [
            mo.md("## 16. Zusammenfassung der ermittelten Daten"),
            mo.md(summary_tables["short_text"]),
            mo.md("### Allgemeine Eingabeparameter"),
            mo.ui.table(summary_tables["general"]),
            mo.md("### Messtemperaturen"),
            mo.ui.table(summary_tables["temperatures"]),
            mo.md("### Kalibrationsgenauigkeit"),
            mo.md(
                r"""
                Die relative Standardabweichung beschreibt die Streuung der Einzelmessungen.
                Der relative Standardfehler des Mittelwerts gibt an, wie genau der Mittelwert aus den Wiederholungen bestimmt wurde.
                """
            ),
            mo.ui.table(summary_tables["calibration_accuracy"]),
            mo.md("### Plausibilitätschecks der Stoffmengenbilanz"),
            mo.ui.table(summary_tables["checks"]),
            mo.md("### Gewählte Bedeckungen für die Isosteren"),
            mo.ui.table(summary_tables["coverages"]),
            mo.md("### Bestimmte Adsorptionsenthalpien"),
            mo.ui.table(summary_tables["enthalpy"]),
            mo.md(f"### Modellvergleich bei {fit_temperature_label}"),
            mo.ui.table(summary_tables["models"]),
            mo.md("### Hinweise zur Genauigkeit und Interpretation"),
            mo.ui.table(summary_tables["accuracy_notes"]),
        ]
    )
    return


@app.cell
def _(
    cal_calc,
    cal_summary,
    dH_out,
    iso_calc,
    isostere_points,
    mo,
    model_table,
):
    calibration_csv = cal_calc.to_csv(index=False)
    calibration_summary_csv = cal_summary.to_csv(index=False)

    isotherm_T1_csv = iso_calc["T1"].to_csv(index=False)
    isotherm_T2_csv = iso_calc["T2"].to_csv(index=False)
    isotherm_T3_csv = iso_calc["T3"].to_csv(index=False)

    isostere_points_csv = isostere_points.to_csv(index=False)
    dH_csv = dH_out.to_csv(index=False)
    model_csv = model_table.to_csv(index=False)

    mo.vstack(
        [
            mo.md("## 16. Export für Protokoll"),
            mo.md(
                "Die folgenden Tabellen können als CSV heruntergeladen und für das Protokoll "
                "weiterverwendet werden."
            ),
            mo.download(
                data=calibration_csv,
                filename="calibration_calculated.csv",
                mimetype="text/csv",
                label="Kalibration als CSV herunterladen",
            ),
            mo.download(
                data=calibration_summary_csv,
                filename="calibration_summary.csv",
                mimetype="text/csv",
                label="Kalibrationsstatistik als CSV herunterladen",
            ),
            mo.download(
                data=isotherm_T1_csv,
                filename="isotherm_T1_calculated.csv",
                mimetype="text/csv",
                label="Isotherme T1 als CSV herunterladen",
            ),
            mo.download(
                data=isotherm_T2_csv,
                filename="isotherm_T2_calculated.csv",
                mimetype="text/csv",
                label="Isotherme T2 als CSV herunterladen",
            ),
            mo.download(
                data=isotherm_T3_csv,
                filename="isotherm_T3_calculated.csv",
                mimetype="text/csv",
                label="Isotherme T3 als CSV herunterladen",
            ),
            mo.download(
                data=isostere_points_csv,
                filename="isostere_points.csv",
                mimetype="text/csv",
                label="Isosterenpunkte als CSV herunterladen",
            ),
            mo.download(
                data=dH_csv,
                filename="dH_ads_table.csv",
                mimetype="text/csv",
                label="Adsorptionsenthalpien als CSV herunterladen",
            ),
            mo.download(
                data=model_csv,
                filename="model_fit_table.csv",
                mimetype="text/csv",
                label="Modellfit-Tabelle als CSV herunterladen",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
