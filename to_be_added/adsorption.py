# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.6",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.ticker import AutoMinorLocator as _AutoMinorLocator
    from scipy.integrate import trapezoid
    from scipy.signal import find_peaks, savgol_filter

    plt.rcParams.update(
        {
            "figure.figsize": (8.8, 5.2),
            "figure.dpi": 140,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "font.size": 10.5,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )

    def prettify_ax(ax, xlabel=None, ylabel=None, title=None):
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

        ax.xaxis.set_minor_locator(_AutoMinorLocator())
        ax.yaxis.set_minor_locator(_AutoMinorLocator())
        ax.grid(True, which="major", alpha=0.22)
        ax.grid(True, which="minor", alpha=0.08)
        return ax

    return find_peaks, mo, np, pd, plt, prettify_ax, savgol_filter, trapezoid


@app.cell
def _(mo):
    mo.md("""
    # Interaktive Auswertung: Physisorption, Porenstruktur, TPR und H₂-Chemisorption

    Upload der Messdateien, Auswahl von Parametern und automatische Berechnung von Plots und Kennzahlen.
    Das Notebook ist upload-basiert aufgebaut und kommt ohne lokale, absolute Dateipfade aus.
    """)
    return


@app.cell
def _(mo):
    data_upload = mo.ui.file(
        filetypes=[".txt"],
        multiple=True,
        kind="area",
        label="Messdateien hochladen",
    )

    mo.vstack(
        [
            mo.md(
                """
    ## 1. Daten hochladen

    Erwartete Dateitypen:
    - N₂-Isotherme: Dateiname enthält `powder_isotherm`
    - BET-Linearisierung: Dateiname enthält `MP_BET`
    - DFT/NLDFT-Porenstruktur: Dateiname enthält `DFT_pore_size`
    - H₂-Chemisorption: Dateiname enthält `sorption_60deg` oder `ChemIso`
    - H₂-TPR: Dateiname enthält `TPR`
    """
            ),
            data_upload,
            mo.md(
                """
    Primärer Workflow: Dateien hier hochladen. Nicht hochgeladene Datensätze werden später als
    `nicht verfügbar` markiert; nur die davon abhängigen Abschnitte werden übersprungen.
    """
            ),
        ]
    )
    return (data_upload,)


@app.cell
def _(pd):
    dataset_labels = {
        "bet_isotherm": "N₂-Isotherme",
        "bet_summary": "BET-Linearisierung",
        "dft_pores": "DFT/NLDFT-Porenstruktur",
        "h2_chemisorption": "H₂-Chemisorption",
        "tpr": "H₂-TPR",
    }

    def infer_dataset_key(filename):
        _name = str(filename).lower()

        if "dft_pore_size" in _name:
            return "dft_pores"
        if "powder_isotherm" in _name:
            return "bet_isotherm"
        if "mp_bet" in _name:
            return "bet_summary"
        if "sorption_60deg" in _name or "chemiso" in _name:
            return "h2_chemisorption"
        if "tpr" in _name:
            return "tpr"
        return None

    def _safe_float(token):
        return float(
            str(token)
            .strip()
            .replace("\u2212", "-")
            .replace("\u2013", "-")
            .replace(",", ".")
        )

    def text_from_upload(file_obj):
        if file_obj is None:
            return ""

        _raw = None

        if isinstance(file_obj, dict):
            _raw = file_obj.get("contents")
            if _raw is None:
                _raw = file_obj.get("content")
            if _raw is None:
                _raw = file_obj.get("value")
        elif isinstance(file_obj, (str, bytes, bytearray)):
            _raw = file_obj
        else:
            if hasattr(file_obj, "contents"):
                _raw = file_obj.contents
                if callable(_raw):
                    _raw = _raw()

            if _raw is None and hasattr(file_obj, "read") and callable(file_obj.read):
                try:
                    _raw = file_obj.read()
                except TypeError:
                    _raw = None

            if _raw is None and hasattr(file_obj, "path") and getattr(file_obj, "path"):
                with open(file_obj.path, "rb") as _handle:
                    _raw = _handle.read()

        if _raw is None:
            raise ValueError("Upload-Inhalt konnte nicht gelesen werden.")

        if isinstance(_raw, bytearray):
            _raw = bytes(_raw)

        if isinstance(_raw, bytes):
            _text = _raw.decode("utf-8-sig", errors="replace")
        elif isinstance(_raw, str):
            _text = _raw
        else:
            _text = str(_raw)

        return _text.replace("\r\n", "\n").replace("\r", "\n")

    def numeric_table_from_fixed_width_export(
        text,
        expected_columns,
        column_names,
        marker="- - - - - Data Points - - - - -",
    ):
        _lines = text.splitlines()
        _start_index = None

        for _index, _line in enumerate(_lines):
            if marker in _line:
                _start_index = _index + 1
                break

        if _start_index is None:
            raise ValueError(f"Marker nicht gefunden: {marker}")

        _rows = []
        for _line in _lines[_start_index:]:
            _stripped = _line.strip()

            if not _stripped:
                continue

            _parts = _stripped.split()
            if len(_parts) != expected_columns:
                continue

            try:
                _rows.append([_safe_float(_part) for _part in _parts])
            except ValueError:
                continue

        if not _rows:
            raise ValueError("Keine numerischen Datenzeilen gefunden.")

        return pd.DataFrame(_rows, columns=column_names)

    def read_tpr_text(text):
        _lines = text.splitlines()
        _start_index = None

        for _index, _line in enumerate(_lines):
            if _line.strip() == "Messdaten:":
                _start_index = _index + 1
                break

        if _start_index is None:
            raise ValueError("'Messdaten:' wurde nicht gefunden.")

        _rows = []
        _current_segment = None
        _header_seen = False

        for _line in _lines[_start_index:]:
            _stripped = _line.strip()

            if not _stripped:
                continue

            if _stripped.startswith("Segment:"):
                _current_segment = int(_stripped.split(":", 1)[1])
                _header_seen = False
                continue

            if _stripped == "Zeit(s);Signal(mV);Temperatur(°C)":
                _header_seen = True
                continue

            if _current_segment is None or not _header_seen:
                continue

            _parts = _stripped.split(";")
            if len(_parts) != 3:
                continue

            try:
                _time_s = _safe_float(_parts[0])
                _signal_mV = _safe_float(_parts[1])
                _temperature_C = _safe_float(_parts[2])
            except ValueError:
                continue

            _rows.append(
                {
                    "segment": _current_segment,
                    "time_s": _time_s,
                    "time_min": _time_s / 60.0,
                    "signal_mV": _signal_mV,
                    "temperature_C": _temperature_C,
                }
            )

        if not _rows:
            raise ValueError("Keine TPR-Messdaten gefunden.")

        return pd.DataFrame(_rows)

    return (
        dataset_labels,
        infer_dataset_key,
        numeric_table_from_fixed_width_export,
        read_tpr_text,
        text_from_upload,
    )


@app.cell
def _(
    data_upload,
    dataset_labels,
    infer_dataset_key,
    mo,
    pd,
    text_from_upload,
):
    _uploaded_files = list(data_upload.value or [])

    mo.stop(
        len(_uploaded_files) == 0,
        mo.md(
            """
        ## 2. Dateierkennung

        Bitte lade mindestens eine `.txt`-Messdatei hoch, damit die Auswertung startet.
        """
        ),
    )

    recognized_uploads = {key: None for key in dataset_labels}
    _upload_log_rows = []

    for _file_obj in _uploaded_files:
        _file_name = getattr(_file_obj, "name", None) or "unbenannte_datei.txt"
        _dataset_key = infer_dataset_key(_file_name)

        if _dataset_key is None:
            _upload_log_rows.append(
                {
                    "Dateiname": _file_name,
                    "Erkennung": "nicht erkannt",
                    "Status": "ignoriert",
                }
            )
            continue

        if recognized_uploads[_dataset_key] is None:
            recognized_uploads[_dataset_key] = {
                "name": _file_name,
                "text": text_from_upload(_file_obj),
            }
            _upload_log_rows.append(
                {
                    "Dateiname": _file_name,
                    "Erkennung": dataset_labels[_dataset_key],
                    "Status": "verwendet",
                }
            )
        else:
            _upload_log_rows.append(
                {
                    "Dateiname": _file_name,
                    "Erkennung": dataset_labels[_dataset_key],
                    "Status": "zusätzlich hochgeladen, erste Datei verwendet",
                }
            )

    _file_status_table = pd.DataFrame(
        [
            {
                "Datensatz": dataset_labels[_key],
                "Dateiname": (
                    recognized_uploads[_key]["name"]
                    if recognized_uploads[_key] is not None
                    else "nicht hochgeladen"
                ),
                "Status": "bereit" if recognized_uploads[_key] is not None else "fehlt",
            }
            for _key in dataset_labels
        ]
    )

    _upload_log_table = pd.DataFrame(_upload_log_rows)

    _outputs = [
        mo.md("## 2. Dateierkennung"),
        mo.ui.table(_file_status_table),
    ]

    if not _upload_log_table.empty:
        _outputs.append(
            mo.accordion(
                {
                    "Upload-Protokoll": mo.ui.table(_upload_log_table),
                }
            )
        )

    mo.vstack(_outputs)
    return (recognized_uploads,)


@app.cell
def _(mo):
    mass_physisorption_g = mo.ui.number(
        value=0.625,
        start=0,
        step=0.001,
        label="Probenmasse Physisorption / g",
    )
    mass_chemisorption_g = mo.ui.number(
        value=0.317,
        start=0,
        step=0.001,
        label="Probenmasse H₂-Chemisorption / g",
    )
    mass_tpr_g = mo.ui.number(
        value=0.623,
        start=0,
        step=0.001,
        label="Probenmasse TPR / g",
    )
    metal_loading_wt = mo.ui.number(
        value=1.12,
        start=0,
        step=0.01,
        label="Gesamtmetallbeladung / Gew.-%",
    )
    pd_fraction_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.01,
        value=0.5,
        show_value=True,
        label="Pd-Anteil am Gesamtmetall",
    )

    tpr_smoothing_slider = mo.ui.slider(
        start=5,
        stop=101,
        step=2,
        value=21,
        show_value=True,
        label="TPR-Glättungsfenster",
    )
    tpr_peak_prominence_slider = mo.ui.slider(
        start=0.1,
        stop=5.0,
        step=0.1,
        value=0.35,
        show_value=True,
        label="TPR-Peak-Prominenzfaktor",
    )
    tpr_baseline_method = mo.ui.radio(
        options=["none", "polynomial", "robust polynomial"],
        value="robust polynomial",
        label="TPR-Baseline-Methode",
    )
    tpr_baseline_low_max_C = mo.ui.number(
        value=45,
        start=0,
        step=5,
        label="Baselinebereich 1 bis / °C",
    )
    tpr_baseline_mid_min_C = mo.ui.number(
        value=135,
        start=0,
        step=5,
        label="Baselinebereich 2 von / °C",
    )
    tpr_baseline_mid_max_C = mo.ui.number(
        value=250,
        start=0,
        step=5,
        label="Baselinebereich 2 bis / °C",
    )
    tpr_baseline_high_min_C = mo.ui.number(
        value=330,
        start=0,
        step=5,
        label="Baselinebereich 3 ab / °C",
    )

    chemi_logx_checkbox = mo.ui.checkbox(
        value=False,
        label="Druckachse logarithmisch",
    )
    dispersion_branch_selector = mo.ui.radio(
        options=["strong", "combined", "combined - weak"],
        value="strong",
        label="Uptake-Basis für Dispersionsabschätzung",
    )
    sites_per_h2_factor = mo.ui.number(
        value=2.0,
        start=0.1,
        step=0.1,
        label="Metallplätze pro H₂",
    )

    show_isotherm_checkbox = mo.ui.checkbox(value=True, label="N₂-Isotherme")
    show_bet_checkbox = mo.ui.checkbox(value=True, label="BET")
    show_dft_checkbox = mo.ui.checkbox(value=True, label="DFT/NLDFT")
    show_bjh_checkbox = mo.ui.checkbox(value=True, label="Kelvin/BJH")
    show_tpr_checkbox = mo.ui.checkbox(value=True, label="TPR")
    show_chemi_checkbox = mo.ui.checkbox(value=True, label="H₂-Chemisorption")

    mo.vstack(
        [
            mo.md("## 3. Analyseparameter"),
            mo.md("### Proben- und Metallparameter"),
            mo.hstack(
                [
                    mass_physisorption_g,
                    mass_chemisorption_g,
                    mass_tpr_g,
                ]
            ),
            mo.hstack(
                [
                    metal_loading_wt,
                    pd_fraction_slider,
                ]
            ),
            mo.md("### TPR-Parameter"),
            mo.hstack(
                [
                    tpr_smoothing_slider,
                    tpr_peak_prominence_slider,
                ]
            ),
            tpr_baseline_method,
            mo.hstack(
                [
                    tpr_baseline_low_max_C,
                    tpr_baseline_mid_min_C,
                    tpr_baseline_mid_max_C,
                    tpr_baseline_high_min_C,
                ]
            ),
            mo.md("### H₂-Chemisorption"),
            mo.hstack(
                [
                    chemi_logx_checkbox,
                    sites_per_h2_factor,
                ]
            ),
            dispersion_branch_selector,
            mo.md("### Sichtbare Analyseabschnitte"),
            mo.hstack(
                [
                    show_isotherm_checkbox,
                    show_bet_checkbox,
                    show_dft_checkbox,
                    show_bjh_checkbox,
                    show_tpr_checkbox,
                    show_chemi_checkbox,
                ]
            ),
        ]
    )
    return (
        chemi_logx_checkbox,
        dispersion_branch_selector,
        mass_chemisorption_g,
        mass_physisorption_g,
        mass_tpr_g,
        metal_loading_wt,
        pd_fraction_slider,
        show_bet_checkbox,
        show_bjh_checkbox,
        show_chemi_checkbox,
        show_dft_checkbox,
        show_isotherm_checkbox,
        show_tpr_checkbox,
        sites_per_h2_factor,
        tpr_baseline_high_min_C,
        tpr_baseline_low_max_C,
        tpr_baseline_method,
        tpr_baseline_mid_max_C,
        tpr_baseline_mid_min_C,
        tpr_peak_prominence_slider,
        tpr_smoothing_slider,
    )


@app.cell
def _(
    chemi_logx_checkbox,
    dispersion_branch_selector,
    mass_chemisorption_g,
    mass_physisorption_g,
    mass_tpr_g,
    metal_loading_wt,
    mo,
    pd,
    pd_fraction_slider,
    show_bet_checkbox,
    show_bjh_checkbox,
    show_chemi_checkbox,
    show_dft_checkbox,
    show_isotherm_checkbox,
    show_tpr_checkbox,
    sites_per_h2_factor,
    tpr_baseline_high_min_C,
    tpr_baseline_low_max_C,
    tpr_baseline_method,
    tpr_baseline_mid_max_C,
    tpr_baseline_mid_min_C,
    tpr_peak_prominence_slider,
    tpr_smoothing_slider,
):
    _pd_fraction = float(pd_fraction_slider.value)
    _pt_fraction = 1.0 - _pd_fraction

    sample_params = {
        "mass_physisorption_g": float(mass_physisorption_g.value),
        "mass_chemisorption_g": float(mass_chemisorption_g.value),
        "mass_tpr_g": float(mass_tpr_g.value),
        "metal_loading_wt": float(metal_loading_wt.value),
        "pd_fraction": _pd_fraction,
        "pt_fraction": _pt_fraction,
    }

    tpr_settings = {
        "smoothing_window": int(tpr_smoothing_slider.value),
        "peak_prominence_factor": float(tpr_peak_prominence_slider.value),
        "baseline_method": tpr_baseline_method.value,
        "baseline_low_max_C": float(tpr_baseline_low_max_C.value),
        "baseline_mid_min_C": float(tpr_baseline_mid_min_C.value),
        "baseline_mid_max_C": float(tpr_baseline_mid_max_C.value),
        "baseline_high_min_C": float(tpr_baseline_high_min_C.value),
    }

    chemi_settings = {
        "logx": bool(chemi_logx_checkbox.value),
        "dispersion_basis": dispersion_branch_selector.value,
        "sites_per_h2": float(sites_per_h2_factor.value),
    }

    section_flags = {
        "show_isotherm": bool(show_isotherm_checkbox.value),
        "show_bet": bool(show_bet_checkbox.value),
        "show_dft": bool(show_dft_checkbox.value),
        "show_bjh": bool(show_bjh_checkbox.value),
        "show_tpr": bool(show_tpr_checkbox.value),
        "show_chemi": bool(show_chemi_checkbox.value),
    }

    _parameter_table = pd.DataFrame(
        [
            {
                "Parameter": "Probenmasse Physisorption / g",
                "Wert": f"{sample_params['mass_physisorption_g']:.3f}",
            },
            {
                "Parameter": "Probenmasse H₂-Chemisorption / g",
                "Wert": f"{sample_params['mass_chemisorption_g']:.3f}",
            },
            {
                "Parameter": "Probenmasse TPR / g",
                "Wert": f"{sample_params['mass_tpr_g']:.3f}",
            },
            {
                "Parameter": "Gesamtmetallbeladung / Gew.-%",
                "Wert": f"{sample_params['metal_loading_wt']:.2f}",
            },
            {
                "Parameter": "Pd-Anteil",
                "Wert": f"{sample_params['pd_fraction']:.2f}",
            },
            {
                "Parameter": "Pt-Anteil",
                "Wert": f"{sample_params['pt_fraction']:.2f}",
            },
            {
                "Parameter": "TPR-Baseline",
                "Wert": tpr_settings["baseline_method"],
            },
            {
                "Parameter": "Chemisorptions-Basis",
                "Wert": chemi_settings["dispersion_basis"],
            },
        ]
    )

    mo.vstack(
        [
            mo.md("## 4. Aktive Parameter"),
            mo.ui.table(_parameter_table),
        ]
    )
    return chemi_settings, sample_params, section_flags, tpr_settings


@app.cell
def _(
    dataset_labels,
    mo,
    np,
    numeric_table_from_fixed_width_export,
    pd,
    read_tpr_text,
    recognized_uploads,
):
    bet_iso = None
    bet_summary = None
    dft_pores = None
    h2_chemi = None
    tpr = None

    _data_status_rows = []

    for _dataset_key, _dataset_label in dataset_labels.items():
        _payload = recognized_uploads[_dataset_key]

        if _payload is None:
            _data_status_rows.append(
                {
                    "Dateiname": "nicht hochgeladen",
                    "Datensatz": _dataset_label,
                    "Zeilen": 0,
                    "Status": "nicht verfügbar",
                }
            )
            continue

        try:
            if _dataset_key == "bet_isotherm":
                _df = numeric_table_from_fixed_width_export(
                    _payload["text"],
                    expected_columns=6,
                    column_names=[
                        "pressure_bar",
                        "p0_bar",
                        "relative_pressure",
                        "amount_adsorbed_cm3_stp",
                        "amount_adsorbed_cm3_stp_g",
                        "time_min",
                    ],
                )
                _turning_index = int(_df["relative_pressure"].idxmax())
                _df["branch"] = np.where(
                    _df.index <= _turning_index,
                    "Adsorption",
                    "Desorption",
                )
                bet_iso = _df

            elif _dataset_key == "bet_summary":
                bet_summary = numeric_table_from_fixed_width_export(
                    _payload["text"],
                    expected_columns=3,
                    column_names=[
                        "relative_pressure",
                        "amount_adsorbed_cm3_stp_g",
                        "bet_transform_inv_g",
                    ],
                )
                _df = bet_summary

            elif _dataset_key == "dft_pores":
                dft_pores = numeric_table_from_fixed_width_export(
                    _payload["text"],
                    expected_columns=5,
                    column_names=[
                        "pore_diameter_A",
                        "cumulative_pore_volume_cm3_g",
                        "cumulative_surface_area_m2_g",
                        "dV_d_cm3_A_g",
                        "dS_d_m2_A_g",
                    ],
                )
                _df = dft_pores

            elif _dataset_key == "h2_chemisorption":
                h2_chemi = numeric_table_from_fixed_width_export(
                    _payload["text"],
                    expected_columns=5,
                    column_names=[
                        "branch",
                        "pressure_torr",
                        "amount_adsorbed_cm3_stp",
                        "amount_adsorbed_cm3_stp_g",
                        "time_min",
                    ],
                )
                h2_chemi["branch"] = h2_chemi["branch"].round().astype(int)
                h2_chemi["branch_label"] = h2_chemi["branch"].map(
                    {
                        1: "combined",
                        3: "weak",
                        4: "strong",
                    }
                )
                _df = h2_chemi

            elif _dataset_key == "tpr":
                tpr = read_tpr_text(_payload["text"])
                _df = tpr

            _data_status_rows.append(
                {
                    "Dateiname": _payload["name"],
                    "Datensatz": _dataset_label,
                    "Zeilen": len(_df),
                    "Status": "eingelesen",
                }
            )
        except Exception as _exc:
            _data_status_rows.append(
                {
                    "Dateiname": _payload["name"],
                    "Datensatz": _dataset_label,
                    "Zeilen": 0,
                    "Status": f"Fehler: {_exc}",
                }
            )

    _data_status_table = pd.DataFrame(_data_status_rows)

    _preview_tabs = {}
    if bet_iso is not None:
        _preview_tabs["N₂-Isotherme"] = mo.ui.table(bet_iso.head(8))
    if bet_summary is not None:
        _preview_tabs["BET"] = mo.ui.table(bet_summary.head(8))
    if dft_pores is not None:
        _preview_tabs["DFT"] = mo.ui.table(dft_pores.head(8))
    if h2_chemi is not None:
        _preview_tabs["H₂-Chemisorption"] = mo.ui.table(h2_chemi.head(8))
    if tpr is not None:
        _preview_tabs["TPR"] = mo.ui.table(tpr.head(8))

    _outputs = [
        mo.md("## 5. Eingelesene Datensätze"),
        mo.ui.table(_data_status_table),
    ]

    if _preview_tabs:
        _outputs.append(
            mo.accordion(
                {
                    "Kompakte Datenvorschau": mo.ui.tabs(_preview_tabs),
                }
            )
        )

    mo.vstack(_outputs)
    return bet_iso, bet_summary, dft_pores, h2_chemi, tpr


@app.cell
def _(np, trapezoid):
    def unavailable_result(message):
        return {"available": False, "message": message}

    def _monotonic_xy(frame, x_col, y_col):
        _df = (
            frame[[x_col, y_col]]
            .dropna()
            .sort_values(x_col)
            .drop_duplicates(subset=x_col)
        )
        return _df[x_col].to_numpy(), _df[y_col].to_numpy()

    def max_hysteresis(adsorption_frame, desorption_frame):
        _x_ads, _y_ads = _monotonic_xy(
            adsorption_frame,
            "relative_pressure",
            "amount_adsorbed_cm3_stp_g",
        )
        _x_des, _y_des = _monotonic_xy(
            desorption_frame,
            "relative_pressure",
            "amount_adsorbed_cm3_stp_g",
        )

        if len(_x_ads) < 2 or len(_x_des) < 2:
            return None

        _x_min = max(_x_ads.min(), _x_des.min())
        _x_max = min(_x_ads.max(), _x_des.max())

        if _x_max <= _x_min:
            return None

        _grid = np.linspace(_x_min, _x_max, 250)
        _ads_interp = np.interp(_grid, _x_ads, _y_ads)
        _des_interp = np.interp(_grid, _x_des, _y_des)
        return float(np.max(np.abs(_des_interp - _ads_interp)))

    def linear_fit_with_r2(x_values, y_values):
        _coefficients = np.polyfit(x_values, y_values, deg=1)
        _slope, _intercept = _coefficients[0], _coefficients[1]
        _fit = _slope * x_values + _intercept
        _ss_res = np.sum((y_values - _fit) ** 2)
        _ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
        _r2 = 1.0 - _ss_res / _ss_tot if _ss_tot > 0 else 1.0
        return _slope, _intercept, _fit, _r2

    def normalize_window_length(requested_window, n_points):
        if n_points < 3:
            return n_points

        _window = int(round(requested_window))
        _window = max(_window, 3)
        _window = min(_window, n_points if n_points % 2 == 1 else n_points - 1)

        if _window % 2 == 0:
            _window -= 1

        return max(_window, 3)

    def robust_polynomial_baseline(x_values, y_values, fit_mask, degree=2, iterations=6):
        _x = np.asarray(x_values, dtype=float)
        _y = np.asarray(y_values, dtype=float)
        _mask = np.asarray(fit_mask, dtype=bool)

        _valid = _mask & np.isfinite(_x) & np.isfinite(_y)
        if _valid.sum() < degree + 1:
            _valid = np.isfinite(_x) & np.isfinite(_y)

        if _valid.sum() < degree + 1:
            return np.zeros_like(_y)

        _weights = _valid.astype(float)
        _baseline = np.zeros_like(_y)

        for _ in range(iterations):
            _coefficients = np.polyfit(
                _x[_valid],
                _y[_valid],
                deg=degree,
                w=_weights[_valid],
            )
            _baseline = np.polyval(_coefficients, _x)
            _residuals = _y - _baseline
            _scale = np.median(np.abs(_residuals[_valid]))

            if not np.isfinite(_scale) or _scale < 1e-12:
                break

            _standardized = _residuals / (6.0 * _scale)
            _weights = _valid.astype(float) / (1.0 + _standardized**2)
            _weights = np.clip(_weights, 1e-3, 1.0) * _valid

        return _baseline

    def positive_area(x_values, y_values):
        _y_positive = np.clip(y_values, 0.0, None)
        if len(_y_positive) < 2:
            return None
        return float(trapezoid(_y_positive, x_values))

    return (
        linear_fit_with_r2,
        max_hysteresis,
        normalize_window_length,
        positive_area,
        robust_polynomial_baseline,
        unavailable_result,
    )


@app.cell
def _(bet_iso, max_hysteresis, unavailable_result):
    if bet_iso is None or bet_iso.empty:
        isotherm_results = unavailable_result(
            "Keine N₂-Isothermen-Datei hochgeladen oder erfolgreich eingelesen."
        )
    else:
        _adsorption_frame = bet_iso[bet_iso["branch"] == "Adsorption"].copy()
        _desorption_frame = bet_iso[bet_iso["branch"] == "Desorption"].copy()

        if _adsorption_frame.empty or _desorption_frame.empty:
            isotherm_results = unavailable_result(
                "Adsorptions- oder Desorptionszweig konnte nicht getrennt werden."
            )
        else:
            _hysteresis_max = max_hysteresis(_adsorption_frame, _desorption_frame)
            _max_uptake = float(bet_iso["amount_adsorbed_cm3_stp_g"].max())

            if _hysteresis_max is None:
                _interpretation = (
                    "Die Zweigtrennung ist nur eingeschränkt möglich; die Isotherme dient vor allem "
                    "als qualitativer Überblick."
                )
            elif _hysteresis_max > 3.0:
                _interpretation = (
                    "Die deutliche Hysterese spricht für Mesoporosität und Kapillarkondensation "
                    "im mittleren bis hohen Relativdruckbereich."
                )
            elif _hysteresis_max > 1.0:
                _interpretation = (
                    "Es ist eine merkliche Hysterese vorhanden; das passt zu mesoporösen Anteilen "
                    "mit begrenzter Kapillarkondensation."
                )
            else:
                _interpretation = (
                    "Die Hysterese ist eher schwach; mesoporöse Effekte sind vorhanden, aber nicht dominant."
                )

            isotherm_results = {
                "available": True,
                "message": "",
                "adsorption_frame": _adsorption_frame,
                "desorption_frame": _desorption_frame,
                "max_uptake_cm3_stp_g": _max_uptake,
                "hysteresis_max_cm3_stp_g": _hysteresis_max,
                "interpretation": _interpretation,
            }
    return (isotherm_results,)


@app.cell
def _(isotherm_results, mo, pd, plt, prettify_ax, section_flags):
    mo.stop(
        not section_flags["show_isotherm"],
        mo.md("## 6. N₂-Isotherme\nAbschnitt deaktiviert."),
    )
    mo.stop(
        not isotherm_results["available"],
        mo.md(f"## 6. N₂-Isotherme\n{isotherm_results['message']}"),
    )

    _adsorption = isotherm_results["adsorption_frame"]
    _desorption = isotherm_results["desorption_frame"]
    _fig, _ax = plt.subplots()

    _ax.plot(
        _adsorption["relative_pressure"],
        _adsorption["amount_adsorbed_cm3_stp_g"],
        marker="o",
        markersize=4,
        label="Adsorption",
    )
    _ax.plot(
        _desorption["relative_pressure"],
        _desorption["amount_adsorbed_cm3_stp_g"],
        marker="s",
        markersize=4,
        label="Desorption",
    )

    prettify_ax(
        _ax,
        xlabel="Relativdruck p/p₀",
        ylabel="adsorbierte Menge / cm³(STP) g⁻¹",
        title="N₂-Isotherme",
    )
    _ax.legend()

    _metrics_table = pd.DataFrame(
        {
            "Kennzahl": [
                "max. adsorbierte Menge / cm³(STP) g⁻¹",
                "max. Hysterese / cm³(STP) g⁻¹",
            ],
            "Wert": [
                f"{isotherm_results['max_uptake_cm3_stp_g']:.2f}",
                (
                    f"{isotherm_results['hysteresis_max_cm3_stp_g']:.2f}"
                    if isotherm_results["hysteresis_max_cm3_stp_g"] is not None
                    else "nicht verfügbar"
                ),
            ],
        }
    )

    mo.vstack(
        [
            mo.md("## 6. N₂-Isotherme"),
            _fig,
            mo.ui.table(_metrics_table),
            mo.md(isotherm_results["interpretation"]),
        ]
    )
    return


@app.cell
def _(bet_summary, linear_fit_with_r2, np, unavailable_result):
    if bet_summary is None or bet_summary.empty:
        bet_results = unavailable_result(
            "Keine BET-Linearisierung hochgeladen oder erfolgreich eingelesen."
        )
    else:
        _bet_x = bet_summary["relative_pressure"].to_numpy(dtype=float)
        _bet_y = bet_summary["bet_transform_inv_g"].to_numpy(dtype=float)

        if len(_bet_x) < 2:
            bet_results = unavailable_result(
                "Zu wenige Punkte für eine BET-Linearisierung."
            )
        else:
            _bet_slope, _bet_intercept, _bet_fit, _bet_r2 = linear_fit_with_r2(
                _bet_x, _bet_y
            )

            if _bet_slope + _bet_intercept <= 0:
                bet_results = unavailable_result(
                    "Monolagenkapazität aus der Linearisierung ist nicht physikalisch sinnvoll."
                )
            else:
                _n_A = 6.02214076e23
                _sigma_n2_m2 = 16.2e-20

                _m_m_gN2_per_g = 1.0 / (_bet_slope + _bet_intercept)
                _n_m_mol_per_g = _m_m_gN2_per_g / 28.0134
                _V_m_cm3_stp_per_g = _n_m_mol_per_g * 22414.0
                _S_BET_m2_per_g = _n_m_mol_per_g * _n_A * _sigma_n2_m2
                _bet_C = (
                    1.0 + _bet_slope / _bet_intercept
                    if _bet_intercept != 0
                    else np.nan
                )

                bet_results = {
                    "available": True,
                    "message": "",
                    "bet_x": _bet_x,
                    "bet_y": _bet_y,
                    "bet_fit": _bet_fit,
                    "bet_slope": float(_bet_slope),
                    "bet_intercept": float(_bet_intercept),
                    "bet_r2": float(_bet_r2),
                    "m_m_gN2_per_g": float(_m_m_gN2_per_g),
                    "n_m_mol_per_g": float(_n_m_mol_per_g),
                    "V_m_cm3_stp_per_g": float(_V_m_cm3_stp_per_g),
                    "S_BET_m2_per_g": float(_S_BET_m2_per_g),
                    "bet_C": float(_bet_C),
                }
    return (bet_results,)


@app.cell
def _(bet_results, mo, pd, plt, prettify_ax, section_flags):
    mo.stop(
        not section_flags["show_bet"],
        mo.md("## 7. BET-Auswertung\nAbschnitt deaktiviert."),
    )
    mo.stop(
        not bet_results["available"],
        mo.md(f"## 7. BET-Auswertung\n{bet_results['message']}"),
    )

    _fig, _ax = plt.subplots()
    _ax.scatter(
        bet_results["bet_x"],
        bet_results["bet_y"],
        s=42,
        label="Messpunkte",
    )
    _ax.plot(
        bet_results["bet_x"],
        bet_results["bet_fit"],
        label="linearer Fit",
    )

    prettify_ax(
        _ax,
        xlabel="Relativdruck p/p₀",
        ylabel="1 / [W((p₀/p) - 1)] / g⁻¹",
        title="BET-Linearisierung",
    )
    _ax.legend()

    _metrics_table = pd.DataFrame(
        {
            "Kennzahl": [
                "S_BET / m² g⁻¹",
                "BET-Konstante C",
                "V_m / cm³(STP) g⁻¹",
                "R²",
            ],
            "Wert": [
                f"{bet_results['S_BET_m2_per_g']:.2f}",
                f"{bet_results['bet_C']:.2f}",
                f"{bet_results['V_m_cm3_stp_per_g']:.2f}",
                f"{bet_results['bet_r2']:.5f}",
            ],
        }
    )

    mo.vstack(
        [
            mo.md("## 7. BET-Auswertung"),
            _fig,
            mo.ui.table(_metrics_table),
        ]
    )
    return


@app.cell
def _(dft_pores, unavailable_result):
    if dft_pores is None or dft_pores.empty:
        dft_results = unavailable_result(
            "Keine DFT/NLDFT-Porendatei hochgeladen oder erfolgreich eingelesen."
        )
    else:
        _dft_distribution = dft_pores.copy()
        _dft_distribution["pore_diameter_nm"] = _dft_distribution["pore_diameter_A"] / 10.0
        _dft_distribution["dV_dD_cm3_g_nm"] = _dft_distribution["dV_d_cm3_A_g"] * 10.0
        _dft_distribution["dS_dD_m2_g_nm"] = _dft_distribution["dS_d_m2_A_g"] * 10.0

        _positive = _dft_distribution[_dft_distribution["dV_dD_cm3_g_nm"] > 0]
        if _positive.empty:
            _peak_pore_diameter_nm = None
        else:
            _peak_index = _positive["dV_dD_cm3_g_nm"].idxmax()
            _peak_pore_diameter_nm = float(
                _dft_distribution.loc[_peak_index, "pore_diameter_nm"]
            )

        dft_results = {
            "available": True,
            "message": "",
            "distribution": _dft_distribution,
            "total_pore_volume_cm3_g": float(
                _dft_distribution["cumulative_pore_volume_cm3_g"].max()
            ),
            "surface_area_m2_g": float(
                _dft_distribution["cumulative_surface_area_m2_g"].max()
            ),
            "peak_pore_diameter_nm": _peak_pore_diameter_nm,
        }
    return (dft_results,)


@app.cell
def _(dft_results, mo, pd, plt, prettify_ax, section_flags):
    mo.stop(
        not section_flags["show_dft"],
        mo.md("## 8. DFT/NLDFT-Porenstruktur\nAbschnitt deaktiviert."),
    )
    mo.stop(
        not dft_results["available"],
        mo.md(f"## 8. DFT/NLDFT-Porenstruktur\n{dft_results['message']}"),
    )

    _df = dft_results["distribution"]
    _fig, _ax = plt.subplots()

    _ax.plot(
        _df["pore_diameter_nm"],
        _df["dV_dD_cm3_g_nm"],
        color="#1f77b4",
        label="DFT/NLDFT dV/dD",
    )

    if dft_results["peak_pore_diameter_nm"] is not None:
        _ax.axvline(
            dft_results["peak_pore_diameter_nm"],
            color="#d62728",
            linestyle="--",
            linewidth=1.4,
            label="Maximum",
        )

    prettify_ax(
        _ax,
        xlabel="Porendurchmesser / nm",
        ylabel="dV/dD / cm³ g⁻¹ nm⁻¹",
        title="DFT/NLDFT-Porengrößenverteilung",
    )
    _ax.legend()

    _metrics_table = pd.DataFrame(
        {
            "Kennzahl": [
                "Gesamtporenvolumen / cm³ g⁻¹",
                "DFT-Oberfläche / m² g⁻¹",
                "DFT-Maximum / nm",
            ],
            "Wert": [
                f"{dft_results['total_pore_volume_cm3_g']:.4f}",
                f"{dft_results['surface_area_m2_g']:.2f}",
                (
                    f"{dft_results['peak_pore_diameter_nm']:.2f}"
                    if dft_results["peak_pore_diameter_nm"] is not None
                    else "nicht verfügbar"
                ),
            ],
        }
    )

    mo.vstack(
        [
            mo.md("## 8. DFT/NLDFT-Porenstruktur"),
            _fig,
            mo.ui.table(_metrics_table),
        ]
    )
    return


@app.cell
def _(bet_iso, bet_results, dft_results, np, pd, unavailable_result):
    if bet_iso is None or bet_iso.empty:
        bjh_results = unavailable_result(
            "Keine N₂-Isothermen-Daten für die Kelvin/BJH-Näherung verfügbar."
        )
    else:
        _desorption_frame = bet_iso[bet_iso["branch"] == "Desorption"].copy()
        if _desorption_frame.empty:
            bjh_results = unavailable_result("Desorptionszweig nicht verfügbar.")
        else:
            _desorption_frame = _desorption_frame[
                _desorption_frame["relative_pressure"].between(
                    0.05, 0.98, inclusive="both"
                )
            ].copy()

            if len(_desorption_frame) < 4:
                bjh_results = unavailable_result(
                    "Zu wenige Desorptionspunkte im Bereich 0.05 ≤ p/p₀ ≤ 0.98."
                )
            else:
                _relative_pressure = np.clip(
                    _desorption_frame["relative_pressure"].to_numpy(dtype=float),
                    1e-6,
                    0.999999,
                )
                _amount_adsorbed_cm3_stp_g = _desorption_frame[
                    "amount_adsorbed_cm3_stp_g"
                ].to_numpy(dtype=float)

                _r_K_m = np.abs(4.15e-10 / np.log10(_relative_pressure))
                _r_K_nm = _r_K_m * 1e9
                _t_nm = 0.354 * (-5.0 / np.log(_relative_pressure)) ** (1.0 / 3.0)
                _pore_diameter_bjh_nm = 2.0 * (_r_K_nm + _t_nm)
                _V_liq_cm3_g = _amount_adsorbed_cm3_stp_g / 22414.0 * 28.0134 / 0.808

                _bjh_distribution = pd.DataFrame(
                    {
                        "relative_pressure": _relative_pressure,
                        "pore_diameter_bjh_nm": _pore_diameter_bjh_nm,
                        "V_liq_cm3_g": _V_liq_cm3_g,
                    }
                ).sort_values("pore_diameter_bjh_nm")

                _bjh_distribution = _bjh_distribution.drop_duplicates(
                    subset="pore_diameter_bjh_nm"
                )

                if len(_bjh_distribution) < 3:
                    bjh_results = unavailable_result(
                        "BJH-Näherung liefert zu wenige eindeutige Durchmesserpunkte."
                    )
                else:
                    _bjh_distribution["dV_dD_cm3_g_nm"] = np.gradient(
                        _bjh_distribution["V_liq_cm3_g"].to_numpy(dtype=float),
                        _bjh_distribution["pore_diameter_bjh_nm"].to_numpy(dtype=float),
                    )

                    _positive = _bjh_distribution[
                        _bjh_distribution["dV_dD_cm3_g_nm"] > 0
                    ]
                    if _positive.empty:
                        _bjh_peak_nm = None
                    else:
                        _peak_index = _positive["dV_dD_cm3_g_nm"].idxmax()
                        _bjh_peak_nm = float(
                            _bjh_distribution.loc[
                                _peak_index, "pore_diameter_bjh_nm"
                            ]
                        )

                    if bet_results["available"] and dft_results["available"]:
                        _four_v_over_s_nm = (
                            4.0
                            * dft_results["total_pore_volume_cm3_g"]
                            / bet_results["S_BET_m2_per_g"]
                            * 1000.0
                        )
                    else:
                        _four_v_over_s_nm = None

                    bjh_results = {
                        "available": True,
                        "message": "",
                        "distribution": _bjh_distribution,
                        "bjh_peak_nm": _bjh_peak_nm,
                        "dft_peak_nm": (
                            dft_results["peak_pore_diameter_nm"]
                            if dft_results["available"]
                            else None
                        ),
                        "four_v_over_s_nm": _four_v_over_s_nm,
                    }
    return (bjh_results,)


@app.cell
def _(bjh_results, dft_results, mo, pd, plt, prettify_ax, section_flags):
    mo.stop(
        not section_flags["show_bjh"],
        mo.md("## 9. Kelvin/BJH-Näherung\nAbschnitt deaktiviert."),
    )
    mo.stop(
        not bjh_results["available"],
        mo.md(f"## 9. Kelvin/BJH-Näherung\n{bjh_results['message']}"),
    )

    _fig, _ax = plt.subplots()
    _bjh_df = bjh_results["distribution"]

    _ax.plot(
        _bjh_df["pore_diameter_bjh_nm"],
        _bjh_df["dV_dD_cm3_g_nm"],
        label="Kelvin/BJH-Näherung",
        color="#ff7f0e",
    )

    if dft_results["available"]:
        _dft_df = dft_results["distribution"]
        _ax.plot(
            _dft_df["pore_diameter_nm"],
            _dft_df["dV_dD_cm3_g_nm"],
            label="DFT/NLDFT",
            color="#1f77b4",
        )

    prettify_ax(
        _ax,
        xlabel="Porendurchmesser / nm",
        ylabel="dV/dD / cm³ g⁻¹ nm⁻¹",
        title="DFT/NLDFT vs. Kelvin/BJH-Näherung",
    )
    _ax.legend()

    _metrics_table = pd.DataFrame(
        {
            "Kennzahl": [
                "DFT-Maximum / nm",
                "BJH-Näherungsmaximum / nm",
                "4V/S-Mittelwert / nm",
            ],
            "Wert": [
                (
                    f"{bjh_results['dft_peak_nm']:.2f}"
                    if bjh_results["dft_peak_nm"] is not None
                    else "nicht verfügbar"
                ),
                (
                    f"{bjh_results['bjh_peak_nm']:.2f}"
                    if bjh_results["bjh_peak_nm"] is not None
                    else "nicht verfügbar"
                ),
                (
                    f"{bjh_results['four_v_over_s_nm']:.2f}"
                    if bjh_results["four_v_over_s_nm"] is not None
                    else "nicht verfügbar"
                ),
            ],
        }
    )

    mo.vstack(
        [
            mo.md("## 9. Kelvin/BJH-Näherung"),
            _fig,
            mo.ui.table(_metrics_table),
            mo.md(
                """
        DFT/NLDFT ist die Geräteauswertung und daher die verlässlichere Porenverteilung.
        Die Kelvin/BJH-Kurve ist hier als anschauliche Näherung aus dem Desorptionszweig gedacht.
        Der 4V/S-Wert ist nur ein geometrischer Mittelwert und keine vollständige Porenverteilung.
        """
            ),
        ]
    )
    return


@app.cell
def _(
    find_peaks,
    normalize_window_length,
    np,
    pd,
    positive_area,
    robust_polynomial_baseline,
    savgol_filter,
    tpr,
    tpr_settings,
    unavailable_result,
):
    if tpr is None or tpr.empty:
        tpr_results = unavailable_result(
            "Keine H₂-TPR-Datei hochgeladen oder erfolgreich eingelesen."
        )
    else:
        _temperature_C = tpr["temperature_C"].to_numpy(dtype=float)
        _raw_signal_mV = tpr["signal_mV"].to_numpy(dtype=float)
        _window_length = normalize_window_length(
            tpr_settings["smoothing_window"],
            len(tpr),
        )
        _polyorder = min(3, max(1, _window_length - 1))

        if len(tpr) >= _window_length and _window_length >= 3:
            _smooth_signal_mV = savgol_filter(
                _raw_signal_mV,
                window_length=_window_length,
                polyorder=_polyorder,
            )
        else:
            _smooth_signal_mV = _raw_signal_mV.copy()

        _baseline_mask = (
            (_temperature_C < tpr_settings["baseline_low_max_C"])
            | (
                (_temperature_C >= tpr_settings["baseline_mid_min_C"])
                & (_temperature_C <= tpr_settings["baseline_mid_max_C"])
            )
            | (_temperature_C > tpr_settings["baseline_high_min_C"])
        )

        _baseline_method = tpr_settings["baseline_method"]
        if _baseline_method == "none":
            _baseline_signal_mV = np.zeros_like(_smooth_signal_mV)
        elif _baseline_method == "polynomial":
            _baseline_signal_mV = robust_polynomial_baseline(
                _temperature_C,
                _smooth_signal_mV,
                _baseline_mask,
                degree=2,
                iterations=1,
            )
        else:
            _baseline_signal_mV = robust_polynomial_baseline(
                _temperature_C,
                _smooth_signal_mV,
                _baseline_mask,
                degree=2,
                iterations=6,
            )

        _corrected_signal_mV = _smooth_signal_mV - _baseline_signal_mV
        _prominence_threshold = max(
            np.std(_corrected_signal_mV) * tpr_settings["peak_prominence_factor"],
            np.ptp(_corrected_signal_mV) * 0.03,
            1e-6,
        )
        _peak_indices, _peak_properties = find_peaks(
            _corrected_signal_mV,
            prominence=_prominence_threshold,
            distance=max(10, len(_corrected_signal_mV) // 25),
        )

        _peak_table = pd.DataFrame(
            {
                "Peak": np.arange(1, len(_peak_indices) + 1),
                "Temperatur / °C": _temperature_C[_peak_indices],
                "Signal korrigiert / mV": _corrected_signal_mV[_peak_indices],
                "Prominenz / mV": _peak_properties.get("prominences", []),
            }
        )

        if _peak_table.empty:
            _main_peak_temperature_C = None
        else:
            _main_peak_index = _peak_table["Signal korrigiert / mV"].idxmax()
            _main_peak_temperature_C = float(
                _peak_table.loc[_main_peak_index, "Temperatur / °C"]
            )

        tpr_results = {
            "available": True,
            "message": "",
            "temperature_C": _temperature_C,
            "raw_signal_mV": _raw_signal_mV,
            "smooth_signal_mV": _smooth_signal_mV,
            "baseline_signal_mV": _baseline_signal_mV,
            "corrected_signal_mV": _corrected_signal_mV,
            "baseline_mask": _baseline_mask,
            "window_length": _window_length,
            "baseline_method": _baseline_method,
            "peak_table": _peak_table,
            "peak_indices": _peak_indices,
            "main_peak_temperature_C": _main_peak_temperature_C,
            "positive_area": positive_area(_temperature_C, _corrected_signal_mV),
        }
    return (tpr_results,)


@app.cell
def _(mo, plt, prettify_ax, section_flags, tpr_results):
    mo.stop(
        not section_flags["show_tpr"],
        mo.md("## 10. H₂-TPR\nAbschnitt deaktiviert."),
    )
    mo.stop(
        not tpr_results["available"],
        mo.md(f"## 10. H₂-TPR\n{tpr_results['message']}"),
    )

    _fig, _ax = plt.subplots()
    _ax.plot(
        tpr_results["temperature_C"],
        tpr_results["raw_signal_mV"],
        label="Rohsignal",
        color="#7f7f7f",
    )
    _ax.plot(
        tpr_results["temperature_C"],
        tpr_results["smooth_signal_mV"],
        label="geglättet",
        color="#1f77b4",
    )

    prettify_ax(
        _ax,
        xlabel="Temperatur / °C",
        ylabel="Signal / mV",
        title="H₂-TPR: Rohdaten",
    )
    _ax.legend()

    mo.vstack(
        [
            mo.md("## 10. H₂-TPR"),
            _fig,
            mo.md(
                f"""
            Glättungsfenster: `{tpr_results["window_length"]}` Punkte.  
            Baseline-Methode: `{tpr_results["baseline_method"]}`.
            """
            ),
        ]
    )
    return


@app.cell
def _(mo, pd, plt, prettify_ax, section_flags, tpr_results):
    mo.stop(
        not section_flags["show_tpr"],
        mo.md("## 11. H₂-TPR: Peakmarkierung\nAbschnitt deaktiviert."),
    )
    mo.stop(
        not tpr_results["available"],
        mo.md(f"## 11. H₂-TPR: Peakmarkierung\n{tpr_results['message']}"),
    )

    _fig, _ax = plt.subplots()
    _ax.plot(
        tpr_results["temperature_C"],
        tpr_results["corrected_signal_mV"],
        label="korrigiertes Signal",
        color="#d62728",
    )

    if tpr_results["baseline_method"] != "none":
        _ax.plot(
            tpr_results["temperature_C"],
            tpr_results["baseline_signal_mV"],
            label="Baseline",
            color="#2ca02c",
            linestyle="--",
        )

    if len(tpr_results["peak_indices"]) > 0:
        _peak_indices = tpr_results["peak_indices"]
        _ax.scatter(
            tpr_results["temperature_C"][_peak_indices],
            tpr_results["corrected_signal_mV"][_peak_indices],
            s=55,
            zorder=5,
            label="Peaks",
        )

        for _peak_number, _peak_index in enumerate(_peak_indices, start=1):
            _ax.annotate(
                f"{_peak_number}",
                xy=(
                    tpr_results["temperature_C"][_peak_index],
                    tpr_results["corrected_signal_mV"][_peak_index],
                ),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

    prettify_ax(
        _ax,
        xlabel="Temperatur / °C",
        ylabel="korrigiertes Signal / mV",
        title="H₂-TPR: geglättet und basislinienkorrigiert",
    )
    _ax.legend()

    _metrics_table = pd.DataFrame(
        {
            "Kennzahl": [
                "Anzahl Peaks",
                "Hauptpeak / °C",
                "relative positive Fläche / mV·°C",
            ],
            "Wert": [
                str(len(tpr_results["peak_table"])),
                (
                    f"{tpr_results['main_peak_temperature_C']:.1f}"
                    if tpr_results["main_peak_temperature_C"] is not None
                    else "nicht verfügbar"
                ),
                (
                    f"{tpr_results['positive_area']:.1f}"
                    if tpr_results["positive_area"] is not None
                    else "nicht verfügbar"
                ),
            ],
        }
    )

    mo.vstack(
        [
            mo.md("## 11. H₂-TPR: Peakmarkierung"),
            _fig,
            mo.ui.table(_metrics_table),
            mo.ui.table(tpr_results["peak_table"]),
            mo.md(
                """
            Ohne TCD-Kalibrierung ist daraus keine belastbare absolute H₂-Verbrauchsmenge ableitbar.
            Die Peaklagen und die Form des korrigierten Signals sind hier daher vor allem qualitativ zu interpretieren.
            """
            ),
        ]
    )
    return


@app.cell
def _(chemi_settings, h2_chemi, np, sample_params, unavailable_result):
    if h2_chemi is None or h2_chemi.empty:
        chemi_results = unavailable_result(
            "Keine H₂-Chemisorptionsdatei hochgeladen oder erfolgreich eingelesen."
        )
    else:
        _branch_frames = {}
        _max_uptake_by_branch = {}

        for _branch_label in ["combined", "weak", "strong"]:
            _branch_df = (
                h2_chemi[h2_chemi["branch_label"] == _branch_label]
                .copy()
                .sort_values("pressure_torr")
            )
            if not _branch_df.empty:
                _branch_frames[_branch_label] = _branch_df
                _max_uptake_by_branch[_branch_label] = float(
                    _branch_df["amount_adsorbed_cm3_stp_g"].max()
                )

        _selected_curve = None
        _selected_uptake_cm3_stp_g = None

        if chemi_settings["dispersion_basis"] == "combined - weak":
            if "combined" in _branch_frames and "weak" in _branch_frames:
                _combined = _branch_frames["combined"]
                _weak = _branch_frames["weak"]
                _x_min = max(
                    _combined["pressure_torr"].min(),
                    _weak["pressure_torr"].min(),
                )
                _x_max = min(
                    _combined["pressure_torr"].max(),
                    _weak["pressure_torr"].max(),
                )

                if _x_max > _x_min:
                    _grid = np.linspace(_x_min, _x_max, 250)
                    _combined_interp = np.interp(
                        _grid,
                        _combined["pressure_torr"],
                        _combined["amount_adsorbed_cm3_stp_g"],
                    )
                    _weak_interp = np.interp(
                        _grid,
                        _weak["pressure_torr"],
                        _weak["amount_adsorbed_cm3_stp_g"],
                    )
                    _selected_curve = {
                        "pressure_torr": _grid,
                        "amount_adsorbed_cm3_stp_g": _combined_interp - _weak_interp,
                    }
                    _selected_uptake_cm3_stp_g = float(
                        max(0.0, np.max(_combined_interp - _weak_interp))
                    )
        else:
            _selected_uptake_cm3_stp_g = _max_uptake_by_branch.get(
                chemi_settings["dispersion_basis"]
            )

        _max_combined = _max_uptake_by_branch.get("combined")
        _max_weak = _max_uptake_by_branch.get("weak")
        _max_strong = _max_uptake_by_branch.get("strong")

        if (
            _max_weak is not None
            and _max_strong is not None
            and (_max_weak + _max_strong) > 0
        ):
            _strong_fraction = _max_strong / (_max_weak + _max_strong)
        else:
            _strong_fraction = None

        _m_metal_g = (
            sample_params["mass_chemisorption_g"]
            * sample_params["metal_loading_wt"]
            / 100.0
        )
        _M_avg = (
            sample_params["pd_fraction"] * 106.42
            + sample_params["pt_fraction"] * 195.084
        )
        _n_metal_total = _m_metal_g / _M_avg if _M_avg > 0 else None

        if _selected_uptake_cm3_stp_g is not None:
            _V_H2_cm3_stp = (
                _selected_uptake_cm3_stp_g * sample_params["mass_chemisorption_g"]
            )
            _n_H2_ads = _V_H2_cm3_stp / 22414.0
            _n_sites = chemi_settings["sites_per_h2"] * _n_H2_ads
        else:
            _V_H2_cm3_stp = None
            _n_H2_ads = None
            _n_sites = None

        if _n_sites is not None and _n_metal_total not in (None, 0):
            _dispersion = _n_sites / _n_metal_total
        else:
            _dispersion = None

        chemi_results = {
            "available": True,
            "message": "",
            "branch_frames": _branch_frames,
            "selected_curve": _selected_curve,
            "selected_basis": chemi_settings["dispersion_basis"],
            "max_uptake_combined_cm3_stp_g": _max_combined,
            "max_uptake_weak_cm3_stp_g": _max_weak,
            "max_uptake_strong_cm3_stp_g": _max_strong,
            "strong_fraction": _strong_fraction,
            "selected_uptake_cm3_stp_g": _selected_uptake_cm3_stp_g,
            "m_metal_g": _m_metal_g,
            "M_avg": _M_avg,
            "n_metal_total": _n_metal_total,
            "n_H2_ads": _n_H2_ads,
            "n_sites": _n_sites,
            "dispersion": _dispersion,
        }
    return (chemi_results,)


@app.cell
def _(chemi_results, chemi_settings, mo, plt, prettify_ax, section_flags):
    mo.stop(
        not section_flags["show_chemi"],
        mo.md("## 12. H₂-Chemisorption\nAbschnitt deaktiviert."),
    )
    mo.stop(
        not chemi_results["available"],
        mo.md(f"## 12. H₂-Chemisorption\n{chemi_results['message']}"),
    )

    _fig, _ax = plt.subplots()
    _color_map = {
        "combined": "#1f77b4",
        "weak": "#ff7f0e",
        "strong": "#2ca02c",
    }

    for _branch_label in ["combined", "weak", "strong"]:
        if _branch_label not in chemi_results["branch_frames"]:
            continue

        _df = chemi_results["branch_frames"][_branch_label]
        _ax.plot(
            _df["pressure_torr"],
            _df["amount_adsorbed_cm3_stp_g"],
            marker="o",
            markersize=3.5,
            label=_branch_label,
            color=_color_map[_branch_label],
        )

    if chemi_settings["logx"]:
        _ax.set_xscale("log")

    prettify_ax(
        _ax,
        xlabel="Druck / Torr",
        ylabel="adsorbierte Menge / cm³(STP) g⁻¹",
        title="H₂-Chemisorption: Branch 1, 3 und 4",
    )
    _ax.legend()

    mo.vstack(
        [
            mo.md("## 12. H₂-Chemisorption"),
            _fig,
        ]
    )
    return


@app.cell
def _(chemi_results, chemi_settings, mo, pd, plt, prettify_ax, section_flags):
    mo.stop(
        not section_flags["show_chemi"],
        mo.md("## 13. H₂-Chemisorption: Kennzahlen\nAbschnitt deaktiviert."),
    )
    mo.stop(
        not chemi_results["available"],
        mo.md(f"## 13. H₂-Chemisorption: Kennzahlen\n{chemi_results['message']}"),
    )

    _fig, _ax = plt.subplots()
    _color_map = {
        "combined": "#1f77b4",
        "weak": "#ff7f0e",
        "strong": "#2ca02c",
    }

    for _branch_label in ["combined", "weak", "strong"]:
        if _branch_label not in chemi_results["branch_frames"]:
            continue

        _df = chemi_results["branch_frames"][_branch_label]
        _zoom_df = _df[_df["pressure_torr"] >= 100].copy()
        if _zoom_df.empty:
            _zoom_df = _df

        _ax.plot(
            _zoom_df["pressure_torr"],
            _zoom_df["amount_adsorbed_cm3_stp_g"],
            marker="o",
            markersize=3.2,
            label=_branch_label,
            color=_color_map[_branch_label],
        )

    if chemi_settings["logx"]:
        _ax.set_xscale("log")

    prettify_ax(
        _ax,
        xlabel="Druck / Torr",
        ylabel="adsorbierte Menge / cm³(STP) g⁻¹",
        title="H₂-Chemisorption: kompakter Vergleich",
    )
    _ax.legend()

    _metrics_table = pd.DataFrame(
        {
            "Kennzahl": [
                "max. uptake combined / cm³(STP) g⁻¹",
                "max. uptake weak / cm³(STP) g⁻¹",
                "max. uptake strong / cm³(STP) g⁻¹",
                "strong / (weak + strong)",
                f"Uptake-Basis ({chemi_results['selected_basis']}) / cm³(STP) g⁻¹",
                "geschätzte Dispersion",
            ],
            "Wert": [
                (
                    f"{chemi_results['max_uptake_combined_cm3_stp_g']:.4f}"
                    if chemi_results["max_uptake_combined_cm3_stp_g"] is not None
                    else "nicht verfügbar"
                ),
                (
                    f"{chemi_results['max_uptake_weak_cm3_stp_g']:.4f}"
                    if chemi_results["max_uptake_weak_cm3_stp_g"] is not None
                    else "nicht verfügbar"
                ),
                (
                    f"{chemi_results['max_uptake_strong_cm3_stp_g']:.4f}"
                    if chemi_results["max_uptake_strong_cm3_stp_g"] is not None
                    else "nicht verfügbar"
                ),
                (
                    f"{chemi_results['strong_fraction']:.3f}"
                    if chemi_results["strong_fraction"] is not None
                    else "nicht verfügbar"
                ),
                (
                    f"{chemi_results['selected_uptake_cm3_stp_g']:.4f}"
                    if chemi_results["selected_uptake_cm3_stp_g"] is not None
                    else "nicht verfügbar"
                ),
                (
                    f"{chemi_results['dispersion']:.3f}"
                    if chemi_results["dispersion"] is not None
                    else "nicht verfügbar"
                ),
            ],
        }
    )

    mo.vstack(
        [
            mo.md("## 13. H₂-Chemisorption: Kennzahlen"),
            _fig,
            mo.ui.table(_metrics_table),
            mo.md(
                """
            Die Dispersionsabschätzung ist nur eine grobe Näherung.
            Sie hängt stark von der Branch-Auswahl, der angenommenen H₂:Metall-Stöchiometrie
            sowie vom Pd/Pt-Verhältnis ab.
            """
            ),
        ]
    )
    return


@app.cell
def _(
    bet_results,
    bjh_results,
    chemi_results,
    dft_results,
    mo,
    pd,
    tpr_results,
):
    def _fmt(value, pattern):
        if value is None:
            return "nicht verfügbar"
        return format(value, pattern)

    _summary_table = pd.DataFrame(
        [
            {
                "Kennzahl": "BET-Oberfläche / m² g⁻¹",
                "Wert": (
                    _fmt(bet_results["S_BET_m2_per_g"], ".2f")
                    if bet_results["available"]
                    else "nicht verfügbar"
                ),
            },
            {
                "Kennzahl": "DFT-Oberfläche / m² g⁻¹",
                "Wert": (
                    _fmt(dft_results["surface_area_m2_g"], ".2f")
                    if dft_results["available"]
                    else "nicht verfügbar"
                ),
            },
            {
                "Kennzahl": "Porenvolumen / cm³ g⁻¹",
                "Wert": (
                    _fmt(dft_results["total_pore_volume_cm3_g"], ".4f")
                    if dft_results["available"]
                    else "nicht verfügbar"
                ),
            },
            {
                "Kennzahl": "DFT-Porenmaximum / nm",
                "Wert": (
                    _fmt(dft_results["peak_pore_diameter_nm"], ".2f")
                    if dft_results["available"]
                    else "nicht verfügbar"
                ),
            },
            {
                "Kennzahl": "BJH-Näherungsmaximum / nm",
                "Wert": (
                    _fmt(bjh_results["bjh_peak_nm"], ".2f")
                    if bjh_results["available"]
                    else "nicht verfügbar"
                ),
            },
            {
                "Kennzahl": "TPR-Hauptpeak / °C",
                "Wert": (
                    _fmt(tpr_results["main_peak_temperature_C"], ".1f")
                    if tpr_results["available"]
                    else "nicht verfügbar"
                ),
            },
            {
                "Kennzahl": "H₂-strong uptake / cm³(STP) g⁻¹",
                "Wert": (
                    _fmt(chemi_results["max_uptake_strong_cm3_stp_g"], ".4f")
                    if chemi_results["available"]
                    else "nicht verfügbar"
                ),
            },
            {
                "Kennzahl": "geschätzte Dispersion",
                "Wert": (
                    _fmt(chemi_results["dispersion"], ".3f")
                    if chemi_results["available"]
                    else "nicht verfügbar"
                ),
            },
        ]
    )

    _narrative_parts = []

    if bet_results["available"]:
        _surface = bet_results["S_BET_m2_per_g"]
        if _surface < 10:
            _narrative_parts.append("niedrige Oberfläche")
        elif _surface < 50:
            _narrative_parts.append("moderate Oberfläche")
        else:
            _narrative_parts.append("hohe Oberfläche")

    if dft_results["available"] and dft_results["peak_pore_diameter_nm"] is not None:
        _peak = dft_results["peak_pore_diameter_nm"]
        if 2 <= _peak <= 50:
            _narrative_parts.append(
                f"mesoporöse Struktur mit dominanten Poren um {_peak:.1f} nm"
            )
        else:
            _narrative_parts.append(
                f"Porenmaximum aus DFT bei {_peak:.1f} nm"
            )

    if tpr_results["available"] and tpr_results["main_peak_temperature_C"] is not None:
        _narrative_parts.append(
            f"TPR-Hauptreduktion bei {tpr_results['main_peak_temperature_C']:.1f} °C"
        )

    if chemi_results["available"]:
        if chemi_results["max_uptake_strong_cm3_stp_g"] is not None:
            _narrative_parts.append(
                "H₂-Chemisorption zeigt einen auswertbaren starken Uptake-Anteil"
            )
        if chemi_results["dispersion"] is not None:
            _narrative_parts.append(
                f"grob geschätzte Dispersion {chemi_results['dispersion']:.3f}"
            )

    if _narrative_parts:
        _summary_text = ", ".join(_narrative_parts) + "."
    else:
        _summary_text = (
            "Noch keine vollständige Zusammenfassung verfügbar, weil noch keine passenden "
            "Datensätze hochgeladen oder erfolgreich eingelesen wurden."
        )

    mo.vstack(
        [
            mo.md("## 14. Kompakte Zusammenfassung"),
            mo.ui.table(_summary_table),
            mo.md(_summary_text),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
