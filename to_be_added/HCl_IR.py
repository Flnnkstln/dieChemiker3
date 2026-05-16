import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exp5: IR-Spektroskopie — Rotations-Schwingungs-Banden von HCl

    Dieses Notebook führt durch eine halbautomatische Auswertung von HCl-Gasspektren:

    1. CSV-Dateien hochladen
    2. Luftspektren vergleichen
    3. Differenzspektrum \(\mathrm{HCl+Luft}-\mathrm{Luft}\) bilden
    4. P- und R-Zweig automatisch erkennen
    5. Isotopenlinien \(\mathrm{H}^{35}\mathrm{Cl}\) und \(\mathrm{H}^{37}\mathrm{Cl}\) zuordnen
    6. Intensitätsverhältnis bestimmen
    7. \(B_0\), \(B_1\), \(\Theta\) und \(R\) berechnen
    8. Tabellen und Abbildungen exportieren

    **Hinweis:** Die automatische Zuordnung ist eine Auswertehilfe. Für das Protokoll müssen die markierten Peaks und J-Zuordnungen im Plot kontrolliert werden.
    """)
    return


@app.cell
def _():
    import io
    import json
    import math
    import re
    import zipfile
    from dataclasses import dataclass
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks, savgol_filter
    from scipy.stats import t

    return curve_fit, find_peaks, go, io, mo, np, pd, zipfile


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. CSV-Dateien hochladen

    Lade mindestens ein Luftspektrum und ein HCl-Spektrum hoch. Erwartetes Format:

    ```text
    Wellenzahl_cm-1,Signal
    400.078,0.0174362
    400.257,0.0172583
    ```

    Die Dateinamen dürfen beliebig sein. Die Zuordnung erfolgt unten über Auswahlfelder.
    """)
    return


@app.cell
def _(mo):
    file_upload = mo.ui.file(
        filetypes=[".csv", ".CSV", ".txt", ".dat"],
        multiple=True,
        kind="area",
        label="CSV-/TXT-/DAT-Dateien hier ablegen oder auswählen",
        max_size=50_000_000,
    )
    file_upload
    return (file_upload,)


@app.cell
def _(io, pd):
    def read_two_column_spectrum(name: str, contents: bytes) -> pd.DataFrame:
        text = contents.decode("utf-8", errors="replace")
        # Kommentar- und Leerzeilen entfernen, aber Rohformat tolerant lassen
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lines.append(stripped)
        cleaned = "\n".join(lines)
        if not cleaned:
            raise ValueError(f"{name}: Datei enthält keine lesbaren Zahlenzeilen.")

        # Separator automatisch: Komma, Semikolon, Tab oder beliebige Leerzeichen
        df = pd.read_csv(
            io.StringIO(cleaned),
            header=None,
            comment="#",
            sep=r"[,;\t ]+",
            engine="python",
        )
        if df.shape[1] < 2:
            raise ValueError(f"{name}: Es wurden weniger als zwei Spalten gefunden.")
        df = df.iloc[:, :2].copy()
        df.columns = ["wavenumber_cm-1", "signal"]
        df["wavenumber_cm-1"] = pd.to_numeric(df["wavenumber_cm-1"], errors="coerce")
        df["signal"] = pd.to_numeric(df["signal"], errors="coerce")
        df = df.dropna().sort_values("wavenumber_cm-1").drop_duplicates("wavenumber_cm-1")
        if len(df) < 20:
            raise ValueError(f"{name}: Zu wenige gültige Datenpunkte ({len(df)}).")
        return df.reset_index(drop=True)

    def uploaded_spectra(upload_value):
        spectra = {}
        errors = []
        for item in upload_value or []:
            try:
                spectra[item.name] = read_two_column_spectrum(item.name, item.contents)
            except Exception as exc:
                errors.append(f"{item.name}: {exc}")
        return spectra, errors

    return (uploaded_spectra,)


@app.cell
def _(file_upload, mo, pd, uploaded_spectra):
    spectra, import_errors = uploaded_spectra(file_upload.value)
    spectra_summary = pd.DataFrame(
        [
            {
                "Datei": name,
                "Punkte": len(df),
                "min / cm⁻¹": float(df["wavenumber_cm-1"].min()),
                "max / cm⁻¹": float(df["wavenumber_cm-1"].max()),
                "mittlerer Schritt / cm⁻¹": float(df["wavenumber_cm-1"].diff().dropna().median()),
                "Signal min": float(df["signal"].min()),
                "Signal max": float(df["signal"].max()),
            }
            for name, df in spectra.items()
        ]
    )
    mo.md(
        "### Importübersicht\n"
        + ("Keine Dateien geladen." if spectra_summary.empty else spectra_summary.to_markdown(index=False))
        + ("\n\n**Importwarnungen:**\n" + "\n".join(f"- {e}" for e in import_errors) if import_errors else "")
    )
    return (spectra,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Messparameter und Auswerteoptionen

    - Die **Auflösung** wird für Fehlerabschätzungen verwendet.
    - Das Notebook erkennt automatisch, ob Absorptionsbanden als Maxima oder Minima erscheinen. Das kann unten überschrieben werden.
    - Für die quantitative Auswertung wählst du genau ein Luftspektrum und das dazugehörige HCl-Spektrum aus.
    """)
    return


@app.cell
def _(mo, spectra):
    file_options = list(spectra.keys())
    default_air = next((x for x in file_options if "air" in x.lower() and "hcl" not in x.lower()), file_options[0] if file_options else None)
    default_hcl = next((x for x in file_options if "hcl" in x.lower()), file_options[1] if len(file_options) > 1 else default_air)

    air_select = mo.ui.dropdown(options=file_options, value=default_air, label="Luft-/Referenzspektrum")
    hcl_select = mo.ui.dropdown(options=file_options, value=default_hcl, label="HCl-Spektrum")
    resolution_input = mo.ui.number(value=1.0, start=0.01, stop=32.0, step=0.01, label="Auflösung der quantitativen Messung / cm⁻¹")
    signal_mode = mo.ui.radio(
        options=["automatisch", "Absorbanz: Peaks nach oben", "Transmission: Absorption nach unten"],
        value="automatisch",
        label="Signalinterpretation",
    )
    hcl_min = mo.ui.number(value=2600.0, start=1000.0, stop=4000.0, step=1.0, label="HCl-Bereich min / cm⁻¹")
    hcl_max = mo.ui.number(value=3100.0, start=1000.0, stop=4500.0, step=1.0, label="HCl-Bereich max / cm⁻¹")
    prominence_factor = mo.ui.slider(start=0.2, stop=8.0, value=1.5, step=0.1, label="Peak-Empfindlichkeit: Prominence-Faktor")
    min_peak_distance = mo.ui.number(
        start=0.2,
        stop=10.0,
        step=0.1,
        value=1.0,
        label="Mindestabstand zwischen Peaks / cm⁻¹",
    )

    isotope_pair_max_sep = mo.ui.number(
        start=1.0,
        stop=50.0,
        step=0.5,
        value=12.0,
        label="Max. Abstand eines Isotopen-Dubletts / cm⁻¹",
    )
    max_j = mo.ui.number(value=10, start=4, stop=25, step=1, label="maximal verwendete J-Zahl")

    controls = mo.vstack([
        mo.hstack([air_select, hcl_select, resolution_input]),
        mo.hstack([signal_mode, hcl_min, hcl_max]),
        mo.hstack([prominence_factor, min_peak_distance, isotope_pair_max_sep, max_j]),
    ])
    controls
    return (
        air_select,
        controls,
        hcl_max,
        hcl_min,
        hcl_select,
        isotope_pair_max_sep,
        max_j,
        min_peak_distance,
        prominence_factor,
        resolution_input,
        signal_mode,
    )


@app.cell
def _(go, mo, spectra):
    fig_air = go.Figure()
    for name, df in spectra.items():
        if "air" in name.lower() and "hcl" not in name.lower():
            fig_air.add_trace(go.Scatter(x=df["wavenumber_cm-1"], y=df["signal"], mode="lines", name=name))
    fig_air.update_layout(
        title="Luftspektren bei verschiedenen Parametern",
        xaxis_title="Wellenzahl / cm⁻¹",
        yaxis_title="Signal",
        height=450,
    )
    mo.as_html(fig_air) if fig_air.data else mo.md("Keine eindeutigen Luftspektren im Dateinamen erkannt. Die quantitative Auswertung funktioniert trotzdem über die Auswahlfelder.")
    return


@app.cell
def _(curve_fit, np):
    def interpolate_to_reference(x_ref, df_other):
        return np.interp(x_ref, df_other["wavenumber_cm-1"].to_numpy(), df_other["signal"].to_numpy())

    def robust_smooth(y):
        n = len(y)
        win = max(7, min(51, (n // 80) * 2 + 1))
        if win >= n:
            win = n - 1 if (n - 1) % 2 == 1 else n - 2
        if win < 7:
            return y.copy()
        from scipy.signal import savgol_filter
        return savgol_filter(y, window_length=win, polyorder=3)

    def mad_sigma(y):
        med = np.nanmedian(y)
        return 1.4826 * np.nanmedian(np.abs(y - med))

    def local_quadratic_peak(x, y, idx, half_window=2):
        lo = max(0, idx - half_window)
        hi = min(len(x), idx + half_window + 1)
        xs = x[lo:hi]
        ys = y[lo:hi]
        if len(xs) < 3:
            return float(x[idx]), float(y[idx]), np.nan
        try:
            a, b, c = np.polyfit(xs, ys, 2)
            if a >= 0:
                return float(x[idx]), float(y[idx]), np.nan
            xv = -b / (2 * a)
            if xs.min() <= xv <= xs.max():
                yv = a * xv**2 + b * xv + c
                residual = ys - (a * xs**2 + b * xs + c)
                return float(xv), float(yv), float(np.std(residual, ddof=1) if len(residual) > 2 else np.nan)
        except Exception:
            pass
        return float(x[idx]), float(y[idx]), np.nan

    def gaussian(x, amp, mu, sigma, offset, slope):
        return offset + slope * (x - mu) + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def refine_gaussian(x, y, peak_x, resolution_cm, window_cm=4.0):
        mask = (x >= peak_x - window_cm) & (x <= peak_x + window_cm)
        xs = x[mask]
        ys = y[mask]
        if len(xs) < 7:
            return peak_x, np.nan, np.nan
        amp0 = float(np.nanmax(ys) - np.nanmedian(ys))
        sigma0 = max(resolution_cm / 2.355, (xs.max() - xs.min()) / 8)
        p0 = [amp0, peak_x, sigma0, float(np.nanmedian(ys)), 0.0]
        try:
            popt, pcov = curve_fit(gaussian, xs, ys, p0=p0, maxfev=10000)
            mu = float(popt[1])
            sigma_mu = float(np.sqrt(abs(pcov[1, 1]))) if pcov.shape == (5, 5) else np.nan
            area = float(abs(popt[0] * abs(popt[2]) * np.sqrt(2 * np.pi)))
            if xs.min() <= mu <= xs.max():
                return mu, sigma_mu, area
        except Exception:
            pass
        return peak_x, np.nan, np.nan

    return (
        interpolate_to_reference,
        local_quadratic_peak,
        mad_sigma,
        refine_gaussian,
        robust_smooth,
    )


@app.cell
def _(air_select, hcl_select, interpolate_to_reference, pd, spectra):
    def make_difference_spectrum(spectra_dict, air_name, hcl_name):
        if not air_name or not hcl_name or air_name not in spectra_dict or hcl_name not in spectra_dict:
            return pd.DataFrame(columns=["wavenumber_cm-1", "air", "hcl", "difference"])
        air_df = spectra_dict[air_name]
        hcl_df = spectra_dict[hcl_name]
        x_min = max(air_df["wavenumber_cm-1"].min(), hcl_df["wavenumber_cm-1"].min())
        x_max = min(air_df["wavenumber_cm-1"].max(), hcl_df["wavenumber_cm-1"].max())
        ref = hcl_df[(hcl_df["wavenumber_cm-1"] >= x_min) & (hcl_df["wavenumber_cm-1"] <= x_max)].copy()
        x = ref["wavenumber_cm-1"].to_numpy()
        air_interp = interpolate_to_reference(x, air_df)
        out = pd.DataFrame({
            "wavenumber_cm-1": x,
            "air": air_interp,
            "hcl": ref["signal"].to_numpy(),
        })
        out["difference"] = out["hcl"] - out["air"]
        return out

    diff_df = make_difference_spectrum(spectra, air_select.value, hcl_select.value)
    return (diff_df,)


@app.cell
def _(diff_df, go, hcl_max, hcl_min, mo):
    fig_diff = go.Figure()
    if not diff_df.empty:
        fig_diff.add_trace(go.Scatter(x=diff_df["wavenumber_cm-1"], y=diff_df["air"], mode="lines", name="Luft"))
        fig_diff.add_trace(go.Scatter(x=diff_df["wavenumber_cm-1"], y=diff_df["hcl"], mode="lines", name="HCl + Luft"))
        fig_diff.add_trace(go.Scatter(x=diff_df["wavenumber_cm-1"], y=diff_df["difference"], mode="lines", name="Differenz: HCl - Luft"))
        fig_diff.add_vrect(x0=hcl_min.value, x1=hcl_max.value, opacity=0.12, line_width=0, annotation_text="HCl-Auswertebereich")
    fig_diff.update_layout(title="Rohspektren und Differenzspektrum", xaxis_title="Wellenzahl / cm⁻¹", yaxis_title="Signal", height=520)
    mo.as_html(fig_diff) if fig_diff.data else mo.md("Bitte Dateien hochladen und Luft/HCl-Spektrum auswählen.")
    return


@app.cell
def _(
    diff_df,
    find_peaks,
    hcl_max,
    hcl_min,
    isotope_pair_max_sep,
    local_quadratic_peak,
    mad_sigma,
    max_j,
    min_peak_distance,
    np,
    pd,
    prominence_factor,
    refine_gaussian,
    resolution_input,
    robust_smooth,
    signal_mode,
):
    def detect_absorption_orientation(y_raw, mode):
        if mode.startswith("Absorbanz"):
            return 1
        if mode.startswith("Transmission"):
            return -1

        # Automatisch: Absorptionslinien sind die Richtung mit stärkerer Peak-Struktur.
        y0 = y_raw - robust_smooth(y_raw)
        return 1 if abs(np.nanmax(y0)) >= abs(np.nanmin(y0)) else -1


    def build_peak_table(diff, xmin, xmax, mode, resolution_cm, prominence_fac, min_dist_cm):
        if diff.empty:
            return pd.DataFrame()

        reg = diff[
            (diff["wavenumber_cm-1"] >= xmin)
            & (diff["wavenumber_cm-1"] <= xmax)
        ].copy()

        if len(reg) < 20:
            return pd.DataFrame()

        x = reg["wavenumber_cm-1"].to_numpy()
        y_raw = reg["difference"].to_numpy()

        orientation = detect_absorption_orientation(y_raw, mode)
        y_abs = orientation * y_raw
        y_base = robust_smooth(y_abs)
        noise = mad_sigma(y_abs - y_base)

        dx = np.nanmedian(np.diff(x))
        distance_pts = max(1, int(round(min_dist_cm / dx)))
        prominence = max(noise * prominence_fac, np.ptp(y_abs) * 0.01)

        idx, props = find_peaks(
            y_abs,
            prominence=prominence,
            distance=distance_pts,
        )

        rows = []

        for k, i in enumerate(idx):
            x_quad, height_quad, _ = local_quadratic_peak(
                x,
                y_abs,
                int(i),
                half_window=2,
            )

            x_g, sx_g, area_g = refine_gaussian(
                x,
                y_abs,
                x_quad,
                resolution_cm,
                window_cm=max(3.0 * resolution_cm, 2.5),
            )

            pos = x_g if np.isfinite(x_g) else x_quad
            sigma_pos = (
                sx_g
                if np.isfinite(sx_g) and sx_g > 0
                else resolution_cm / np.sqrt(12)
            )

            rows.append(
                {
                    "peak_id": k + 1,
                    "wavenumber_cm-1": pos,
                    "sigma_cm-1": sigma_pos,
                    "height": float(np.interp(pos, x, y_abs)),
                    "area": area_g if np.isfinite(area_g) else np.nan,
                    "prominence": float(
                        props.get("prominences", [np.nan] * len(idx))[k]
                    ),
                }
            )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "peak_id",
                    "wavenumber_cm-1",
                    "sigma_cm-1",
                    "height",
                    "area",
                    "prominence",
                ]
            )

        return (
            pd.DataFrame(rows)
            .sort_values("wavenumber_cm-1")
            .reset_index(drop=True)
        )


    def pair_isotopes(peaks, center, max_sep=4.0, max_j_value=12):
        columns = [
            "peak_id",
            "wavenumber_cm-1",
            "sigma_cm-1",
            "height",
            "area",
            "prominence",
            "branch",
            "J",
            "isotope",
            "pair_id",
        ]

        if peaks is None or peaks.empty:
            return pd.DataFrame(columns=columns)

        p = peaks.copy().sort_values("wavenumber_cm-1").reset_index(drop=True)

        left = p[p["wavenumber_cm-1"] < center].copy()
        right = p[p["wavenumber_cm-1"] > center].copy()

        rows = []

        def add_branch_pairs(branch_df, branch_name):
            if branch_df.empty:
                return

            # P-Zweig: vom Bandenzentrum nach links: P(1), P(2), ...
            # R-Zweig: vom Bandenzentrum nach rechts: R(0), R(1), ...
            if branch_name == "P":
                ordered = branch_df.sort_values(
                    "wavenumber_cm-1",
                    ascending=False,
                ).reset_index(drop=True)
                first_j = 1
            else:
                ordered = branch_df.sort_values(
                    "wavenumber_cm-1",
                    ascending=True,
                ).reset_index(drop=True)
                first_j = 0

            used = set()
            j_counter = first_j

            i = 0
            while i < len(ordered) and j_counter <= max_j_value:
                if i in used:
                    i += 1
                    continue

                current = ordered.iloc[i]

                best_k = None
                best_dist = None

                for k in range(i + 1, len(ordered)):
                    if k in used:
                        continue

                    candidate = ordered.iloc[k]
                    dist = abs(
                        float(candidate["wavenumber_cm-1"])
                        - float(current["wavenumber_cm-1"])
                    )

                    if dist <= max_sep and (
                        best_dist is None or dist < best_dist
                    ):
                        best_k = k
                        best_dist = dist

                if best_k is None:
                    row = current.to_dict()
                    row.update(
                        {
                            "branch": branch_name,
                            "J": int(j_counter),
                            "isotope": "H35Cl",
                            "pair_id": f"{branch_name}{j_counter}_single",
                        }
                    )
                    rows.append(row)

                    used.add(i)
                    i += 1
                    j_counter += 1
                    continue

                partner = ordered.iloc[best_k]

                pair = (
                    pd.DataFrame([current, partner])
                    .sort_values("wavenumber_cm-1")
                    .reset_index(drop=True)
                )

                low_nu = pair.iloc[0].to_dict()
                high_nu = pair.iloc[1].to_dict()

                # H35Cl liegt wegen kleinerer reduzierter Masse näherungsweise
                # bei etwas höheren Wellenzahlen als H37Cl.
                low_nu.update(
                    {
                        "branch": branch_name,
                        "J": int(j_counter),
                        "isotope": "H37Cl",
                        "pair_id": f"{branch_name}{j_counter}",
                    }
                )

                high_nu.update(
                    {
                        "branch": branch_name,
                        "J": int(j_counter),
                        "isotope": "H35Cl",
                        "pair_id": f"{branch_name}{j_counter}",
                    }
                )

                rows.extend([low_nu, high_nu])

                used.add(i)
                used.add(best_k)

                i += 1
                j_counter += 1

        add_branch_pairs(left, "P")
        add_branch_pairs(right, "R")

        if not rows:
            return pd.DataFrame(columns=columns)

        out = pd.DataFrame(rows)

        for col in columns:
            if col not in out.columns:
                out[col] = np.nan

        return (
            out[columns]
            .sort_values(["isotope", "branch", "J"])
            .reset_index(drop=True)
        )


    peak_table_auto = build_peak_table(
        diff_df,
        hcl_min.value,
        hcl_max.value,
        signal_mode.value,
        float(resolution_input.value),
        float(prominence_factor.value),
        float(min_peak_distance.value),
    )

    # Bandenzentrum automatisch als größte Lücke im HCl-Bereich schätzen;
    # fallback: Literaturbereich Mitte.
    if not peak_table_auto.empty and len(peak_table_auto) > 4:
        ws = peak_table_auto["wavenumber_cm-1"].to_numpy()
        gaps = np.diff(ws)
        center_est = (
            float((ws[np.argmax(gaps)] + ws[np.argmax(gaps) + 1]) / 2)
            if len(gaps)
            else 2886.0
        )
    else:
        center_est = 2886.0

    assigned_auto = pair_isotopes(
        peak_table_auto,
        center_est,
        float(isotope_pair_max_sep.value),
        int(max_j.value),
    )

    assignment_warning = ""

    if assigned_auto.empty:
        assignment_warning = (
            "Keine automatische P-/R-/Isotopen-Zuordnung möglich. "
            "Prüfe HCl-Bereich, Peak-Prominence, Signalrichtung und Isotopen-Maximalabstand."
        )
    return assigned_auto, center_est, peak_table_auto


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Automatische Peakliste und manuelle Korrektur

    Kontrolliere die Zuordnung im Plot. Falls die automatische Peak-/J-Zuordnung nicht stimmt, kannst du unten eine eigene Tabelle einfügen.

    Erwartete Spalten für manuelle Korrektur:

    ```csv
    branch,J,isotope,wavenumber_cm-1,sigma_cm-1,height,area
    P,1,H35Cl,2860.12,0.29,0.123,0.456
    P,1,H37Cl,2858.03,0.29,0.041,0.153
    R,0,H35Cl,2904.50,0.29,0.120,0.450
    R,0,H37Cl,2902.40,0.29,0.040,0.150
    ```

    Wenn das Feld leer bleibt, wird die automatische Tabelle verwendet.
    """)
    return


@app.cell
def _(mo):
    manual_peak_csv = mo.ui.text_area(
        value="",
        label="Optionale manuelle Peak-/J-Tabelle als CSV",
        rows=8,
        full_width=True,
    )
    manual_peak_csv
    return (manual_peak_csv,)


@app.cell
def _(assigned_auto, io, manual_peak_csv, np, pd, resolution_input):
    def parse_manual_assignment(text, fallback_sigma):
        if not text or not text.strip():
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(text.strip()))
        required = {"branch", "J", "isotope", "wavenumber_cm-1"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Manuelle Tabelle ohne Pflichtspalten: {missing}")
        if "sigma_cm-1" not in df.columns:
            df["sigma_cm-1"] = fallback_sigma
        for optional in ["height", "area"]:
            if optional not in df.columns:
                df[optional] = np.nan
        df["J"] = df["J"].astype(int)
        return df

    manual_error = ""
    try:
        manual_assigned = parse_manual_assignment(manual_peak_csv.value, float(resolution_input.value) / np.sqrt(12))
    except Exception as exc:
        manual_assigned = pd.DataFrame()
        manual_error = str(exc)
    assigned = manual_assigned if not manual_assigned.empty else assigned_auto.copy()
    return assigned, manual_error


@app.cell
def _(
    assigned,
    center_est,
    controls,
    diff_df,
    go,
    hcl_max,
    hcl_min,
    manual_error,
    mo,
    peak_table_auto,
):
    fig_peaks = go.Figure()

    if not diff_df.empty:
        reg = diff_df[
            (diff_df["wavenumber_cm-1"] >= hcl_min.value)
            & (diff_df["wavenumber_cm-1"] <= hcl_max.value)
        ].copy()

        fig_peaks.add_trace(
            go.Scatter(
                x=reg["wavenumber_cm-1"],
                y=reg["difference"],
                mode="lines",
                name="Differenzspektrum",
            )
        )

        def y_at_peak(x_values):
            return [
                float(
                    reg.iloc[
                        (reg["wavenumber_cm-1"] - float(x)).abs().argmin()
                    ]["difference"]
                )
                for x in x_values
            ]

        if not peak_table_auto.empty:
            auto_y = y_at_peak(peak_table_auto["wavenumber_cm-1"])
            fig_peaks.add_trace(
                go.Scatter(
                    x=peak_table_auto["wavenumber_cm-1"],
                    y=auto_y,
                    mode="markers",
                    name="gefundene Peaks",
                    marker={"symbol": "x", "size": 9},
                )
            )

        if not assigned.empty:
            for _plot_isotope in sorted(assigned["isotope"].unique()):
                _plot_sub = assigned[assigned["isotope"] == _plot_isotope].copy()
                _plot_y = y_at_peak(_plot_sub["wavenumber_cm-1"])
                _plot_labels = [
                    f"{row.branch}({int(row.J)}) {row.isotope}"
                    for row in _plot_sub.itertuples()
                ]

                fig_peaks.add_trace(
                    go.Scatter(
                        x=_plot_sub["wavenumber_cm-1"],
                        y=_plot_y,
                        mode="markers+text",
                        text=_plot_labels,
                        textposition="bottom center",
                        textfont={"size": 12},
                        name=_plot_isotope,
                        marker={"size": 8},
                    )
                )

        fig_peaks.add_vline(
            x=center_est,
            line_dash="dash",
            annotation_text="geschätztes Bandenzentrum/Q-Lücke",
        )

    fig_peaks.update_layout(
        title="HCl-Differenzspektrum mit Peak- und J-Zuordnung",
        xaxis_title="Wellenzahl / cm⁻¹",
        yaxis_title="Differenzsignal",
        height=650,
    )

    display = mo.vstack(
        [
            mo.as_html(fig_peaks)
            if fig_peaks.data
            else mo.md("Noch kein Differenzspektrum verfügbar."),
            mo.md(
                (
                    "**Manuelle Tabelle wird verwendet.**"
                    if not assigned.empty and manual_error == ""
                    else ""
                )
                + (
                    f"\n\n**Fehler in manueller Tabelle:** {manual_error}"
                    if manual_error
                    else ""
                )
            ),
            mo.md(
                "### Aktive Peak-/J-Tabelle\n"
                + (
                    assigned.to_markdown(index=False)
                    if not assigned.empty
                    else "Keine Peaks zugeordnet."
                )
            ),

        ]
    )

    mo.vstack([
        display,
        controls,
    ])
    return


@app.cell
def _(np, pd):
    def weighted_linear_fit(x, y, sy):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        sy = np.asarray(sy, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(sy) & (sy > 0)
        x, y, sy = x[mask], y[mask], sy[mask]
        n = len(x)
        if n < 3:
            return None
        X = np.vstack([x, np.ones_like(x)]).T
        W = np.diag(1 / sy**2)
        beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
        cov = np.linalg.inv(X.T @ W @ X)
        yfit = X @ beta
        chi2 = float(np.sum(((y - yfit) / sy) ** 2))
        dof = max(1, n - 2)
        cov_scaled = cov * chi2 / dof
        slope, intercept = beta
        slope_err, intercept_err = np.sqrt(np.diag(cov_scaled))
        return {
            "n": n,
            "slope": float(slope),
            "slope_err": float(slope_err),
            "intercept": float(intercept),
            "intercept_err": float(intercept_err),
            "chi2_red": chi2 / dof,
            "x": x,
            "y": y,
            "sy": sy,
            "yfit": yfit,
        }

    def make_fit_tables(assigned):
        rows_b1 = []
        rows_b0 = []
        if assigned.empty:
            return pd.DataFrame(), pd.DataFrame()
        for isotope in sorted(assigned["isotope"].unique()):
            sub = assigned[assigned["isotope"] == isotope]
            def get(branch, J):
                m = sub[(sub["branch"] == branch) & (sub["J"] == J)]
                if m.empty:
                    return None
                return m.iloc[0]
            Js = sorted(set(sub["J"].astype(int)))
            for J in Js:
                r = get("R", J); p = get("P", J)
                if r is not None and p is not None and J >= 1:
                    rows_b1.append({
                        "isotope": isotope, "J": J, "x=2J+1": 2 * J + 1,
                        "nu_R": r["wavenumber_cm-1"], "nu_P": p["wavenumber_cm-1"],
                        "y=nu_R-nu_P": r["wavenumber_cm-1"] - p["wavenumber_cm-1"],
                        "sigma_y": float(np.hypot(r["sigma_cm-1"], p["sigma_cm-1"])),
                    })
            for J in range(1, max(Js) + 1 if Js else 1):
                r = get("R", J - 1); p = get("P", J + 1)
                if r is not None and p is not None:
                    rows_b0.append({
                        "isotope": isotope, "J": J, "x=2J+1": 2 * J + 1,
                        "nu_R(J-1)": r["wavenumber_cm-1"], "nu_P(J+1)": p["wavenumber_cm-1"],
                        "y=nu_R(J-1)-nu_P(J+1)": r["wavenumber_cm-1"] - p["wavenumber_cm-1"],
                        "sigma_y": float(np.hypot(r["sigma_cm-1"], p["sigma_cm-1"])),
                    })
        return pd.DataFrame(rows_b1), pd.DataFrame(rows_b0)

    return make_fit_tables, weighted_linear_fit


@app.cell
def _(assigned, make_fit_tables, pd, weighted_linear_fit):
    b1_table, b0_table = make_fit_tables(assigned)
    fit_results = []

    _fit_isotopes = sorted(
        set(list(b1_table.get("isotope", [])) + list(b0_table.get("isotope", [])))
    )

    for _fit_isotope in _fit_isotopes:
        for _fit_label, _fit_table, _fit_ycol in [
            ("B1", b1_table, "y=nu_R-nu_P"),
            ("B0", b0_table, "y=nu_R(J-1)-nu_P(J+1)"),
        ]:
            _fit_sub = (
                _fit_table[_fit_table["isotope"] == _fit_isotope]
                if not _fit_table.empty
                else pd.DataFrame()
            )

            _fit_result = (
                weighted_linear_fit(
                    _fit_sub["x=2J+1"],
                    _fit_sub[_fit_ycol],
                    _fit_sub["sigma_y"],
                )
                if len(_fit_sub) >= 3
                else None
            )

            if _fit_result is not None:
                fit_results.append(
                    {
                        "isotope": _fit_isotope,
                        "Konstante": _fit_label,
                        "n": _fit_result["n"],
                        "Steigung 2B / cm⁻¹": _fit_result["slope"],
                        "σ(Steigung) / cm⁻¹": _fit_result["slope_err"],
                        "B / cm⁻¹": _fit_result["slope"] / 2,
                        "σ(B) / cm⁻¹": _fit_result["slope_err"] / 2,
                        "d / cm⁻¹": _fit_result["intercept"],
                        "σ(d) / cm⁻¹": _fit_result["intercept_err"],
                        "|d| < 2σd?": abs(_fit_result["intercept"])
                        < 2 * _fit_result["intercept_err"],
                        "χ²_red": _fit_result["chi2_red"],
                        "_fit": _fit_result,
                    }
                )

    fit_summary = pd.DataFrame(
        [{k: v for k, v in row.items() if k != "_fit"} for row in fit_results]
    )
    return b0_table, b1_table, fit_results, fit_summary


@app.cell
def _(b0_table, b1_table, fit_summary, mo):
    mo.md(
        "## 4. Fit-Tabellen\n\n"
        "### Tabelle für B₁\n"
        + (b1_table.to_markdown(index=False) if not b1_table.empty else "Keine B₁-Tabelle möglich.")
        + "\n\n### Tabelle für B₀\n"
        + (b0_table.to_markdown(index=False) if not b0_table.empty else "Keine B₀-Tabelle möglich.")
        + "\n\n### Fit-Ergebnisse\n"
        + (fit_summary.to_markdown(index=False) if not fit_summary.empty else "Noch keine linearen Fits möglich. Prüfe Peak-/J-Zuordnung.")
    )
    return


@app.cell
def _(fit_results, go, mo, np):
    fig_fits = go.Figure()
    for res in fit_results:
        fit = res["_fit"]
        label = f"{res['isotope']} {res['Konstante']}"
        fig_fits.add_trace(go.Scatter(
            x=fit["x"], y=fit["y"], error_y={"type": "data", "array": fit["sy"]},
            mode="markers", name=label + " Daten"
        ))
        xs = np.linspace(min(fit["x"]), max(fit["x"]), 100)
        ys = res["Steigung 2B / cm⁻¹"] * xs + res["d / cm⁻¹"]
        fig_fits.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=label + " Fit"))
    fig_fits.update_layout(title="Lineare Regressionen für B₀ und B₁", xaxis_title="2J + 1", yaxis_title="Differenz der Linienpositionen / cm⁻¹", height=620)
    mo.as_html(fig_fits) if fig_fits.data else mo.md("Keine Fitplots verfügbar.")
    return


@app.cell
def _(fit_summary, np, pd):
    # Konstanten und Literatur-/Referenzwerte
    h = 6.62607015e-34          # J s
    c = 2.99792458e8            # m/s
    u = 1.66053906660e-27       # kg

    m_H = 1.007825 * u
    m_35Cl = 34.968853 * u
    m_37Cl = 36.965903 * u

    masses = {
        "H35Cl": m_H * m_35Cl / (m_H + m_35Cl),
        "H37Cl": m_H * m_37Cl / (m_H + m_37Cl),
    }

    # Näherungs-/Vergleichswerte; im Protokoll mit eigener Quelle belegen.
    literature = pd.DataFrame(
        [
            {
                "isotope": "H35Cl",
                "Konstante": "B0",
                "B_lit_cm-1": 10.5934,
                "R_lit_A": 1.2746,
                "Haeufigkeit_Cl_%": 75.78,
            },
            {
                "isotope": "H35Cl",
                "Konstante": "B1",
                "B_lit_cm-1": 10.1360,
                "R_lit_A": 1.2746,
                "Haeufigkeit_Cl_%": 75.78,
            },
            {
                "isotope": "H37Cl",
                "Konstante": "B0",
                "B_lit_cm-1": 10.3520,
                "R_lit_A": 1.2746,
                "Haeufigkeit_Cl_%": 24.22,
            },
            {
                "isotope": "H37Cl",
                "Konstante": "B1",
                "B_lit_cm-1": 9.9030,
                "R_lit_A": 1.2746,
                "Haeufigkeit_Cl_%": 24.22,
            },
        ]
    )


    def calc_molecular_results(fit_df):
        result_columns = [
            "isotope",
            "Konstante",
            "B / cm⁻¹",
            "σ(B) / cm⁻¹",
            "Theta / kg m²",
            "σ(Theta) / kg m²",
            "R / Å",
            "σ(R) / Å",
            "B_lit_cm-1",
            "R_lit_A",
            "Haeufigkeit_Cl_%",
            "Abw. B / %",
            "Abw. R / %",
        ]

        if fit_df is None or fit_df.empty:
            return pd.DataFrame(columns=result_columns)

        rows = []

        for _, row in fit_df.iterrows():
            isotope_name = row["isotope"]
            constant_name = row["Konstante"]

            if isotope_name not in masses:
                continue

            B_cm = float(row["B / cm⁻¹"])
            sB_cm = float(row["σ(B) / cm⁻¹"])

            B_m = B_cm * 100.0
            sB_m = sB_cm * 100.0

            theta = h / (8 * np.pi**2 * c * B_m)
            stheta = theta * sB_m / B_m

            R = np.sqrt(theta / masses[isotope_name])
            sR = 0.5 * R * stheta / theta

            rows.append(
                {
                    "isotope": isotope_name,
                    "Konstante": constant_name,
                    "B / cm⁻¹": B_cm,
                    "σ(B) / cm⁻¹": sB_cm,
                    "Theta / kg m²": theta,
                    "σ(Theta) / kg m²": stheta,
                    "R / Å": R * 1e10,
                    "σ(R) / Å": sR * 1e10,
                }
            )

        if not rows:
            return pd.DataFrame(columns=result_columns)

        out = pd.DataFrame(rows)

        out = out.merge(
            literature,
            on=["isotope", "Konstante"],
            how="left",
        )

        out["Abw. B / %"] = (
            100.0 * (out["B / cm⁻¹"] - out["B_lit_cm-1"]) / out["B_lit_cm-1"]
        )

        out["Abw. R / %"] = (
            100.0 * (out["R / Å"] - out["R_lit_A"]) / out["R_lit_A"]
        )

        return out


    molecular_results = calc_molecular_results(fit_summary)
    return (molecular_results,)


@app.cell
def _(mo, molecular_results):
    mo.md(
        "## 5. Molekulare Ergebnisse: B, Θ und R\n\n"
        + (molecular_results.to_markdown(index=False, floatfmt=".6g") if not molecular_results.empty else "Noch keine molekularen Ergebnisse verfügbar.")
        + "\n\n**Standardform:** Gib im Protokoll z. B. `B = (10.59 ± 0.03) cm⁻¹` an. Für sehr kleine Θ-Werte wissenschaftliche Schreibweise verwenden."
    )
    return


@app.cell
def _(assigned, mo, np, pd):
    def isotope_intensity_ratios(assigned):
        rows = []
        if assigned.empty:
            return pd.DataFrame()
        keys = assigned.groupby(["branch", "J"])
        for (branch, J), g in keys:
            if {"H35Cl", "H37Cl"}.issubset(set(g["isotope"])):
                a = g[g["isotope"] == "H35Cl"].iloc[0]
                b = g[g["isotope"] == "H37Cl"].iloc[0]
                row = {"branch": branch, "J": int(J)}
                for metric in ["height", "area"]:
                    va = float(a[metric]) if metric in a and np.isfinite(a[metric]) else np.nan
                    vb = float(b[metric]) if metric in b and np.isfinite(b[metric]) else np.nan
                    row[f"I35/I37 aus {metric}"] = va / vb if vb and np.isfinite(vb) else np.nan
                    row[f"H35 Anteil aus {metric} / %"] = 100 * va / (va + vb) if np.isfinite(va) and np.isfinite(vb) and (va + vb) != 0 else np.nan
                    row[f"H37 Anteil aus {metric} / %"] = 100 * vb / (va + vb) if np.isfinite(va) and np.isfinite(vb) and (va + vb) != 0 else np.nan
                rows.append(row)
        return pd.DataFrame(rows)

    ratio_table = isotope_intensity_ratios(assigned)
    mo.md(
        "## 6. Isotopen-Intensitätsverhältnis\n\n"
        + (ratio_table.to_markdown(index=False, floatfmt=".4g") if not ratio_table.empty else "Keine vollständigen Isotopenpaare gefunden.")
        + "\n\nZum Vergleich: natürliche Chlorhäufigkeit ungefähr 75.8 % ³⁵Cl und 24.2 % ³⁷Cl, also etwa 3:1."
    )
    return (ratio_table,)


@app.cell
def _(fit_summary, mo, molecular_results):
    check_warnings = []

    if fit_summary is not None and not fit_summary.empty:
        for _, _check_row in fit_summary.iterrows():
            _check_n = int(_check_row["n"])
            _check_ok_d = bool(_check_row["|d| < 2σd?"])
            _check_name = f"{_check_row['isotope']} {_check_row['Konstante']}"

            if _check_n < 6:
                check_warnings.append(
                    f"{_check_name}: nur {_check_n} Fitpunkte; "
                    "gefordert sind mindestens 6 J-Werte."
                )

            if not _check_ok_d:
                check_warnings.append(
                    f"{_check_name}: Achsenabschnitt d ist nicht mit 0 verträglich "
                    "→ J-Zuordnung/Überlagerungen prüfen."
                )

    if molecular_results is not None and not molecular_results.empty:
        for _check_iso in sorted(molecular_results["isotope"].unique()):
            _check_sub = molecular_results[
                molecular_results["isotope"] == _check_iso
            ]

            if {"B0", "B1"}.issubset(set(_check_sub["Konstante"])):
                _check_b0 = float(
                    _check_sub.loc[
                        _check_sub["Konstante"] == "B0",
                        "B / cm⁻¹",
                    ].iloc[0]
                )
                _check_b1 = float(
                    _check_sub.loc[
                        _check_sub["Konstante"] == "B1",
                        "B / cm⁻¹",
                    ].iloc[0]
                )

                if not _check_b0 > _check_b1:
                    check_warnings.append(
                        f"{_check_iso}: Erwartung B₀ > B₁ nicht erfüllt; "
                        "Peakzuordnung prüfen."
                    )

    check_text = (
        "Keine Warnungen."
        if not check_warnings
        else "\n".join(["- " + _check_warning for _check_warning in check_warnings])
    )

    check_display = mo.md(
        "## 7. Plausibilitätschecks\n\n" + check_text
    ).callout(kind="warn" if check_warnings else "success")

    check_display
    return


@app.cell
def _(
    b0_table,
    b1_table,
    fit_summary,
    io,
    molecular_results,
    pd,
    peak_table_auto,
    ratio_table,
    zipfile,
):
    def dataframe_to_csv_bytes(df):
        return df.to_csv(index=False).encode("utf-8")

    def build_results_zip():
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            tables = {
                "01_gefundene_peaks.csv": peak_table_auto,
                "02_fit_tabelle_B1.csv": b1_table,
                "03_fit_tabelle_B0.csv": b0_table,
                "04_fit_ergebnisse.csv": fit_summary.drop(columns=[c for c in fit_summary.columns if c.startswith("_")], errors="ignore"),
                "05_molekulare_ergebnisse.csv": molecular_results,
                "06_isotopenverhaeltnis.csv": ratio_table,
            }
            for name, df in tables.items():
                zf.writestr(name, dataframe_to_csv_bytes(df if isinstance(df, pd.DataFrame) else pd.DataFrame()))
            readme = """Exp5 IR-Auswertung HCl\n\nDiese ZIP-Datei enthält die aus dem marimo-Notebook exportierten Tabellen.\nPlots können im Notebook über die Plotly-Kamera bzw. Browserfunktionen exportiert werden.\n\nWichtig: Literaturwerte im Protokoll mit eigener Quelle belegen.\n"""
            zf.writestr("README.txt", readme.encode("utf-8"))
        return bio.getvalue()

    zip_bytes = build_results_zip()
    return (zip_bytes,)


@app.cell
def _(mo, zip_bytes):
    try:
        download_widget = mo.download(data=zip_bytes, filename="exp5_ir_hcl_ergebnisse.zip", label="Ergebnistabellen als ZIP herunterladen")
    except Exception:
        download_widget = mo.md("Download-Widget in dieser marimo-Version nicht verfügbar. Die Tabellen können direkt aus den angezeigten Tabellen kopiert werden.")
    download_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Was im Protokoll diskutiert werden sollte

    - **Messparameter:** Mehr Scans verbessern normalerweise das Signal-Rausch-Verhältnis; höhere spektrale Auflösung trennt nahe Linien besser, kostet aber Messzeit und Signalqualität.
    - **P-/R-Zweig:** P-Zweig liegt bei kleineren Wellenzahlen, R-Zweig bei größeren Wellenzahlen. Ein Q-Zweig fehlt für den hier betrachteten Übergang.
    - **Achsenabschnitt d:** Bei korrekter Zuordnung sollte d innerhalb der Unsicherheit mit 0 vereinbar sein. Ein signifikanter d-Wert spricht für systematische Fehler.
    - **B₀ vs. B₁:** Meist gilt \(B_1 < B_0\), weil der mittlere Bindungsabstand im angeregten Schwingungszustand größer ist; dadurch steigt das Trägheitsmoment und B sinkt.
    - **Isotopeneffekt:** Die Isotopenmasse beeinflusst stark \(B\) und \(\Theta\), aber der berechnete Bindungsabstand sollte für H³⁵Cl und H³⁷Cl sehr ähnlich sein.
    - **Intensitätsverhältnis:** Die natürliche Häufigkeit von Chlorisotopen legt grob ein Verhältnis H³⁵Cl:H³⁷Cl von ca. 3:1 nahe; Abweichungen können durch Überlagerung, Rauschen, Linienform und Peak-Auswahl entstehen.
    """)
    return


if __name__ == "__main__":
    app.run()
