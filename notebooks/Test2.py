# /// script
# dependencies = [
#     "marimo>=0.23.6",
# ]
# [tool.marimo.opengraph]
# title = "Gasgesetz-Rechner"
# description = "Interaktive Rechnung mit pV = nRT."
# ///


import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    n = mo.ui.slider(
        start=0.1,
        stop=5.0,
        step=0.1,
        value=1.0,
        label="Stoffmenge n / mol",
    )

    T = mo.ui.slider(
        start=250,
        stop=400,
        step=1,
        value=298,
        label="Temperatur T / K",
    )

    V = mo.ui.slider(
        start=1,
        stop=100,
        step=1,
        value=24,
        label="Volumen V / L",
    )

    mo.vstack([n, T, V])
    return T, V, mo, n


@app.cell
def _(T, V, mo, n):
    R = 0.08314  # L bar mol^-1 K^-1

    p = n.value * R * T.value / V.value

    mo.md(
        rf"""
        ## Ideale Gasgleichung

        \[
        p = \frac{{nRT}}{{V}}
        \]

        Eingesetzt:

        \[
        p = \frac{{{n.value:.2f} \cdot 0.08314 \cdot {T.value:.0f}}}{{{V.value:.0f}}}
        \]

        Ergebnis:

        # \(p = {p:.3f}\,\mathrm{{bar}}\)
        """
    )
    return


if __name__ == "__main__":
    app.run()
