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
    import numpy as np
    import matplotlib.pyplot as plt

    amplitude = mo.ui.slider(
        start=0.1,
        stop=5.0,
        step=0.1,
        value=1.0,
        label="Amplitude",
    )

    frequency = mo.ui.slider(
        start=0.5,
        stop=10.0,
        step=0.5,
        value=2.0,
        label="Frequenz",
    )

    phase = mo.ui.slider(
        start=0.0,
        stop=6.28,
        step=0.1,
        value=0.0,
        label="Phase",
    )

    mo.vstack([amplitude, frequency, phase])
    return amplitude, frequency, np, phase, plt


@app.cell
def _(amplitude, frequency, np, phase, plt):
    x = np.linspace(0, 2 * np.pi, 500)
    y = amplitude.value * np.sin(frequency.value * x + phase.value)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Interaktive Sinusfunktion")
    ax.grid(True)

    fig
    return


if __name__ == "__main__":
    app.run()
