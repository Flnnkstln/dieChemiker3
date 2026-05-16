import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # Kleines Chemie-Dashboard

        Das ist ein sehr einfaches marimo-Notebook als App.

        ## Beispiel: ideale Gasgleichung

        \[
        pV = nRT
        \]

        Für ein ideales Gas gilt:

        - \(p\): Druck
        - \(V\): Volumen
        - \(n\): Stoffmenge
        - \(R\): Gaskonstante
        - \(T\): Temperatur

        **Ziel dieses Notebooks:**  
        Prüfen, ob Markdown, LaTeX und einfache Darstellung beim Export funktionieren.
        """
    )
    return


if __name__ == "__main__":
    app.run()
