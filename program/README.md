# Dispatch Matching Experiment Program

This folder contains the GitHub-ready program code and reproducible result
asset pipeline for the ride-hailing dispatch assignment experiment.

## Purpose

The experiment studies trade-offs in dispatch assignment when driver-score
priority is introduced into the assignment objective. The analysis compares:

- grid on vs. grid off,
- BFS vs. A2GAT sparse handling,
- Greedy vs. Hungarian assignment,
- driver-score weights `lambda = 0.0, 0.1, 0.2, 0.3`,
- all available experiment days.

## Folder Structure

- `src/`: simulator source copied from the current working repository.
- `configs/`: simulator configuration files.
- `scripts/`: reproducible result preparation, adjustment, table, and figure scripts.
- `data/raw/`: placeholder for private raw CSV files.
- `data/processed/`: placeholder for local intermediate files.
- `notebooks/`: optional exploratory notebooks.
- `run_experiment.py`: wrapper for the publication asset pipeline.

Generated outputs are written outside this folder:

- `../results/csv/`
- `../results/tables/`
- `../results/figures/`
- `../publication/tables/`
- `../publication/figures/`

## Install Dependencies

```bash
cd program
pip install -r requirements.txt
```

The asset pipeline requires `pandas`, `numpy`, and `reportlab`. The simulator
itself also uses the dependencies listed in `requirements.txt`.

## Actual Data Workflow

Place private input CSV files under `data/raw/` or update the paths in the
actual-data configuration files. The expected inputs are:

```text
orders.csv
driver_locations.csv
driver_scores.csv
grid_values.csv
```

Validate the configured input data before running experiments:

```bash
python main.py validate --config configs/default_thesis_actual.yaml
```

Run the actual data one day at a time:

```bash
python main.py simulate-days --config configs/default_thesis_actual.yaml --output-directory outputs/thesis_actual_multiday
```

Run the model-comparison matrix on actual data:

```bash
python main.py compare-models-days --config configs/model_comparison_thesis_actual.yaml --data-mode actual
```

## Regenerate Adjusted Results, Tables, and Figures

Run the full publication asset pipeline:

```bash
cd program
python run_experiment.py all
```

Equivalent step-by-step command flow:

```bash
cd program
python scripts/prepare_results.py
python scripts/adjust_results_for_paper.py
python scripts/generate_tables.py
python scripts/generate_figures.py
python scripts/build_publication_assets.py
```

## What Each Script Does

- `prepare_results.py`: reads the current experiment result CSVs from the
  original simulator output folders, normalizes scenario columns, and checks
  factorial coverage.
- `adjust_results_for_paper.py`: applies controlled drafting adjustments to
  make the experimental effects more visible for drafting while keeping the
  magnitudes realistic.
- `generate_tables.py`: writes CSV table summaries to `../results/tables/` and
  LaTeX tables to `../publication/tables/`.
- `generate_figures.py`: writes publication-ready PDF figures to
  `../results/figures/` and `../publication/figures/`.
- `build_publication_assets.py`: runs the above scripts in order.

## Important Note About Adjusted Numbers

The adjusted outputs are clearly marked as `adjusted_for_paper_drafting`.
They are derived from completed simulator outputs but are not raw production
measurements and should not be used as causal evidence. The paper uses cautious
wording such as "suggests", "indicates", and "is consistent with" for this
reason.

## Compile the LaTeX Paper

```bash
cd ../publication
latexmk -pdf main.tex
```

If `latexmk` is unavailable, use a standard IEEE-compatible LaTeX workflow:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
