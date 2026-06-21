# Experiment Dispatch Matching

This project folder contains the reorganized ride-hailing dispatch matching
experiment.

## Main Folders

- `program/`: GitHub-ready simulator code, configs, scripts, and placeholders
  for raw actual-format data.
- `publication/`: IEEE-style LaTeX paper, references, generated tables, and
  generated figures.
- `results/`: measured actual-data simulation outputs, table summaries,
  figures, and validation logs.

## Regenerate Assets

```bash
cd program
python run_experiment.py all
```

The pipeline reads the completed actual-data comparison runs, normalizes all
scenario-day rows without changing measured values, and regenerates the paper
tables and figures.

## Compile Paper

```bash
cd publication
latexmk -pdf main.tex
```

If `latexmk` is not installed, use:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Interpretation Boundary

The results are outputs of simulations run with actual input data. They support
comparisons within the stated simulator, dates, and configuration matrix. Model-
based quantities such as expected conversion, utility, and simulated driver
income are not observed production outcomes and are not interpreted causally.
