# Experiment Dispatch Matching

This project folder contains the reorganized ride-hailing dispatch matching
experiment.

## Main Folders

- `program/`: GitHub-ready simulator code, configs, scripts, and placeholders
  for raw actual-format data.
- `publication/`: IEEE-style LaTeX paper, references, generated tables, and
  generated figures.
- `results/`: adjusted CSV outputs, table summaries, figures, and validation
  logs.

## Regenerate Assets

```bash
cd program
python run_experiment.py all
```

The pipeline reads the current simulator comparison outputs, normalizes all
scenario-day rows, applies controlled paper-drafting adjustments, and regenerates
the paper tables and figures.

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

## Important Note

The result files under `results/` are adjusted experiment outputs for paper
drafting. They are derived from completed simulator runs, but they are not raw
production measurements and should not be presented as causal evidence.
