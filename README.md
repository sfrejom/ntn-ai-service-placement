# Cross-Layer NTN Microservice Placement

Reproducibility artefacts for the journal article *"Experimental Validation
of an AI-Driven Microservice Placement Framework Across Non-Terrestrial
Network Layers"*.

The repository contains:

- a discrete-event simulator that reproduces the spatial, temporal, and
  energetic dynamics of a Non-Terrestrial Network composed of LEO
  satellites, High-Altitude Platform Stations (HAPS), and Unmanned Aerial
  Vehicles (UAVs);
- four placement policies: a uniform-random baseline, a latency-greedy
  heuristic, a layer-aware rule-based policy, and an Integer Linear
  Programming (ILP) oracle;
- a Deep Reinforcement Learning agent (Proximal Policy Optimization with
  attention-based, autoregressive node selection);
- training, evaluation, scalability and ablation pipelines, and the
  figure-generation code used in the paper;
- the LaTeX source of the article, including the trained policy
  checkpoint and the saved evaluation outputs.

## Repository layout

```
simulator/        NTN environment, node and service models
agents/           placement policies (baselines, ILP, DRL)
experiments/      training and evaluation scripts, figure generation
results/          trained policy and saved evaluation outputs (regenerable)
figures/          generated figures used in the paper
paper/            LaTeX source, bibliography, and compiled PDF
TECHNICAL_GUIDE.md   end-to-end implementation walkthrough
CHANGES.md           detailed log of formulation refinements
requirements.txt     Python dependencies
```

## Requirements

- Python 3.13 or newer
- A working LaTeX distribution (TeX Live 2023+ recommended) to build the
  PDF
- The Python packages listed in `requirements.txt`

## Quick start

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train the DRL agent (about 95 seconds on a single CPU thread):

```bash
python -m experiments.run_training --episodes 800
```

Evaluate every policy across the four scenarios (about three minutes):

```bash
python -m experiments.run_evaluation --seeds 5
```

Run the scalability and ablation studies (about five minutes):

```bash
python -m experiments.run_scalability
```

Generate every figure used in the paper:

```bash
python -m experiments.make_figures
python -m experiments.make_figures_extras
```

Build the PDF:

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

The full pipeline (training, evaluation, scalability, ablation, figures,
and PDF compilation) runs end-to-end in under ten minutes on a single CPU
thread.

## License and citation

This artefact is released under the same license as the article. If you
use it, please cite the journal article as the canonical reference.
