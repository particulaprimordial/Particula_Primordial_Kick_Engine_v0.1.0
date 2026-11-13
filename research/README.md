# Research Notebook — Particula Primordial Kick Engine

This folder is the project's technical record and learning companion. It is deliberately separate from the application code:

- `PAPER.md` is the living IMRaD-style technical paper.
- `EXPERIMENT_PROTOCOL.md` defines repeatable tests and the distinction between measurements and listening judgments.
- `EXPERIMENT_LOG_TEMPLATE.md` is a blank record for future design experiments.
- `GLOSSARY.md` explains the signal-processing terms used by the project.
- `REFERENCES.md` provides books, technical documentation and standards for further study.
- `data/` stores measurements calculated from the included factory WAVs.
- `figures/` stores static charts generated from those measurements for use in a paper or portfolio.

## Regenerate the evidence

From the repository root, run:

```bash
python -m pip install -r requirements-research.txt  # from project root; optional
python scripts/generate_research_artifacts.py
```

The optional dependency file installs Matplotlib for static figure generation; it is not required for the online app. The script reads the WAV files in `examples/factory-kicks/`, recomputes the descriptors, and overwrites the CSV, JSON and PNG figures. It does not fabricate listener ratings or subjective conclusions.

## How to read the paper

Read the Introduction to understand the design problem, Methods to understand what the software computes, Results to inspect the initial five-sample dataset, and Limitations before making claims. The paper is a technical project report, not a peer-reviewed claim that the generated kicks are objectively better or more danceable.
