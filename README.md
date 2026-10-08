# 2DMaterialsGraphHull

**2DMaterialsGraphHull** provides the composition/configuration-space (CCS) datasets, DFT and graph neural network (GNN) energetics, and post-processing code developed for studying chemical disorder and point defects in two-dimensional transition metal dichalcogenide (TMD) monolayers — alloyed and defected MeX₂ (Me = Mo, W; X = S, Se) systems. This repository is part of publication [Combined DFT/GNN analysis of compositional disorder and defects in MeX₂ (Me = Mo, W; X = S, Se) monolayers](https://doi.org/10.1088/2053-1583/aeb142).

<p align="center">
  <img src="./figures/logo.png" width="100%" title="2DMaterialsGraphHull datasets" alt="2DMaterialsGraphHull datasets"/>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Analysis Workflow](#analysis-workflow)
- [Data Requirements](#data-requirements)
- [Installation and Usage](#installation-and-usage)
- [Dependencies](#dependencies)
- [Model Naming Convention](#model-naming-convention)
- [Related Repositories](#related-repositories)
- [Citation](#citation)
- [License](#license)

---

## Overview

Modeling disorder and defects in 2D materials is hard because the number of possible atomic arrangements grows combinatorially with supercell size, which puts exhaustive first-principles treatment out of reach. This project addresses that with a workflow that combines:

- **Symmetry-aware configuration sampling (CCS)** — enumeration of symmetrically inequivalent defect/alloy arrangements in MeX₂ supercells, built with the *Supercell* program and analyzed with *Spglib*/*pymatgen*, while retaining the degeneracy weights needed for thermodynamic averaging.
- **DFT calculations (VASP, PBE)** — reference formation energies and relaxed structures for training, validation, and holdout evaluation.
- **GNN inference (Allegro and NequIP, E(3)-equivariant)** — models trained on different CCS- and 2DMD-derived subsets (including low- and high-defect-concentration splits) to predict energetics of unrelaxed configurations at scale, letting convex-hull and finite-temperature free-energy analysis be carried out across full configuration sets.
- **Configurational entropy / finite-temperature free energy** — partition-function-based evaluation of ΔG using CCS weights, to go beyond 0 K convex-hull screening.
- **Interpretable ML (Ridge Regression / Random Forest on defect-count descriptors)** — used to relate local defect motifs to energetic favorability alongside the GNN-based analysis.

Two complementary data sources feed this workflow:

- **CCS (Composition/Configuration Space)** — the symmetry-unique defect/alloy configurations generated and DFT/GNN-evaluated in this work.
- **2DMD dataset** — an existing library of 2D-material point-defect structures (Huang P., Lukin R., et al. Unveiling the complex structure-property correlation of defects in 2D materials based on high throughput datasets, *npj 2D Mater Appl* **7**, 6 (2023)), processed and augmented here via the companion [2DMD_at_a_Glance](https://github.com/AIRI-Institute/2DMD_at_a_Glance) code for use as additional training/validation/holdout data.

The notebooks cover CCS statistics and symmetry, 0 K convex hulls, DFT/GNN model evaluation, configurational free energies, cell-size sensitivity, and vibrational contributions. The [finite-temperature analysis](finite_temperature_hull_analysis.ipynb) rebuilds the lower convex hull at each temperature and evaluates stability across the 2×2×1 and 4×4×1 CCSs. Its default workflow uses the included compact datasets; an optional mode recomputes second-inference free energies from individual GNN predictions.

---

## Repository Structure

| File or directory | Description |
|---|---|
| [train_ccs_statistics.ipynb](train_ccs_statistics.ipynb) | Training CCS statistics: space groups, symmetry weights, composition correlations and defect counts. |
| [convex_hull_analysis.ipynb](convex_hull_analysis.ipynb) | The 0 K reference hull from training CCS, pure elements and 2DMD data; formation energies, distances above hull and composition-simplex views. |
| [preliminary_test.ipynb](preliminary_test.ipynb) | Preliminary DFT/GNN comparison on low-symmetry training CCS configurations and model errors across hull-distance cutoffs. |
| [holdout_test_1.ipynb](holdout_test_1.ipynb) | First inference CCS and HT#1: model evaluation, configurational free energies and structural-descriptor analysis at two fixed compositions. |
| [holdout_test_2.ipynb](holdout_test_2.ipynb) | Second inference CCS and HT#2: model-guided selection, DFT/GNN comparisons, composition-resolved stability and DFT minimum export. |
| [finite_temperature_hull_analysis.ipynb](finite_temperature_hull_analysis.ipynb) | Configurational free energies, temperature-dependent hulls, effective-cell-size estimates, phonon tie planes and onset shifts. |
| [tools.py](tools.py) | Utilities for data processing, hull calculations, configurational entropy and structure manipulation. |
| [analysis/](analysis/) | Shared thermodynamic, finite-size, phonon and plotting functions. |
| [requirements.txt](requirements.txt) | Python dependencies for all notebooks. |
| [data/](data/) | Included compact inputs, reference tables and downloaded datasets. |
| [phonons_results/](phonons_results/) | Structures, phonopy outputs and compressed VASP force-calculation records for 25 cells, with calculation provenance. |
| [figures/](figures/) | Publication figures and the repository logo. |
| [results/](#results) | Generated tables and figures. |

---

## Analysis Workflow

Each notebook opens with a short description, links to the repository guide and adjacent analyses, and a contents list for its sections. The analyses follow this sequence:

1. [Training CCS statistics](train_ccs_statistics.ipynb) — characterize the composition and configuration sets.
2. [0 K convex hull](convex_hull_analysis.ipynb) — construct the energetic reference from CCS, pure-element and 2DMD data.
3. [Preliminary testing](preliminary_test.ipynb) — compare model predictions on low-symmetry configurations.
4. [First holdout test](holdout_test_1.ipynb) — assess configurational generalization at two fixed compositions and inspect structural descriptors.
5. [Second holdout test](holdout_test_2.ipynb) — evaluate the second inference CCS and export the calculated DFT minima.
6. [Finite-temperature stability](finite_temperature_hull_analysis.ipynb) — evaluate configurational stabilization, cell-size sensitivity and vibrational contributions.

### Finite-temperature stability

The finite-temperature notebook evaluates stability across the 2×2×1 and 4×4×1 CCSs, explains the thermodynamic assumptions, and reproduces the corresponding tables and figures.

| Notebook section | Input and purpose | Publication result |
|---|---|---|
| [Configurational free energy and full hull](finite_temperature_hull_analysis.ipynb#configurational-free-energy) | 11,159 DFT configurations in the 420-composition 2×2×1 CCS; rebuild the hull at every temperature | Full finite-temperature hull table |
| [Vacancy-free hull](finite_temperature_hull_analysis.ipynb#vacancy-free-hull) | Rebuild the hull from the 45 vacancy-free compositions, including 21 quaternaries | Supplementary Figure 24: [PNG](figures/finite_temperature/figure_quaternary_onhull_vacancyfree.png) · [PDF](figures/finite_temperature/figure_quaternary_onhull_vacancyfree.pdf) |
| [Second inference CCS](finite_temperature_hull_analysis.ipynb#second-inference-ccs) | Included DFT-anchored GNN hull-distance table for 544 compositions of the 4×4×1 CCS | Figure 6d: [PNG](figures/finite_temperature/fig441_6d_MoW.png) · [PDF](figures/finite_temperature/fig441_6d_MoW.pdf); Supplementary Figure 23: [PNG](figures/finite_temperature/fig441_6d_SSe.png) · [PDF](figures/finite_temperature/fig441_6d_SSe.pdf) |
| [Effective cell size](finite_temperature_hull_analysis.ipynb#effective-cell-size) | Combinatorial entropy deficits and rescaled configurational stabilization | Supplementary Table 4; 4×4×1 entropy-deficit estimates |
| [Phonon gates and moving tie planes](finite_temperature_hull_analysis.ipynb#phonon-acceptance-gates) | 25 included phonopy records, six targets and their endmembers | Supplementary Figure 25: [PNG](figures/phonons/figure_dfvib_onset_window.png) · [PDF](figures/phonons/figure_dfvib_onset_window.pdf) |
| [Onset shifts](finite_temperature_hull_analysis.ipynb#onset-shifts) | Vibrational residuals against the vacancy-free hull | Supplementary Table 5: [CSV](data/phonons/supplementary_table_5.csv) |

### Running the finite-temperature analysis

After installing the `requirements.txt`, open the finite-temperature notebook and run all cells. Its default `run_full_gnn = False` uses the included tables and phonopy outputs; it requires neither VASP/phonopy execution nor the large downloaded inference datasets. A batch run of the same notebook is available through Jupyter:

```bash
jupyter nbconvert --to notebook --execute finite_temperature_hull_analysis.ipynb \
  --output finite_temperature_hull_analysis.executed.ipynb --output-dir results \
  --ExecutePreprocessor.timeout=600
```

### Results storage

Generated tables and figures are saved to `results/finite_temperature/`. The output directory can be redirected with `GRAPHHULL_OUTPUT_DIR`.

All internal thermodynamic energies and long-format free-energy columns are **eV/atom**. Wide `dGhull_*` columns and vibrational tables are **meV/atom**. The temperature grid spans 0–1200 K in 10 K steps. Ordered elemental references remain at zero formation energy. The full and vacancy-free hulls use different composition sets; the phonon analysis uses the vacancy-free hull.

Effective-cell-size stabilization is an estimate that transfers the base-cell energetic selectivity. Table 5 reports two onset-shift approximations. Its direct estimate uses the reconstructed 40 K backward-slope window documented in the notebook. All 15 supporting planes are computable; 7 of 10 competing planes are computable from the supplied phonon records.

---

## Data Requirements

The compact thermodynamic inputs and calculation records are included in the repository:

| Included path | Content |
|---|---|
| [`data/ccs_2x2x1_full_energetics.csv`](data/ccs_2x2x1_full_energetics.csv) | Shared DFT energetics and symmetry weights for 11,159 configurations across 420 compositions. |
| [`data/finite_temperature/`](data/finite_temperature/) | Full and vacancy-free 2×2×1 hull exports, the 4×4×1 hull-distance/onset cache, and the relaxed HT#2 energy record `11_wmo_full.out`. |
| [`data/phonons/`](data/phonons/) | Vibrational free energies, target-window definitions, facet accounting and selection, and the reference Table 5. |
| [`phonons_results/cells/`](phonons_results/cells/) | Unmodified per-cell phonopy and VASP calculation records. |

The broader CCS statistics, model evaluation and optional full GNN workflow use the following downloaded datasets and generated outputs in the root `data/` directory.


| Dataset                                     | Description                                                                                                                                                                                                    |
|---------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `2d-materials-point-defects-all-table`      | Processed 2DMD dataset (12,866 samples), prepared with the related [2DMD_at_a_Glance](https://github.com/AIRI-Institute/2DMD_at_a_Glance) project.                                                                                                                                                                        |
| `neat_elements-dft_subset`                  | Reference DFT-derived energies of the pure elements (W, Mo, Se, S) (4 samples).                                                                                                                                |
| `train_ccs-gnn_predictions_dft_subset`      | Training CCS with detailed description of structures, DFT-derived and GNN-predicted energies (11159 samples).                                                                                                  |
| `train_ccs-2dmd-convex_hull`                | Training CCS and 2DMD structures with convex-hull energies and compositional simplex coordinates (11159 + 12866 + 4 samples).                                                                                  |
| `2dmd-W1Mo14Se2S28_W14Mo1Se28S2-dft_subset` | The most favorable/unfavorable structures of 2DMD dataset with W<sub>1</sub>Mo<sub>14</sub>Se<sub>2</sub>S<sub>28</sub> and W<sub>14</sub>Mo<sub>1</sub>Se<sub>28</sub>S<sub>2</sub> compositions (4 samples). |
| `inference1_ccs-dft_subset`                 | Subset of the first inference CCS with DFT-derived energies (HT#1) (5556 samples).                                                                                                                             |
| `inference1_ccs-gnn_predictions_subset`     | The first inference CCS with GNN-predicted energies (542196 samples).                                                                                                                                          |
| `inference2_ccs-composition_size`           | Numbers of symmetry-inequivalent structures corresponding to each composition in the second inference CCS (544 samples).                                                                                       |
| `inference2_ccs-gnn_predictions_subset`     | Subset of the second inference CCS with GNN-predicted energies (5004502 samples). It includes 11486 HT#2 structures; their DFT energies are processed from VASP results in `holdout_test_2.ipynb`.     |
| `inference2_ccs-dft-minima`                 | Optional table generated by `holdout_test_2.ipynb`: the lowest calculated HT#2 DFT formation energy for each second-inference composition, indexed by `formula_str`. Used for DFT correction in the optional full GNN run; if this export is absent, the included relaxed-energy text record provides the anchors. |
| `inference2_ccs-finite_temperature_hull`    | Generated long-format table containing composition, temperature, $F_{conf}$, the finite-temperature hull reference, $F_{above hull}$, hull membership, and CCS coverage metadata.                          |

---

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/AIRI-Institute/2DMaterialsGraphHull.git
   cd 2DMaterialsGraphHull
   ```
2. Create a Python virtual environment, activate it and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Use the [2DMD_at_a_Glance](https://github.com/AIRI-Institute/2DMD_at_a_Glance) project to prepare the `2d-materials-point-defects-all-table` dataset and place it in the `data/` directory. The [CSSLib](https://github.com/AIRI-Institute/CSSLib) library (used to generate and analyze the CCSs themselves) is available separately.
4. Download datasets to `data/` directory by links below:
   - [neat_elements-dft_subset](https://2d-materials-graph-hull.obs.ru-moscow-1.hc.sbercloud.ru/neat_elements-dft_subset.pkl.gz)
   - [train_ccs-gnn_predictions_dft_subset](https://2d-materials-graph-hull.obs.ru-moscow-1.hc.sbercloud.ru/train_ccs-gnn_predictions_dft_subset.pkl.gz)
   - [train_ccs-2dmd-convex_hull](https://2d-materials-graph-hull.obs.ru-moscow-1.hc.sbercloud.ru/train_ccs-2dmd-convex_hull.pkl.gz)
   - [2dmd-W1Mo14Se2S28_W14Mo1Se28S2-dft_subset](https://2d-materials-graph-hull.obs.ru-moscow-1.hc.sbercloud.ru/2dmd-W1Mo14Se2S28_W14Mo1Se28S2-dft_subset.pkl.gz)
   - [inference1_ccs-dft_subset](https://2d-materials-graph-hull.obs.ru-moscow-1.hc.sbercloud.ru/inference1_ccs-dft_subset.pkl.gz)
   - [inference1_ccs-gnn_predictions_subset](https://2d-materials-graph-hull.obs.ru-moscow-1.hc.sbercloud.ru/inference1_ccs-gnn_predictions_subset.pkl.gz)
   - [inference2_ccs-composition_size](https://2d-materials-graph-hull.obs.ru-moscow-1.hc.sbercloud.ru/inference2_ccs-composition_size.pkl.gz)
   - [inference2_ccs-gnn_predictions_subset](https://2d-materials-graph-hull.obs.ru-moscow-1.hc.sbercloud.ru/inference2_ccs-gnn_predictions_subset.pkl.gz)
5. Launch Jupyter Lab or Notebook:
   ```bash
   jupyter lab
   ```
Each notebook contains explanatory comments and can be executed independently once its required inputs are in place. The finite-temperature notebook’s compact default uses the included inputs; the other notebooks require the downloaded datasets and, for DFT processing, the original VASP records.

The finite-temperature notebook uses the included compact inputs by default. Set `run_full_gnn = True` to read `inference2_ccs-gnn_predictions_subset.pkl.gz` and recompute configurational free energies for the second inference CCS. DFT minimum correction is enabled by default in this mode; set `apply_dft_minimum_correction = False` for uncorrected GNN energies. The included 4×4×1 cache contains hull distances and onsets, not absolute free energies, so it cannot replace the individual predictions for this recomputation.

For DFT anchoring, the notebook prefers `data/inference2_ccs-dft-minima.pkl.gz` exported by `holdout_test_2.ipynb`, or falls back to the included `data/finite_temperature/11_wmo_full.out` with explicit elemental reference energies. The HT#2 export groups `structures_fi_result` by `formula_str`, validates complete composition coverage and finite energies, records the energy unit and elemental references, and writes the per-composition minimum. Generating that export requires the original VASP results under `dft/neat_elements/3res_structures/` and `dft/WMoUSeSNp_enh2_low_en/3res_structures/`. A minimum over these calculated configurations is not guaranteed to be the global DFT minimum. Missing anchors stop the corrected calculation. A full run writes `data/inference2_ccs-finite_temperature_hull.pkl.gz` and an inspectable CSV under `results/finite_temperature/`.

---

## Dependencies

Main packages (see `requirements.txt`):

- Python ≥ 3.11
- Jupyter, Matplotlib, NumPy, Pandas, Plotly, Seaborn
- PyYAML (reading included phonopy records)
- PyMatGen (structure manipulation, space-group analysis)
- SciPy, scikit-learn (Ridge Regression, Random Forest, GridSearchCV)
- NetworkX

GNN models (Allegro, NequIP) are not included here; the notebooks read pre-computed predictions from the inference files. Model training used the *AFLOWLib*-derived pretraining set alongside CCS- and 2DMD-based fine-tuning data, as described in the paper's Methods section.

---

## Model Naming Convention

Trained/inference model identifiers follow the pattern:

```
architecture_(pretrain set)_(training/fine-tuning dataset)
```

e.g., a model named `allegro_random_2dmd-ldc` refers to the *Allegro* architecture, pretrained on the random-sampling pretraining set, and fine-tuned on the low-defect-concentration (LDC) 2DMD subset. Use this convention to identify which model produced a given set of predictions in the inference files.

---

## Related Repositories

- [2DMD_at_a_Glance](https://github.com/AIRI-Institute/2DMD_at_a_Glance) — data processing, unification, and augmentation of the 2DMD datasets.
- [CSSLib](https://github.com/AIRI-Institute/CSSLib) — generation of composition/configuration spaces (CCSs) and analysis of their space-group/group-subgroup relations.

---

## Citation

If you use this work, please cite the accompanying paper:

```
@article{10.1088/2053-1583/aeb142,
  title={Combined DFT/GNN analysis of compositional disorder and defects in MeX2 (Me = Mo, W; X = S, Se) monolayers},
  author={Eremin, Roman A. and Krautsou, Aliaksei V. and Humonen, Innokentiy S. and Ryabov, Alexander A. and Khrabrov, Kuzma A. and Dembitskiy, Artem D. and Solovykh, Alexander A. and Antropov, Aleksandr S. and Efimov, Albert R. and Novoselov, Kostya S. and Budennyy, Semen A.},
  journal={2D Materials},
  url={http://iopscience.iop.org/article/10.1088/2053-1583/aeb142},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
