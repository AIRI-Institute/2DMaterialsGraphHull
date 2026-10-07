"""Configurational free energies and the project's composition-simplex hull.

Energies inside this module are eV/atom, except configuration_free_energy's
explicitly per-cell input. Ordered elemental vertices have zero formation energy.
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

KB_EV = 8.617333262e-5
ELEMENTS = ['W', 'Mo', 'Se', 'S']
TEMPERATURES = np.arange(0., 1210., 10.)
HULL_TOLERANCE_EV = 1e-9  # 1e-6 meV/atom
ELEMENTAL_ENERGIES = dict(W=-12.93967184, Mo=-10.93534834, Se=-3.50518052, S=-4.12466045)


def configuration_free_energy(energies_cell, weights, temperature, n_atoms=1, k_b_eV=KB_EV):
    """Return -kT log(sum(w exp(-Ecell/kT)))/N, with the exact T=0 limit."""
    e, w = np.asarray(energies_cell, float), np.asarray(weights, float)
    if (e.ndim != 1 or e.size == 0 or e.shape != w.shape
            or not np.isfinite(e).all() or not np.isfinite(w).all()
            or (w <= 0).any() or not np.isfinite(temperature) or temperature < 0
            or not np.isfinite(n_atoms) or n_atoms <= 0 or not np.isfinite(k_b_eV) or k_b_eV <= 0):
        raise ValueError('Finite energies, positive weights/atom count and nonnegative T are required')
    e0 = e.min()
    if temperature == 0:
        return float(e0 / n_atoms)
    z = np.sum(w * np.exp(-(e - e0) / (k_b_eV * temperature)))
    return float((e0 - k_b_eV * temperature * np.log(z)) / n_atoms)


def simplex_basis(n):
    """Regular n-dimensional simplex, retaining the original notebook convention."""
    basis = np.vstack((np.eye(n), (1 + np.sqrt(n + 1)) / n * np.ones((1, n))))
    basis -= np.mean(basis, axis=0)
    return basis / np.sqrt(2)


def transform_composition(composition, basis):
    composition = np.asarray(composition, float)
    return np.sum((composition / composition.sum()).reshape(-1, 1) * basis, axis=0)


def hull_reference(point, hull, return_simplice=False):
    """Legacy lower-hull definition, including its published facet masks."""
    mask = [all(hull.points[i][-1] < 1e-5 for i in simplex) for simplex in hull.simplices]
    equations, simplices = hull.equations[mask], hull.simplices[mask]
    mask = (np.abs(equations[:, -2]) > 1e-5) & (np.abs(1 - equations[:, -2]) > 1e-5)
    equations, simplices = equations[mask], simplices[mask]
    if not len(equations):
        raise ValueError('No usable lower-hull facets')
    planes = equations.T
    values = -(np.dot(planes[:-2].T, point) + planes[-1]) / planes[-2]
    i = np.argmax(values)
    return (float(values[i]), simplices[i]) if return_simplice else float(values[i])


def lower_hull(counts, formation_energies):
    """Return reference, energy above hull, and simplex coordinates (eV/atom)."""
    counts, energies = np.asarray(counts, float), np.asarray(formation_energies, float)
    if (counts.ndim != 2 or counts.shape[1] < 2 or len(counts) != len(energies)
            or energies.ndim != 1 or not np.isfinite(counts).all()
            or not np.isfinite(energies).all() or (counts < 0).any()
            or (counts.sum(axis=1) <= 0).any()):
        raise ValueError('Invalid composition counts or formation energies')
    basis = simplex_basis(counts.shape[1] - 1)
    coordinates = np.array([transform_composition(c, basis) for c in counts])
    try:
        hull = ConvexHull(np.column_stack([coordinates, energies]))
    except QhullError as exc:
        raise ValueError('Compositions cannot form a nondegenerate hull') from exc
    reference = np.array([hull_reference(p, hull) for p in coordinates])
    return reference, energies - reference, coordinates


def formula_label(counts, include_zero=True):
    return ''.join(f'{e}{int(n)}' for e, n in zip(ELEMENTS, counts) if include_zero or n)


def validate_dft_minima(minima, formulas):
    """Validate and order the HT#2 per-composition minimum export."""
    if 'formation_energy_per_atom' not in minima or not minima.index.is_unique:
        raise ValueError('DFT minima require unique composition labels and formation_energy_per_atom')
    if minima.index.hasnans or not np.isfinite(minima.formation_energy_per_atom.to_numpy(float)).all():
        raise ValueError('DFT minima must be finite and labelled')
    missing = pd.Index(formulas).difference(minima.index)
    if len(missing):
        raise ValueError(f'Missing DFT minima for {len(missing)} compositions: {missing.tolist()}')
    return minima.loc[list(formulas)].copy()


def read_dft_minima(path, elemental_energies=None):
    """Read the supplied relaxed HT#2 text export, converting total to formation energy.

    The minimum is taken over the calculated configurations. It is not guaranteed
    to be the global DFT minimum. The elemental reference energies are stored explicitly.
    """
    mu = ELEMENTAL_ENERGIES if elemental_energies is None else elemental_energies
    best = {}
    token = re.compile(r'^([A-Z][a-z]?)(\d+)$')
    with Path(path).open() as stream:
        for line in stream:
            parts = line.split()
            if len(parts) < 4 or parts[0] == 'structure':
                continue
            counts = dict.fromkeys(ELEMENTS, 0)
            for value in parts[1:]:
                match = token.fullmatch(value)
                if match and match[1] in counts:
                    counts[match[1]] += int(match[2])
            n = sum(counts.values())
            if not n:
                continue
            ef = (float(parts[-2]) - sum(counts[e] * mu[e] for e in ELEMENTS)) / n
            if not np.isfinite(ef):
                raise ValueError('Nonfinite DFT energy in text export')
            key = tuple(counts[e] for e in ELEMENTS)
            best[key] = min(best.get(key, np.inf), ef)
    if not best:
        raise ValueError('No DFT compositions found')
    result = pd.DataFrame([dict(zip(ELEMENTS, k), formation_energy_per_atom=v) for k, v in sorted(best.items())])
    result.index = pd.Index([formula_label(k) for k in sorted(best)], name='formula_str')
    result.attrs['elemental_energies_eV_atom'] = dict(mu)
    result.attrs['source'] = str(path)
    return result


def prepare_gnn_configurations(predictions, model='nequip_random_ccs', dft_minima=None):
    """Convert the existing inference schema into canonical per-atom configuration rows.

    The original GNN columns store formation energy per cell. A supplied minimum
    table anchors every composition by a rigid per-atom shift; incomplete anchoring
    is an error. Omitting minima deliberately selects uncorrected GNN energies.
    """
    source_model = model if model in predictions else f'{model}_all'
    required = ['formula_str', 'formula', 'natoms', 'weight', source_model]
    missing = set(required).difference(predictions.columns)
    if missing:
        raise ValueError(f'Missing prediction columns: {sorted(missing)}')
    d = predictions[required + (['type'] if 'type' in predictions else [])].copy()
    if d.empty or d.formula_str.isna().any():
        raise ValueError('Prediction compositions must be nonempty and labelled')
    numeric = d[['natoms', 'weight', source_model]].to_numpy(float)
    if not np.isfinite(numeric).all() or (d.natoms <= 0).any() or (d.weight <= 0).any():
        raise ValueError('Predictions require finite energies and positive atom counts/weights')
    for e in ELEMENTS:
        d[e] = d['formula'].map(lambda f: f.get(e, 0))
    d['n_vac_Me'] = d['formula'].map(lambda f: f.get('U', 0))
    d['n_vac_X'] = d['formula'].map(lambda f: f.get('Np', 0))
    counts = d[ELEMENTS + ['n_vac_Me', 'n_vac_X']].to_numpy(float)
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError('Composition counts must be nonnegative integers')
    if not (d[ELEMENTS].sum(axis=1) == d.natoms).all():
        raise ValueError('Formula atom counts disagree with natoms')
    d['source_formula'] = d.formula_str
    d['formula'] = [formula_label(row) for row in d[ELEMENTS].to_numpy()]
    metadata = d.groupby('source_formula')[ELEMENTS + ['natoms', 'n_vac_Me', 'n_vac_X']].nunique()
    if (metadata != 1).any().any() or d.groupby('formula').source_formula.nunique().max() != 1:
        raise ValueError('Inconsistent or duplicate composition labels')
    d['n_atoms'] = d.natoms
    d['E_form_eV_per_atom'] = d[source_model] / d.natoms
    if dft_minima is not None:
        if 'formation_energy_per_atom' not in dft_minima or not dft_minima.index.is_unique:
            raise ValueError('DFT minima require unique labels and formation_energy_per_atom')
        anchor = {}
        for fs, group in d.groupby('source_formula'):
            label = fs if fs in dft_minima.index else group.formula.iloc[0]
            if label not in dft_minima.index:
                raise ValueError(f'Missing DFT minimum for {fs}')
            row = dft_minima.loc[label]
            value = float(row.formation_energy_per_atom)
            if not np.isfinite(value):
                raise ValueError(f'Invalid DFT minimum for {fs}')
            if all(e in row.index for e in ELEMENTS) and not np.array_equal(row[ELEMENTS].to_numpy(float), group[ELEMENTS].iloc[0].to_numpy(float)):
                raise ValueError(f'DFT composition counts disagree for {fs}')
            anchor[fs] = value
        shift = d.groupby('source_formula').E_form_eV_per_atom.min() - pd.Series(anchor)
        d['E_form_eV_per_atom'] -= d.source_formula.map(shift)
    d['coverage_type'] = d['type'] if 'type' in d else 'unspecified'
    return d


def build_free_energy_table(configurations, temperatures=TEMPERATURES, vacancy_free=False):
    """Return composition metadata and an (n_compositions, n_T) eV/atom array."""
    d = configurations.copy()
    required = ELEMENTS + ['n_atoms', 'weight', 'E_form_eV_per_atom', 'n_vac_Me', 'n_vac_X']
    missing = set(required).difference(d.columns)
    if missing:
        raise ValueError(f'Missing configuration columns: {sorted(missing)}')
    values = d[required].to_numpy(float)
    if (d.empty or not np.isfinite(values).all() or (d.n_atoms <= 0).any()
            or (d.weight <= 0).any() or (d[ELEMENTS + ['n_vac_Me', 'n_vac_X']] < 0).any().any()
            or not (d[ELEMENTS].sum(axis=1) == d.n_atoms).all()):
        raise ValueError('Invalid configuration energetics or composition metadata')
    counts = d[ELEMENTS + ['n_atoms', 'n_vac_Me', 'n_vac_X']].to_numpy(float)
    if not np.equal(counts, np.floor(counts)).all():
        raise ValueError('Atom and vacancy counts must be integers')
    T = np.asarray(temperatures, float)
    if T.ndim != 1 or not len(T) or not np.isfinite(T).all() or (T < 0).any() or (np.diff(T) <= 0).any():
        raise ValueError('Temperatures must be finite, nonnegative and strictly increasing')
    if vacancy_free:
        d = d[(d.n_vac_Me == 0) & (d.n_vac_X == 0)]
    rows, energies = [], []
    for counts, group in d.groupby(ELEMENTS, sort=True):
        if (group[['n_atoms', 'n_vac_Me', 'n_vac_X']].nunique() != 1).any():
            raise ValueError(f'Inconsistent atom/vacancy metadata for {counts}')
        n = int(group.n_atoms.iloc[0])
        row = dict(zip(ELEMENTS, counts), formula=formula_label(counts), n_atoms=n,
                   n_elements=sum(v > 0 for v in counts), n_configs=len(group),
                   sum_weight=float(group.weight.sum()), n_vac_Me=int(group.n_vac_Me.iloc[0]),
                   n_vac_X=int(group.n_vac_X.iloc[0]))
        if 'coverage_type' in group:
            row['coverage_type'] = ', '.join(sorted(group.coverage_type.unique()))
        else:
            row['coverage_type'] = 'complete 2x2x1 CCS'
        if 'source_formula' in group:
            row['source_formula'] = group.source_formula.iloc[0]
        rows.append(row)
        cell_energy = group.E_form_eV_per_atom.to_numpy(float) * n
        energies.append([configuration_free_energy(cell_energy, group.weight, t, n) for t in T])
    if not rows:
        raise ValueError('No compositions selected')
    return pd.DataFrame(rows), np.asarray(energies)


def temperature_hulls(metadata, free_energies, temperatures=TEMPERATURES, tolerance=HULL_TOLERANCE_EV):
    """Rebuild the hull at every temperature; return eV/atom arrays and grid onsets."""
    T, G = np.asarray(temperatures, float), np.asarray(free_energies, float)
    if G.shape != (len(metadata), len(T)) or not np.isfinite(G).all() or tolerance < 0:
        raise ValueError('Invalid free-energy grid or hull tolerance')
    counts = np.vstack([metadata[ELEMENTS].to_numpy(float), np.eye(4)])
    reference, above = np.empty_like(G), np.empty_like(G)
    for j in range(len(T)):
        ref, delta, _ = lower_hull(counts, np.r_[G[:, j], np.zeros(4)])
        reference[:, j], above[:, j] = ref[:len(G)], delta[:len(G)]
    if above.min() < -max(10 * tolerance, 1e-8):
        raise ValueError('Computed hull has appreciably negative energy above hull')
    above[np.abs(above) <= tolerance] = 0.
    on_hull = above <= tolerance
    onsets = np.array([T[np.flatnonzero(row)[0]] if row.any() else np.nan for row in on_hull])
    return dict(reference=reference, above=above, on_hull=on_hull, onsets=onsets)


def hull_distance_table(metadata, hull, temperatures=TEMPERATURES):
    """Archive-compatible wide table; dGhull columns are in meV/atom."""
    result = metadata.drop(columns=['coverage_type'], errors='ignore').copy()
    result['onset_T_K'] = hull['onsets']
    distances = pd.DataFrame(hull['above'] * 1000, columns=[f'dGhull_{int(t)}K' for t in temperatures])
    return pd.concat([result.reset_index(drop=True), distances], axis=1)


def long_result_table(metadata, free_energies, hull, temperatures=TEMPERATURES):
    """Long-format notebook result; free-energy columns are in eV/atom."""
    result = metadata.loc[metadata.index.repeat(len(temperatures))].reset_index(drop=True)
    result['T'] = np.tile(temperatures, len(metadata))
    result['F_conf'] = np.asarray(free_energies).ravel()
    result['F_hull_reference'] = hull['reference'].ravel()
    result['F_above_hull'] = hull['above'].ravel()
    result['on_hull'] = hull['on_hull'].ravel()
    result.rename(columns={'n_configs': 'num_configurations', 'sum_weight': 'sum_weights'}, inplace=True)
    return result
