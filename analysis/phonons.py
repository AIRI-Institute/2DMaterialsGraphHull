"""Phonon acceptance gates, moving tie planes and onset-shift estimates.

Vibrational tables and residuals use meV/atom. Configurational energies passed
to the tie-plane linear program use eV/atom. Targets are excluded from their
own tie-plane search; planes above the onset describe competition.
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linprog

from .thermodynamics import (
    KB_EV, ELEMENTS, TEMPERATURES, build_free_energy_table, formula_label,
)

KJMOL_TO_MEV = 10.364269
SLOPE_WINDOW_K = 40.  # reconstructed backward window used by the supplied Table 5


def phonon_gate_report(mesh, thermal):
    """Report the published acceptance criteria, including the strict Γ threshold."""
    frequencies = np.array([b['frequency'] for q in mesh['phonon'] for b in q['band']], float)
    gamma = [q for q in mesh['phonon'] if np.linalg.norm(q['q-position']) < 1e-9]
    if not len(frequencies) or not gamma or len(gamma[0]['band']) < 3 or not np.isfinite(frequencies).all():
        raise ValueError('Phonon mesh must contain finite modes and three Γ acoustic modes')
    acoustic = max(abs(b['frequency']) for b in gamma[0]['band'][:3])
    imaginary = int((frequencies < -.05).sum())
    n = thermal['natom']
    return dict(gamma_acoustic_THz=float(acoustic), modes_below_minus_005_THz=imaginary,
                n_atoms=int(n), maximum_frequency_THz=float(frequencies.max()),
                accepted=bool(acoustic < .05 and imaginary == 0 and n == 12))


def build_vibrational_table(record_root):
    """Read all cells.txt records; reject any missing or unacceptable calculation."""
    root = Path(record_root)
    cells = [s.strip() for s in (root / 'cells.txt').read_text().splitlines() if s.strip()]
    if not cells or len(set(cells)) != len(cells):
        raise ValueError('cells.txt must contain a nonempty unique list')
    columns, reports, temperature_grid = {}, [], None
    for cell in cells:
        with (root / 'cells' / cell / 'mesh.yaml').open() as stream:
            mesh = yaml.safe_load(stream)
        with (root / 'cells' / cell / 'thermal_properties.yaml').open() as stream:
            thermal = yaml.safe_load(stream)
        report = phonon_gate_report(mesh, thermal)
        reports.append(dict(cell=cell, **report))
        if not report['accepted']:
            raise ValueError(f'{cell}: phonon acceptance gate failed: {report}')
        properties = thermal['thermal_properties']
        T = np.array([p['temperature'] for p in properties], float)
        F = np.array([p['free_energy'] for p in properties], float) * KJMOL_TO_MEV / thermal['natom']
        if not np.isfinite(T).all() or not np.isfinite(F).all() or (np.diff(T) <= 0).any():
            raise ValueError(f'{cell}: invalid thermal properties')
        if temperature_grid is None:
            temperature_grid = T
        if not np.array_equal(T, temperature_grid):
            raise ValueError(f'{cell}: temperature grid differs')
        columns[cell] = F
    return pd.DataFrame(columns, index=pd.Index(temperature_grid, name='T')), pd.DataFrame(reports)


def _vacancy_free_energies(configurations, temperatures):
    meta, G = build_free_energy_table(configurations, temperatures, vacancy_free=True)
    labels = [formula_label(c, include_zero=False) for c in meta[ELEMENTS].to_numpy()]
    # Match the archived LP column order: labels sorted lexicographically.
    order = np.argsort(labels)
    return meta.iloc[order].reset_index(drop=True), [labels[i] for i in order], G[order]


def account_facets(configurations, vibrational, targets, temperatures=TEMPERATURES):
    """Enumerate every target-excluded tie-plane regime and its phonon coverage."""
    T = np.asarray(temperatures, float)
    meta, labels, G = _vacancy_free_energies(configurations, T)
    counts = meta[ELEMENTS].to_numpy(float)
    fractions = counts / counts.sum(axis=1)[:, None]
    have = set(vibrational.columns)
    if not set(T).issubset(vibrational.index):
        raise ValueError('Missing vibrational temperatures')
    rows = []
    for target, info in targets.items():
        if target not in labels or target not in have:
            raise ValueError(f'Missing target {target}')
        index = labels.index(target)
        others = [i for i in range(len(labels)) if i != index]
        A = np.vstack([fractions[others].T, np.ones(len(others))])
        b = np.r_[fractions[index], 1.]
        facets = []
        for j in range(len(T)):
            solution = linprog(G[others, j], A_eq=A, b_eq=b,
                               bounds=[(0, None)] * len(others), method='highs')
            if not solution.success:
                raise ValueError(f'{target} at {T[j]} K: tie-plane LP failed: {solution.message}')
            facets.append(tuple(sorted((labels[others[i]], round(value, 4))
                                       for i, value in enumerate(solution.x) if value > 1e-7)))
        start = 0
        while start < len(T):
            end = start + 1
            while end < len(T) and facets[end] == facets[start]:
                end += 1
            lo, hi = T[start], T[end - 1]
            members = dict(facets[start])
            regime = 'supporting' if lo <= info['onset'] else 'competing'
            missing = [c for c in members if c not in have]
            maximum = None
            if not missing and regime == 'supporting':
                residual = vibrational[target] - sum(w * vibrational[c] for c, w in members.items())
                maximum = float(residual.loc[lo:min(hi, info['onset'])].abs().max())
            rows.append(dict(target=target, lo=int(lo), hi=int(hi), regime=regime,
                             ok=not missing, miss=missing, dmax=maximum, members=members))
            start = end
    return pd.DataFrame(rows)


def select_facets(accounting, targets):
    """Select every supporting plane and the first competing plane if computable."""
    selected = {}
    for target, info in targets.items():
        group = accounting[accounting.target == target].sort_values('lo')
        runs, competing_seen = [], 0
        for row in group.itertuples():
            if row.regime == 'competing':
                competing_seen += 1
                if competing_seen != 1 or not row.ok:
                    continue
            if row.regime == 'supporting' and not row.ok:
                raise ValueError(f'{target}: missing supporting endmembers {row.miss}')
            runs.append(dict(lo=int(row.lo), hi=int(row.hi), role=row.regime, ok=bool(row.ok),
                             members={c: round(w, 4) for c, w in row.members.items()}))
        selected[target] = dict(onset=info['onset'], n_all=len(group),
                                n_sup=int((group.regime == 'supporting').sum()),
                                n_mar=int((group.regime == 'competing').sum()), runs=runs)
    return selected


def pretty_formula(label):
    return ''.join(e + ('' if n == '1' else n) for e, n in re.findall(r'([A-Z][a-z]?)(\d+)', label))


def onset_shifts(configurations, vibrational, selected, vacancy_free_hull,
                 temperatures=TEMPERATURES, slope_window_K=SLOPE_WINDOW_K):
    """Reproduce Table 5's two approximations; neither is a full corrected hull."""
    T = np.asarray(temperatures, float)
    _, labels, G = _vacancy_free_energies(configurations, T)
    energies = dict(zip(labels, G))
    hull = vacancy_free_hull.set_index('formula')
    columns = [f'dGhull_{int(t)}K' for t in T]
    if not set(T).issubset(vibrational.index) or set(columns).difference(hull.columns):
        raise ValueError('Incomplete onset-shift temperature grid')
    rows = []
    for target, info in sorted(selected.items(), key=lambda item: item[1]['onset']):
        onset = float(info['onset'])
        planes = [r for r in sorted(info['runs'], key=lambda r: r['lo'])
                  if r['role'] == 'supporting' and r['lo'] <= onset <= r['hi']]
        if not planes:
            raise ValueError(f'{target}: no supporting window contains onset')
        members = planes[-1]['members']
        if set([target, *members]).difference(vibrational.columns):
            raise ValueError(f'{target}: missing vibrational endmember')
        D = energies[target] - sum(w * energies[c] for c, w in members.items())
        dF = (vibrational[target] - sum(w * vibrational[c] for c, w in members.items())).reindex(T).to_numpy()
        j = int(np.argmin(abs(T - onset)))
        if T[j] != onset or j == 0 or j == len(T) - 1:
            raise ValueError('Onset must be an interior grid temperature')
        dT = T[j + 1] - T[j - 1]
        dSconf = -(D[j + 1] - D[j - 1]) / dT / KB_EV
        dSvib = -(dF[j + 1] - dF[j - 1]) / dT / 1000 / KB_EV
        counts = dict(re.findall(r'([A-Z][a-z]?)(\d+)', target))
        label = formula_label([int(counts.get(e, 0)) for e in ELEMENTS])
        if label not in hull.index:
            raise ValueError(f'{target}: missing vacancy-free hull row')
        delta = hull.loc[label, columns].to_numpy(float)
        back = int(np.argmin(abs(T - (onset - slope_window_K))))
        slope = (delta[j] - delta[back]) / (T[j] - T[back])
        if abs(dSconf) < 1e-15 or abs(slope) < 1e-15:
            raise ValueError(f'{target}: undefined onset-shift slope')
        ratio = dSvib / dSconf
        rows.append(dict(composition=pretty_formula(target), onset_K=onset,
                         tie_plane=' + '.join(f'{w:.3f} {pretty_formula(c)}' for c, w in members.items()),
                         dFvib_at_onset_meV_atom=float(dF[j]), dSvib_1e3_kB_atom=dSvib * 1000,
                         dSconf_1e3_kB_atom=dSconf * 1000, ratio_pct=ratio * 100,
                         shift_eq6_K=onset * (1 / (1 + ratio) - 1), shift_direct_K=-dF[j] / slope))
    return pd.DataFrame(rows)


def scoped_statistics(vibrational, accounting, selected):
    """Cancellation and affine bounds over targets plus supporting endmembers only."""
    supporting = accounting[accounting.regime == 'supporting']
    if not supporting.ok.all():
        raise ValueError('Supporting coverage is incomplete')
    scope = sorted(set(selected) | {c for m in supporting.members for c in m})
    fractions = []
    for label in scope:
        parsed = dict(re.findall(r'([A-Z][a-z]?)(\d+)', label))
        counts = np.array([int(parsed.get(e, 0)) for e in ELEMENTS], float)
        fractions.append(counts / counts.sum())
    X = np.array(fractions)
    bound_rows, entropy_max = [], 0.
    for T in (0, 300, 1200):
        y = vibrational.loc[T, scope].to_numpy(float)
        residual = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        maximum = float(np.abs(residual).max())
        bound_rows.append(dict(T=T, max_affine_residual_meV_atom=maximum, bound_2g_meV_atom=2 * maximum))
    for row in supporting.itertuples():
        residual = vibrational[row.target] - sum(w * vibrational[c] for c, w in row.members.items())
        segment = residual.loc[row.lo:min(row.hi, selected[row.target]['onset'])]
        if len(segment) > 1:
            entropy_max = max(entropy_max, float(np.abs(np.gradient(segment, segment.index)).max() / (KB_EV * 1000)))
    return dict(n_scope=len(scope), n_targets=len(selected), n_supporting=len(supporting),
                max_residual_meV_atom=float(supporting.dmax.max()), max_entropy_kB_atom=entropy_max,
                affine_bounds=pd.DataFrame(bound_rows))
