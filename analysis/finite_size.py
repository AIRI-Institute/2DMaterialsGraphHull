"""How much configurational entropy a finite supercell misses.

Two separable pieces.

**Ideal (combinatorial) deficit -- exact, parameter-free.**  A cell with ``m`` metal and
``2m`` chalcogen sites realises ``Omega`` arrangements of a composition, giving
``s_cell = k ln(Omega) / N_atoms``.  The thermodynamic limit of the same concentrations is
the site-entropy expression

    s_inf = -k [ n_me sum_c x_c ln x_c  +  n_X sum_c y_c ln y_c ] / N_atoms

which is cell-size independent.  ``s_cell < s_inf`` always (Stirling), and the gap is the
finite-size underestimation.  It is a pure counting statement: no energies enter.

**Interaction factor -- calibrated, then assumed transferable.**  The realised
stabilisation ``T dS_conf = E_min - G(T)`` is below the ideal ceiling because low-energy
configurations dominate the partition function.  Define

    f(T) = [E_min - G(T)] / (k T ln(Omega) / N)          in [0, 1]

from the exhaustively enumerated small cell, then estimate a larger cell as
``f(T) * k T ln(Omega_large) / N_large``.  This assumes the energetic selectivity ``f``
does not change with cell size -- true if the ordering is short-ranged relative to the
smaller cell, optimistic otherwise.  Always report it as an estimate, and check ``f``
against any directly computed larger-cell value available.
"""

from __future__ import annotations

from math import lgamma, log

import numpy as np
import pandas as pd

from .thermodynamics import ELEMENTS, temperature_hulls

KB_EV = 8.617333262e-5  # eV/K


def ln_multinomial(counts) -> float:
    n = sum(counts)
    out = lgamma(n + 1)
    for c in counts:
        out -= lgamma(c + 1)
    return out


def ln_omega(metal_counts, chalc_counts) -> float:
    """``ln`` of the arrangement count for one composition in one cell."""
    return ln_multinomial(metal_counts) + ln_multinomial(chalc_counts)


def n_atoms_of(metal_counts, chalc_counts, vac_index_me=2, vac_index_x=2) -> int:
    return (sum(metal_counts) - metal_counts[vac_index_me]
            + sum(chalc_counts) - chalc_counts[vac_index_x])


def s_cell(metal_counts, chalc_counts) -> float:
    """Per-atom configurational entropy of the finite cell, in units of k."""
    return ln_omega(metal_counts, chalc_counts) / n_atoms_of(metal_counts, chalc_counts)


def s_infinite(metal_counts, chalc_counts) -> float:
    """Thermodynamic-limit per-atom site entropy of the same concentrations, in k."""
    def sub(counts):
        n = sum(counts)
        out = 0.0
        for c in counts:
            if c:
                x = c / n
                out -= x * log(x)
        return n * out
    return (sub(metal_counts) + sub(chalc_counts)) / n_atoms_of(metal_counts, chalc_counts)


def scale_composition(metal_counts, chalc_counts, n_from: int, n_to: int):
    """Same concentrations in an ``n_to x n_to x 1`` cell; None if not commensurate."""
    s = (n_to * n_to) / (n_from * n_from)
    out = []
    for counts in (metal_counts, chalc_counts):
        scaled = []
        for c in counts:
            v = c * s
            if abs(v - round(v)) > 1e-9:
                return None
            scaled.append(int(round(v)))
        out.append(tuple(scaled))
    return out[0], out[1]


def ts_ideal(metal_counts, chalc_counts, T: float) -> float:
    """``kT ln(Omega)/N`` in eV/atom -- the ceiling for this cell."""
    return KB_EV * T * s_cell(metal_counts, chalc_counts)


def ts_ideal_infinite(metal_counts, chalc_counts, T: float) -> float:
    return KB_EV * T * s_infinite(metal_counts, chalc_counts)


def deficit_table(metal_counts, chalc_counts, sizes=(2, 4, 6, 8), n_from: int = 2):
    """Per-cell-size ideal entropy and its shortfall against the thermodynamic limit."""
    s_inf = s_infinite(metal_counts, chalc_counts)
    rows = []
    for n in sizes:
        sc = scale_composition(metal_counts, chalc_counts, n_from, n)
        if sc is None:
            rows.append((n, None, None, None))
            continue
        s = s_cell(*sc)
        rows.append((n, s, s / s_inf if s_inf else 1.0, s_inf - s))
    return s_inf, rows


def _sublattices(row, base_size):
    metal = (int(row.W), int(row.Mo), base_size**2 - int(row.W) - int(row.Mo))
    chalc = (int(row.Se), int(row.S), 2 * base_size**2 - int(row.Se) - int(row.S))
    if min(metal + chalc) < 0:
        raise ValueError('Composition exceeds the specified base cell')
    if hasattr(row, 'n_vac_Me') and (metal[2] != row.n_vac_Me or chalc[2] != row.n_vac_X):
        raise ValueError('Vacancy counts disagree with the specified base cell')
    return metal, chalc


def entropy_deficits(metadata, base_size=2, sizes=(2, 4, 8, 12, None), temperature=1200):
    """Ideal entropy shortfall over vacancy-free quaternaries, in meV/atom."""
    qs = metadata[(metadata[ELEMENTS] > 0).all(axis=1)]
    if 'n_vac_Me' in qs:
        qs = qs[(qs.n_vac_Me == 0) & (qs.n_vac_X == 0)]
    if qs.empty:
        raise ValueError('No vacancy-free quaternaries for entropy statistics')
    rows = []
    for size in sizes:
        values = []
        for row in qs.itertuples():
            mc, cc = _sublattices(row, base_size)
            if size is None:
                values.append(0.)
                continue
            scaled = scale_composition(mc, cc, base_size, size)
            if scaled is None:
                raise ValueError(f'Composition not commensurate with {size}x{size}x1')
            values.append((s_infinite(mc, cc) - s_cell(*scaled)) * KB_EV * temperature * 1000)
        rows.append(dict(effective_cell='limit' if size is None else f'{size}x{size}x1',
                         entropy_deficit_mean_meV_atom=float(np.mean(values)),
                         entropy_deficit_sd_meV_atom=float(np.std(values)), n_quaternaries=len(qs)))
    return pd.DataFrame(rows)


def size_sensitivity(metadata, free_energies, temperatures, base_size=2, sizes=(2, 4, 8, 12, None)):
    """Rescale configurational stabilization and rebuild each effective-size hull.

    This transfers energetic selectivity from the base cell; it does not calculate
    new larger-cell energetics. Every selected composition must be commensurate.
    Returns the Table 4-style quaternary summary and full hull arrays by cell size.
    """
    G = np.asarray(free_energies, float)
    if len(temperatures) == 0 or temperatures[0] != 0:
        raise ValueError('Size sensitivity requires the T=0 baseline')
    quaternary = (metadata[ELEMENTS] > 0).all(axis=1).to_numpy()
    results, rows = {}, []
    for size in sizes:
        ratios = []
        for row in metadata.itertuples():
            mc, cc = _sublattices(row, base_size)
            entropy_base = s_cell(mc, cc)
            if size is None:
                entropy_target = s_infinite(mc, cc)
            else:
                scaled = scale_composition(mc, cc, base_size, size)
                if scaled is None:
                    raise ValueError(f'Composition not commensurate with {size}x{size}x1')
                entropy_target = s_cell(*scaled)
            ratios.append(entropy_target / entropy_base if entropy_base > 0 else 1.)
        shifted = G[:, :1] + np.asarray(ratios)[:, None] * (G - G[:, :1])
        hull = temperature_hulls(metadata, shifted, temperatures)
        results[size] = hull
        reached = hull['onsets'][quaternary]
        reached = reached[np.isfinite(reached)]
        rows.append(dict(effective_cell='limit' if size is None else f'{size}x{size}x1',
                         reach_hull=len(reached), first_onset_K=float(reached.min()) if len(reached) else np.nan,
                         median_onset_K=float(np.median(reached)) if len(reached) else np.nan))
    summary = pd.DataFrame(rows)
    if None in results:
        limit = summary[summary.effective_cell == 'limit'].iloc[0]
        summary['first_minus_limit_K'] = summary.first_onset_K - limit.first_onset_K
        summary['median_minus_limit_K'] = summary.median_onset_K - limit.median_onset_K
    deficits = entropy_deficits(metadata, base_size, sizes)
    return summary.merge(deficits, on='effective_cell'), results
