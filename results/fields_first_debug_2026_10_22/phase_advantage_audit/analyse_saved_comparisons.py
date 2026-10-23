"""Summarize saved comparisons; no fitting or scenario selection occurs here."""
import argparse
import json
from pathlib import Path

import numpy as np


def gauge_norms(matrix):
    a = np.asarray(matrix, dtype=int)
    inverse = np.rint(np.linalg.inv(a)).astype(int)
    if not np.array_equal(np.einsum('ij,jk->ik', a, inverse), np.eye(len(a))):
        raise ValueError('the matched integer gauge must be unimodular')
    return int(np.abs(a).sum(axis=0).max()), int(np.abs(inverse).sum(axis=0).max())


def analyse(comparisons):
    wins = {key: 0 for key in ('phase_vs_circular', 'phase_vs_toroidal',
        'readings_vs_ssa', 'readings_vs_mssa', 'train_step_vs_circular',
        'train_step_vs_toroidal')}
    setups, records = set(), []
    for comparison in comparisons:
        source = json.loads(Path(comparison['source']).read_text())
        methods = {row['method']: row for row in comparison['rows']}
        ours = methods['fields_first']
        config = comparison['configuration']
        setups.add(tuple((key, config.get(key)) for key in (
            'case', 'seed', 'noise', 'trend', 'phase_mode', 'rate_profile', 'ntrain')))
        _, inverse_norm = gauge_norms(source['train_phase_evaluation']['winding_candidate'])
        phase_upper_bound = inverse_norm * ours['phase_mae_rad']
        bounds = {}
        for name in ('circular', 'toroidal'):
            baseline = methods[name]
            norm, _ = gauge_norms(baseline['train_phase_evaluation']['winding_candidate'])
            lower_bound = baseline['phase_mae_rad'] / norm
            bounds[name] = dict(phase_mae_lower_bound_in_generator_basis=lower_bound,
                ours_phase_mae_upper_bound_in_generator_basis=phase_upper_bound,
                lower_bound_on_advantage_factor=lower_bound / phase_upper_bound,
                ranking_survives_integer_basis_change=lower_bound > phase_upper_bound)
            wins['phase_vs_' + name] += ours['phase_mae_rad'] < baseline['phase_mae_rad']
            wins['train_step_vs_' + name] += (
                ours['train_phase_step_rmse_rad'] < baseline['train_phase_step_rmse_rad'])
        for name in ('ssa', 'mssa'):
            wins['readings_vs_' + name] += ours['reconstruction_rmse'] < methods[name]['reconstruction_rmse']
        records.append(dict(source=comparison['source'], configuration=config,
            phase_mae_ours=ours['phase_mae_rad'],
            best_native_topology_phase_mae=min(methods[key]['phase_mae_rad']
                for key in ('circular', 'toroidal')),
            advantage_over_best_native_topology=min(methods[key]['phase_mae_rad']
                for key in ('circular', 'toroidal')) / ours['phase_mae_rad'],
            generator_basis_bounds=bounds))
    factors = [row['advantage_over_best_native_topology'] for row in records]
    return dict(saved_runs=len(records), distinct_configurations=len(setups),
        wins=wins, native_topology_advantage_factor_range=[min(factors), max(factors)],
        basis_bound='For wrapped coordinate MAE m: m/||A||_1 <= m_generator <= ||A^-1||_1 m; induced column-sum norms.',
        restrictions=['Only successful fields-first runs enter this comparison.',
            'Repeated query lengths are not independent runs.',
            'Training phase-step variation is a fields-first acceptance criterion.',
            'Native CC/TC do not enforce uniform dynamical phases.',
            'No phase-extraction SSA/MSSA baseline is included.'], records=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parents[1] / 'baselines/summary.json')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('saved_comparison_audit.json'))
    args = parser.parse_args()
    report = analyse(json.loads(args.input.read_text()))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('saved_runs', 'distinct_configurations', 'wins', 'native_topology_advantage_factor_range')}))
