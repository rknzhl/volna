"""Build article tables from archived experiment records; no model fitting."""
import csv
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'


def load(name):
    return json.loads((DATA / name).read_text())


def replace_table(marker, content):
    article = BASE.parent / 'paper/main.tex'
    text = article.read_text()
    pattern = rf'(% BEGIN {marker}\n).*?(% END {marker})'
    text, count = re.subn(pattern, lambda match: match.group(1) + content + '\n' + match.group(2), text, flags=re.S)
    if count != 1:
        raise ValueError(f'Missing unique table marker: {marker}')
    article.write_text(text)


def main():
    cases = [('scalar_noise10', 'One nonlinear sensor', 1),
             ('two_waves', 'Two waves', 12),
             *[(f'noise{n:02d}', 'Curved torus', 4) for n in (0, 1, 5, 10)]]
    rows = []
    for name, label, sensors in cases:
        run = load(name + '.json')
        accepted = next(a for a in reversed(run['fit']['attempts'])
                        if a.get('gates', {}).get('passed'))
        rows.append(dict(name=name, model=label, status='accepted', sensors=sensors,
                         noise_percent=100 * run['noise'],
                         phases=accepted['diagnostics']['dimension_selected'],
                         lag=run['fit']['selected_lag'],
                         phase_mae_rad=run['test_phase_evaluation']['mean_absolute_rad'],
                         signal_nrmse=run['reconstruction_against_clean_total']
                         ['train_channel_std_normalized_rmse']))
    for name, label in reversed([('toy_sine', 'Sine toy'), ('toy_torus', 'Two-wave toy')]):
        toy = load(name + '.json')
        if toy['status'] != 'transformed_conditionally':
            raise ValueError(f'{name} is not an accepted toy')
        rows.insert(0, dict(name=name, model=label, status=toy['status'],
                           sensors=toy['m'], noise_percent=0,
                           phases=toy['hat_r'], lag=toy['lag'],
                           phase_mae_rad=toy['phase_mae_rad'],
                           signal_nrmse=toy['sensor_nrmse']))
    with (BASE / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    diagnostic = load('toy_signal.json')
    archive_rows = rows + [dict(name='toy_signal', model='10:1 harmonic diagnostic',
                              status=diagnostic['status'], sensors=1,
                              noise_percent=0, phases=None, lag='4,8',
                              phase_mae_rad=None, signal_nrmse=None)]
    with (BASE / 'all_attempts.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=archive_rows[0].keys())
        writer.writeheader(); writer.writerows(archive_rows)
    header = r'''\begin{table}[H]
\centering\small
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lrrrrr}
\toprule
Signal & Sensors & Noise (\%) & $\widehat r$ & Phase error (rad) & Signal error (\%) \\
\midrule
'''
    lines = []
    for row in rows:
        phase = '--' if row['phase_mae_rad'] is None else f"{row['phase_mae_rad']:.5f}"
        signal = '--' if row['signal_nrmse'] is None else f"{100*row['signal_nrmse']:.3f}"
        dimension = '--' if row['phases'] is None else str(row['phases'])
        lines.append(f"{row['model']} & {row['sensors']} & {row['noise_percent']:g} & "
                     f"{dimension} & {phase} & {signal} \\\\")
    footer = r'''
\bottomrule
\end{tabular}
\caption{TorusFlow results. $\widehat r$ is the estimated phase count.
True phase counts are $1$ for scalar signals, $2$ for two waves and $3$
for the curved torus.}
\label{tab:experiment-results}
\end{table}
'''
    replace_table('RESULTS_TABLE', header + '\n'.join(lines) + footer)

    comparisons = load('baseline_compatibility.json')['records']
    scalar, waves = (row['metrics'] for row in comparisons)
    methods = [('TorusFlow', 'current'), ('SSA', 'ssa'), ('MSSA', 'mssa'),
               ('CC', 'circular'), ('TC', 'toroidal'),
               ('CC+CDF', 'circular_cdf'), ('TC+CDF', 'toroidal_cdf')]
    header = r'''\begin{table}[H]
\centering\small
\setlength{\tabcolsep}{9pt}
\begin{tabular}{lrrrr}
\toprule
& \multicolumn{2}{c}{Phase error (rad)} & \multicolumn{2}{c}{Signal error (\%)}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Method & Nonlinear & Two waves & Nonlinear & Two waves\\
\midrule
'''
    lines = []
    for label, key in methods:
        values = []
        for metric, digits, scale in [('phase_mae_rad', 5, 1), ('sensor_nrmse', 3, 100)]:
            for record in (scalar, waves):
                value = record.get(key + '_' + metric)
                values.append('--' if value is None else f'{scale*value:.{digits}f}')
        lines.append(label + ' & ' + ' & '.join(values) + r' \\')
    footer = r'''
\bottomrule
\end{tabular}
\caption{Comparison on the same test inputs and evaluated times.
The nonlinear and two-wave cases use $10\%$ and $5\%$ noise, respectively.
CDF calibration uses training data only.}
\label{tab:experiment-baselines}
\end{table}
'''
    replace_table('BASELINES_TABLE', header + '\n'.join(lines) + footer)
    print('Wrote current results and comparison tables from archived records.')


if __name__ == '__main__':
    main()
