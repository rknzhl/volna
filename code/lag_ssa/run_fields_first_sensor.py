"""Run the tangent-transport pipeline on CSV/TSV/NPY/NPZ sensor readings."""
import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from fields_first_io import read_sensor_data
from fields_first_sensor_pipeline import SensorPipeline


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path,value):
    """Keep the previous complete checkpoint if a computation is interrupted."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(jsonable(value),indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--query', type=Path, help='separate observed query block')
    parser.add_argument('--train-count', type=int, help='fit on this prefix, transform the suffix')
    parser.add_argument('--time-column', help='explicit header name or zero-based column index')
    parser.add_argument('--key', default='Y', help='NPZ sensor-array key')
    parser.add_argument('--delimiter', default=',')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-candidates',type=int,help='limit the training search and report unresolved if exhausted')
    parser.add_argument('--threads',type=int,default=1,help='maximum numerical-library threads (default: 1)')
    args = parser.parse_args()
    if args.query is not None and args.train_count is not None:
        parser.error('choose either --query or --train-count')
    column = args.time_column
    if column is not None and column.isdecimal():
        column = int(column)
    Y, times = read_sensor_data(args.input, time_column=column,
                               key=args.key, delimiter=args.delimiter)
    if args.query is not None:
        query, query_times = read_sensor_data(args.query, time_column=column,
                                              key=args.key, delimiter=args.delimiter)
        train = Y
        start = len(train)
        if times is not None and query_times is not None:
            if not np.isclose(np.median(np.diff(times)), np.median(np.diff(query_times)),
                              rtol=1e-6, atol=1e-12):
                parser.error('training and query sampling intervals differ')
    elif args.train_count is not None:
        start = args.train_count
        if not 1 < start < len(Y)-1:
            parser.error('--train-count must split the record into two nonempty blocks')
        train, query = Y[:start], Y[start:]
        query_times = None if times is None else times[start:]
    else:
        train, query, query_times, start = Y, Y, times, 0
    args.output.mkdir(parents=True, exist_ok=True)
    def checkpoint(summary):
        write_json(args.output/'fit.progress.json',summary)
    if args.threads<1:parser.error('--threads must be positive')
    with threadpool_limits(limits=args.threads):
        pipeline = SensorPipeline(max_candidates=args.max_candidates).fit(train,progress_callback=checkpoint)
        prediction = pipeline.transform(query, start_index=start)
    fit_summary = pipeline.summary()
    fit_summary['input'] = str(args.input.resolve())
    fit_summary['training_snapshot_count'] = len(train)
    fit_summary['query_snapshot_count'] = len(query)
    fit_summary['query_start_sample_index'] = start
    write_json(args.output/'fit.json',fit_summary)
    array_keys = ('reconstruction', 'residual_reconstruction', 'slow',
                  'conditioned_observations', 'raw_time_indices', 'phases',
                  'window_raw_time_indices')
    report = {k:v for k,v in prediction.items() if k not in array_keys}
    if prediction['status'] == 'transformed_conditionally':
        arrays = {k:prediction[k] for k in array_keys}
        if query_times is not None:
            arrays['time'] = query_times[prediction['raw_time_indices']-start]
            arrays['phase_time'] = query_times[prediction['window_raw_time_indices']-start]
        np.savez_compressed(args.output/'reconstruction.npz', **arrays)
    write_json(args.output/'transform.json',report)
    print(json.dumps({'fit_status':pipeline.status, 'transform_status':prediction['status'],
                      'output':str(args.output.resolve())}, indent=2))


if __name__ == '__main__':
    main()
