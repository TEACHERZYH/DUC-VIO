"""发布复核：只使用归档指标的小样本与模拟执行，不运行完整实验。"""
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.public_module.config import load_config
from experiments.public_module.aggregate_public_module import aggregate
from scripts import run_e0

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'experiments/contracts/public_module_config_v1.5.json'


@pytest.fixture
def small_archive(tmp_path):
    config = load_config(CONFIG)
    config['statistics']['cluster_bootstrap_repeats'] = 8
    root = tmp_path / 'input'
    for sequence in config['dataset']['sequences']:
        source = ROOT / 'experiments/artifacts/public_module_formal_v1_5' / sequence
        with (source / 'pair_method_results.csv').open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        pair = next(row['pair_id'] for row in rows if row['valid'] == 'True')
        rows = [row for row in rows if row['pair_id'] == pair]
        folder = root / sequence
        folder.mkdir(parents=True)
        write_rows(folder / 'pair_method_results.csv', rows)
        write_summary(folder, config, rows)
    return config, root, tmp_path / 'result'


def write_rows(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(folder, config, rows):
    summary = {
        'protocol_version': 'v1.5', 'profile': 'formal',
        'provenance': config['provenance'], 'sequence': folder.name,
        'candidate_pair_count': len({r['pair_id'] for r in rows}),
        'config_sha256': config['_config_sha256'],
        'result_csv_sha256': hashlib.sha256((folder / 'pair_method_results.csv').read_bytes()).hexdigest(),
    }
    (folder / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')


@pytest.mark.parametrize('defect', ['duplicate', 'missing_method', 'wrong_sequence',
                                  'wrong_method', 'nan', 'blank_metric', 'boolean', 'bad_ce95'])
def test_rejects_corrupt_metric_rows_before_writing(small_archive, defect):
    config, root, output = small_archive
    folder = root / config['dataset']['sequences'][0]
    path = folder / 'pair_method_results.csv'
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    valid = next(row for row in rows if row['valid'] == 'True')
    if defect == 'duplicate': rows.append(dict(valid))
    elif defect == 'missing_method': rows.remove(valid)
    elif defect == 'wrong_sequence': valid['sequence'] = 'wrong'
    elif defect == 'wrong_method': valid['method'] = 'unknown'
    elif defect == 'nan': valid['holdout_nll'] = 'nan'
    elif defect == 'blank_metric': valid['ce95'] = ''
    elif defect == 'boolean': valid['valid'] = 'truthy'
    elif defect == 'bad_ce95': valid['ce95'] = '0.73'
    write_rows(path, rows)
    write_summary(folder, config, rows)  # 重新生成摘要哈希，单独检验行结构。
    with pytest.raises(ValueError): aggregate(config, root, output)
    assert not output.exists()


def test_rejects_result_hash_mismatch(small_archive):
    config, root, output = small_archive
    path = root / config['dataset']['sequences'][0] / 'pair_method_results.csv'
    with path.open('a', encoding='utf-8') as stream: stream.write('\n')
    with pytest.raises(ValueError): aggregate(config, root, output)


@pytest.mark.parametrize('profile,kinds,shards,index', [
    ('unknown', ['linear'], 1, 0), ('smoke', [], 1, 0),
    ('smoke', ['bad'], 1, 0), ('smoke', ['linear', 'linear'], 1, 0),
    ('full', ['nonlinear'], 201, 200),
])
def test_e0_invalid_request_creates_no_output(tmp_path, monkeypatch, profile, kinds, shards, index):
    def forbidden(*args, **kwargs): pytest.fail('非法输入不应调用数值内核')
    monkeypatch.setattr(run_e0, 'run_linear_graph', forbidden)
    monkeypatch.setattr(run_e0, 'run_nonlinear_graph', forbidden)
    output = tmp_path / 'bad-request'
    with pytest.raises(ValueError): run_e0.run(output, profile, kinds, index, shards)
    assert not output.exists()


def test_e0_exception_keeps_incomplete_receipt(tmp_path, monkeypatch):
    def interrupted(*args, **kwargs): raise OSError('test infrastructure interruption')
    monkeypatch.setattr(run_e0, 'run_linear_graph', interrupted)
    output = tmp_path / 'interrupted'
    with pytest.raises(OSError): run_e0.run(output, 'smoke', ['linear'])
    receipt = json.loads((output / 'run_receipt.json').read_text(encoding='utf-8'))
    assert receipt['execution_status'] == 'INCOMPLETE_EXCEPTION'
    assert receipt['counts']['linear'] == 0
    assert receipt['exception_type'] == 'OSError'
    assert receipt['replaces_archived_paper_results'] is False


def test_mid_method_exception_does_not_emit_partial_group(tmp_path, monkeypatch):
    from experiments.public_module import run_public_module as runner
    config = load_config(CONFIG)
    records = [SimpleNamespace(timestamp_ns=i, filename=str(i)) for i in range(50)]
    monkeypatch.setattr(runner, 'locate_sequence', lambda *_: tmp_path)
    monkeypatch.setattr(runner, 'read_image_index', lambda *_: records)
    monkeypatch.setattr(runner, 'frame_pair_start_indices', lambda *_: [0])
    monkeypatch.setattr(runner.StereoRectifier, 'from_sequence', lambda *_: SimpleNamespace(rectified_camera=np.eye(3)))
    monkeypatch.setattr(runner, 'extract_tracks', lambda *_: SimpleNamespace(
        detected_count=100, valid_disparity_count=100, tracked_count=100,
        points_3d=np.ones((100, 3)), target_points_2d=np.ones((100, 2))))
    solution = SimpleNamespace(success=True, reason='OK', nfev=1, rank=6, eta=0.,
        fit_min_depth_m=1., holdout_min_depth_m=1., parameters=np.zeros(6),
        residual=np.ones(32), jacobian=np.ones((32, 6)))
    monkeypatch.setattr(runner, 'solve_relative_pose', lambda *_: solution)
    monkeypatch.setattr(runner, 'reprojection_residual', lambda *args: np.ones(80))
    monkeypatch.setattr(runner, 'project_points', lambda *args: (np.ones((40, 2)), np.ones(40)))
    methods = config['calibration']['methods']
    monkeypatch.setattr(runner, 'compute_scale_estimates', lambda *_: SimpleNamespace(
        scales=dict.fromkeys(methods, 1.), timings_ms=dict.fromkeys(methods, .1), diagnostics={}))
    calls = []
    def fail_second(*args):
        calls.append(1)
        if len(calls) % 2 == 0: raise ValueError('injected failure')
        return {'ce95': .1}
    monkeypatch.setattr(runner, 'evaluate_holdout', fail_second)
    monkeypatch.setattr(runner, 'sha256_file', lambda *_: 'test-hash')
    output = tmp_path / 'atomic'
    runner.run_sequence(config, tmp_path, 'V1_01_easy', output, 'tiny', 1)
    with (output / 'pair_method_results.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert all(row['valid'] == 'False' for row in rows)


def test_zero_valid_pairs_remain_unavailable_without_nan(small_archive):
    config, root, output = small_archive
    from experiments.public_module.aggregate_public_module import METRICS
    for folder in root.iterdir():
        path = folder / 'pair_method_results.csv'
        with path.open(encoding='utf-8', newline='') as stream: rows = list(csv.DictReader(stream))
        for row in rows:
            row.update(valid='False', failure_reason='INSUFFICIENT_TRACKS')
            row.update({name: '' for name in METRICS})
        write_rows(path, rows); write_summary(folder, config, rows)
    assert aggregate(config, root, output) == 0
    content = (output / 'gate_review.json').read_text(encoding='utf-8')
    gate = json.loads(content)
    assert 'NaN' not in content and 'Infinity' not in content
    assert gate['status'] == 'FAIL_FINAL'
    assert all(v is None for v in gate['equal_sequence_median_differences'].values())
    assert all(v is None for v in gate['descriptive_cluster_bootstrap_95_percentile_intervals'].values())


@pytest.mark.parametrize('defect', ['empty', 'bad_tolerance'])
def test_comparison_rejects_unusable_input(tmp_path, defect):
    from experiments.public_module.compare_runs import compare
    path = tmp_path / 'compare.csv'
    path.write_text('sequence,pair_id,fit_budget,method,ce95\n', encoding='utf-8')
    if defect == 'bad_tolerance':
        with path.open('a', encoding='utf-8') as stream: stream.write('V1,p,16,Raw,0.05\n')
    with pytest.raises(ValueError): compare(path, path, rtol=-1 if defect == 'bad_tolerance' else 1e-8, atol=1e-10)


def test_changed_scientific_config_is_rejected(tmp_path):
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    config['split']['master_seed'] += 1
    changed = tmp_path / 'changed.json'
    changed.write_text(json.dumps(config), encoding='utf-8')
    with pytest.raises(ValueError): load_config(changed)


@pytest.mark.parametrize('profile,maximum', [('other', None), ('formal', 1), ('tiny', None), ('tiny', True)])
def test_euroc_bad_profile_leaves_no_output(tmp_path, profile, maximum):
    from experiments.public_module.run_public_module import run_sequence
    output = tmp_path / 'bad-profile'
    with pytest.raises(ValueError): run_sequence(load_config(CONFIG), tmp_path, 'V1_01_easy', output, profile, maximum)
    assert not output.exists()


def test_plot_inputs_cannot_mix_review_and_summary(tmp_path):
    from experiments.public_module.result_validation import load_reviewed_aggregate
    folder = ROOT / 'experiments/artifacts/public_module_aggregate_v1_6'
    gate = folder / 'gate_review.json'
    summary = folder / 'sequence_method_summary.csv'
    assert len(load_reviewed_aggregate(gate, summary)[1]) == 50
    altered = tmp_path / 'summary.csv'
    altered.write_bytes(summary.read_bytes() + b'\n')
    with pytest.raises(ValueError): load_reviewed_aggregate(gate, altered)


def test_missing_dataset_does_not_leave_output(tmp_path):
    from experiments.public_module.run_public_module import run_sequence
    from experiments.public_module.euroc import EuRoCDataError
    output = tmp_path / 'missing'
    with pytest.raises(EuRoCDataError): run_sequence(load_config(CONFIG), tmp_path, 'V1_01_easy', output, 'tiny', 1)
    assert not output.exists()


def test_full_entry_preserves_frozen_arguments_and_partial_rows(tmp_path, monkeypatch):
    contract = run_e0.load_frozen_contract()
    identities = list(run_e0.iter_instance_ids(contract, 'linear'))[:2]
    monkeypatch.setattr(run_e0, 'iter_instance_ids', lambda *_: iter(identities))
    calls = []
    def fake(config, setting, instance, **kwargs):
        assert set(kwargs) == {'process_session'}
        kwargs['process_session'].verify(config, identities[instance].stable_id)
        calls.append(instance)
        if instance == 1: raise OSError('injected interruption')
        return {'stable_id': identities[instance].stable_id, 'provenance': 'designed_synthetic_e0',
                'status': 'PASS', 'main_status': 'PASS'}
    monkeypatch.setattr(run_e0, 'run_linear_graph', fake)
    output = tmp_path / 'partial'
    with pytest.raises(OSError): run_e0.run(output, 'full', ['linear'])
    receipt = json.loads((output / 'run_receipt.json').read_text(encoding='utf-8'))
    path = output / 'linear.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert calls == [0, 1] and receipt['counts']['linear'] == 1
    assert receipt['expected_counts']['linear'] == 2
    assert receipt['files_sha256']['linear.jsonl'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert rows[0]['science_claim_eligible'] is False
    assert rows[0]['provenance'] == 'public_reproduction_e0'
