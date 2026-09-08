"""汇总前校验结果结构和代数关系；不改变有效数据的筛选或统计公式。"""
from collections import Counter, defaultdict
import hashlib
import csv
import io
import json
import math
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def validate_sequence_rows(rows, path, sequence, config, metrics):
    methods = set(config['calibration']['methods'])
    budgets = set(config['split']['fit_budgets'])
    groups = defaultdict(dict)
    probability = float(config['calibration']['coverage_probability'])
    seen = set()
    for row in rows:
        try:
            budget = int(row['fit_budget'])
            pair, method = row['pair_id'], row['method']
            flag = row['valid'].strip().lower()
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'{sequence}: 结果身份或状态字段无效') from error
        if (row.get('sequence') != sequence or not pair or budget not in budgets
                or method not in methods or flag not in ('true', 'false')
                or row.get('provenance') != config['provenance']):
            raise ValueError(f'{sequence}: 序列、方法、规模、状态或来源不匹配')
        key = (pair, budget, method)
        if key in seen: raise ValueError(f'{sequence}: 重复结果行 {pair}, N={budget}, {method}')
        seen.add(key)
        groups[(pair, budget)][method] = row
        if flag == 'false':
            if not row.get('failure_reason', '').strip():
                raise ValueError(f'{sequence}: 无效结果缺少原因')
            continue
        if row.get('failure_reason', '').strip():
            raise ValueError(f'{sequence}: 有效结果不应同时含失败原因')
        try:
            values = {name: float(row[name]) for name in metrics}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'{sequence}: 有效行缺少数值指标') from error
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f'{sequence}: 有效行指标含非有限值')
        if (not 0 <= values['coverage95'] <= 1 or not 0 <= values['ce95'] <= 1
                or values['estimated_scale'] <= 0
                or any(value < 0 for name, value in values.items() if name != 'holdout_nll')):
            raise ValueError(f'{sequence}: 指标超出定义范围')
        if not math.isclose(values['ce95'], abs(values['coverage95'] - probability), abs_tol=1e-10):
            raise ValueError(f'{sequence}: CE95与覆盖率不一致')
        if row.get('holdout_scale', '') != '':
            held = float(row['holdout_scale'])
            scale = values['estimated_scale']
            if not math.isfinite(held) or held <= 0: raise ValueError(f'{sequence}: 留出方差无效')
            expected = {'holdout_nll': math.log(2 * math.pi * scale) + held / scale,
                        'absolute_log_scale_ratio': abs(math.log(scale / held))}
            if any(not math.isclose(values[k], v, rel_tol=1e-10, abs_tol=1e-10) for k, v in expected.items()):
                raise ValueError(f'{sequence}: 留出指标与方差不一致')

    pair_sets = defaultdict(set)
    valid_counts = Counter()
    for (pair, budget), group in groups.items():
        if set(group) != methods: raise ValueError(f'{sequence}: 帧对的方法组不完整')
        flags = {row['valid'].strip().lower() for row in group.values()}
        if len(flags) != 1: raise ValueError(f'{sequence}: 共同有效集合不一致')
        pair_sets[budget].add(pair)
        if flags == {'true'}: valid_counts[str(budget)] += 1
    if not groups or set(pair_sets) != budgets or len({frozenset(v) for v in pair_sets.values()}) != 1:
        raise ValueError(f'{sequence}: 拟合集规模或候选帧对不完整')

    summary_path = Path(path).with_name('summary.json')
    if not summary_path.is_file(): raise ValueError(f'{sequence}: 缺少运行摘要，无法核验结果完整性')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    required = {'protocol_version': 'v1.5', 'profile': 'formal', 'sequence': sequence,
                'provenance': config['provenance'], 'config_sha256': config['_config_sha256'],
                'result_csv_sha256': file_hash(path),
                'candidate_pair_count': len(next(iter(pair_sets.values())))}
    if any(summary.get(key) != value for key, value in required.items()):
        raise ValueError(f'{sequence}: 运行摘要与结果或配置不匹配')
    counts = {str(budget): valid_counts[str(budget)] for budget in sorted(budgets)}
    if 'valid_pairs_by_budget' in summary and summary['valid_pairs_by_budget'] != counts:
        raise ValueError(f'{sequence}: 摘要有效帧对数不匹配')
    return counts


def load_reviewed_aggregate(gate_path, summary_path):
    """图表必须使用同一次汇总的审查记录和统计表。"""
    gate = json.loads(Path(gate_path).read_text(encoding='utf-8'))
    raw = Path(summary_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != gate.get('sequence_summary_sha256'):
        raise ValueError('图表统计表与审查记录的哈希不一致')
    if gate.get('protocol_version') != 'v1.6' or gate.get('status') != 'PASS' or gate.get('primary_fit_budget') != 16:
        raise ValueError('图表需要主规模N=16的v1.6通过记录')
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    sequences = [effect['sequence'] for effect in gate['sequence_effects']]
    methods = {'Raw', 'Global-DoF', 'DPR-Exact-Block', 'DUC-K16', 'Permuted-K16'}
    expected = {(sequence, budget, method) for sequence in sequences for budget in (16, 48) for method in methods}
    actual = [(row['sequence'], int(row['fit_budget']), row['method']) for row in rows]
    if len(sequences) != 5 or len(set(sequences)) != 5 or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('图表统计表的序列、规模或方法组不完整')
    return gate, rows
