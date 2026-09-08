"""用独立写出的矩阵和概率公式检查既有计算，不作新效果实验。"""
import numpy as np
import pytest
from scipy.stats import chi2, multivariate_normal

from experiments.public_module.core import compute_scale_estimates, evaluate_holdout, derived_seed
from experiments.public_module.aggregate_public_module import paired_bootstrap_interval
from experiments.e0.e0formal.contract import load_frozen_contract
from experiments.e0.e0formal.linear import _build_leverage_blocks, _full_block_dpr, _scalar_trace_dpr
from experiments.e0.e0formal.statistics import paired_bootstrap_difference_of_medians


@pytest.mark.parametrize('seed', [4, 19, 61])
def test_dense_projector_matches_all_five_scale_formulas(seed):
    rng = np.random.default_rng(seed)
    design = rng.normal(size=(32, 6))
    projector = design @ np.linalg.solve(design.T @ design, design.T)
    residual = (np.eye(32) - projector) @ rng.normal(size=32)
    parts = ('unit', seed, 16)
    got = compute_scale_estimates(residual, design, 16, 1e-6, 17, parts)
    blocks = [slice(i, i + 2) for i in range(0, 32, 2)]
    expected_exact = sum(residual[b] @ np.linalg.solve(np.eye(2) - projector[b, b], residual[b]) for b in blocks) / 32
    probes = np.random.default_rng(derived_seed(17, 'public_module_trace', *parts)).choice([-1., 1.], size=(32, 16))
    projected = projector @ probes
    traces = np.clip([np.sum(probes[b] * projected[b]) / 16 for b in blocks], 0, 2 * (1 - 1e-6))
    permutation = np.random.default_rng(derived_seed(17, 'public_module_permutation', *parts)).permutation(16)
    scalar = lambda t: sum(np.dot(residual[b], residual[b]) / (1 - h / 2) for b, h in zip(blocks, t)) / 32
    expected = {'Raw': np.dot(residual, residual) / 32,
                'Global-DoF': np.dot(residual, residual) / 26,
                'DPR-Exact-Block': expected_exact,
                'DUC-K16': scalar(traces), 'Permuted-K16': scalar(traces[permutation])}
    for method, value in expected.items(): assert got.scales[method] == pytest.approx(value, rel=1e-11)
    assert got.diagnostics['exact_trace_sum'] == pytest.approx(6.)


@pytest.mark.parametrize('scale', [.1, 1., 7.])
def test_holdout_metrics_match_two_dimensional_gaussian(scale):
    residual = np.array([[1., -2.], [.3, .5], [-.1, .2]])
    got = evaluate_holdout(residual, scale)
    coverage = np.mean(np.sum(residual**2, axis=1) <= chi2.ppf(.95, 2) * scale)
    assert got['holdout_nll'] == pytest.approx(-multivariate_normal.logpdf(residual, cov=scale*np.eye(2)).mean())
    assert got['coverage95'] == coverage
    assert got['ce95'] == pytest.approx(abs(coverage - .95))
    assert got['absolute_log_scale_ratio'] == pytest.approx(abs(np.log(scale / np.mean(residual**2))))


def test_linear_press_and_dpr_are_distinct_correct_formulas():
    rng = np.random.default_rng(103)
    design = rng.normal(size=(30, 5)); observed = rng.normal(size=30)
    beta = np.linalg.lstsq(design, observed, rcond=None)[0]
    residual = observed - design @ beta
    q, _ = np.linalg.qr(design, mode='reduced')
    blocks = [slice(i, i+2) for i in range(0, 30, 2)]
    config = load_frozen_contract().config['formal_contract']['numerics']
    leverage, _ = _build_leverage_blocks(q, blocks, config)
    corrected = _full_block_dpr(residual, blocks, leverage)
    traces = np.array([np.trace(q[b] @ q[b].T) for b in blocks])
    scalar = _scalar_trace_dpr(residual, blocks, traces, 1e-6)
    for b in blocks:
        complement = np.eye(2) - q[b] @ q[b].T
        keep = np.ones(30, dtype=bool); keep[b] = False
        deleted_beta = np.linalg.lstsq(design[keep], observed[keep], rcond=None)[0]
        deleted_residual = observed[b] - design[b] @ deleted_beta
        np.testing.assert_allclose(np.linalg.solve(complement, residual[b]), deleted_residual, atol=1e-11)
        assert corrected[b] @ corrected[b] == pytest.approx(residual[b] @ np.linalg.solve(complement, residual[b]))
    np.testing.assert_allclose(scalar.reshape(-1, 2), residual.reshape(-1, 2)/np.sqrt(1-traces[:, None]/2))


def test_bootstrap_uses_paired_difference_of_medians_not_median_of_differences():
    raw = np.array([0., 2., 100.]); corrected = np.array([2., 1., 4.])
    got = paired_bootstrap_difference_of_medians(raw, corrected, repeats=41, seed=17)
    rng = np.random.default_rng(17)
    values = []
    for _ in range(41):
        indices = rng.integers(0, 3, size=3)
        values.append(np.median(corrected[indices]) - np.median(raw[indices]))
    assert got['upper_bound'] == np.quantile(values, .95, method='linear')
    assert got['observed_difference'] == np.median(corrected) - np.median(raw)
    assert got['observed_difference'] != np.median(corrected - raw)


@pytest.mark.parametrize('values,repeats', [([], 10), ([np.nan], 10), ([1.], 0)])
def test_cluster_bootstrap_rejects_invalid_input(values, repeats):
    with pytest.raises(ValueError): paired_bootstrap_interval(np.array(values), repeats, 1)
