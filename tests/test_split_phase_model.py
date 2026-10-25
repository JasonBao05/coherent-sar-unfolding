from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from coherent_sar import SplitPhaseUtilityUnfolding256
from coherent_sar import pipeline
from coherent_sar.formal_wideband_operator import (
    MatrixFreeChunkedWidebandSAROperator,
    WidebandSARGroundPlaneGeometry,
)
from coherent_sar.split_phase_utility_model import SplitPhaseRouter, route_split_phases


def _operator() -> MatrixFreeChunkedWidebandSAROperator:
    return MatrixFreeChunkedWidebandSAROperator(
        WidebandSARGroundPlaneGeometry(
            image_height=32, image_width=32,
            pixel_spacing_ground_range_m=0.5, pixel_spacing_cross_range_m=0.5,
            aperture_count=24, frequency_count=20,
            aperture_length_m=1.6, bandwidth_hz=160.0e6,
            aperture_chunk_size=7,
        )
    )


def _masks() -> tuple[torch.Tensor, torch.Tensor]:
    first, second = torch.zeros(24), torch.zeros(24)
    first[[0, 2, 5, 8, 11, 14, 17, 20, 23]] = 1
    second[[0, 3, 6, 9, 12, 15, 18, 21, 23]] = 1
    return first, second


def _model(mask: torch.Tensor, *, checkpointing: bool = False):
    return SplitPhaseUtilityUnfolding256(
        mask, aperture_count=24, stages=2, hidden_channels=8,
        gradient_checkpointing=checkpointing, initial_probability=0.5,
    )


def _candidates():
    # Include updates crossing the principal-angle branch cut.
    phases = [[3.05, -3.05, 0.2], [-2.9, 2.9, -0.6],
              [-3.04, 3.04, 0.6], [3.02, -3.02, -1.0]]
    amplitudes = [[1., 2., 3.], [2., 3., 4.], [3., 4., 5.], [4., 5., 6.]]
    return [torch.polar(torch.tensor(a, dtype=torch.float64),
                        torch.tensor(p, dtype=torch.float64))[None, None]
            for a, p in zip(amplitudes, phases)]


@pytest.mark.parametrize('value', [0., 1.])
def test_phase_gate_endpoints_preserve_joint_amplitudes(value):
    b1, b2, j1, j2 = _candidates()
    gate = torch.full((1, 2, 1, 3), value, dtype=torch.float64)
    out1, out2 = route_split_phases(b1, b2, j1, j2, gate)
    for out, base, joint in [(out1, b1, j1), (out2, b2, j2)]:
        expected = joint if value == 1 else joint.abs() * base / base.abs()
        torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(out.abs(), joint.abs(), rtol=1e-12, atol=1e-12)
    if value == 0:
        assert not torch.allclose(out1, b1)


def test_common_gate_cancels_from_the_stage_differential_phase():
    values = _candidates()
    gate = torch.full((1, 2, 1, 3), 0.4, dtype=torch.float64)
    first = gate.clone()
    second = gate.clone()
    first[:, 0] = 0
    second[:, 0] = 1
    a1, a2 = route_split_phases(*values, first)
    b1, b2 = route_split_phases(*values, second)
    torch.testing.assert_close(a2 * a1.conj(), b2 * b1.conj(), rtol=1e-12, atol=1e-12)


def test_router_starts_at_half_and_detaches_only_the_evidence():
    router = SplitPhaseRouter(initial_probability=0.5)
    features = torch.randn(2, 12, 8, 8, requires_grad=True)
    gate, _ = router(features)
    torch.testing.assert_close(gate, torch.full_like(gate, 0.5), rtol=0, atol=0)
    gate.sum().backward()
    assert features.grad is None
    assert router.network[-1].weight.grad is not None
    assert router.network[-1].weight.grad.abs().sum() > 0


def test_revised_forward_and_router_features_respect_epoch_exchange():
    torch.manual_seed(925)
    operator = _operator()
    mask1, mask2 = _masks()
    truth1 = torch.complex(torch.randn(1, 32, 32), torch.randn(1, 32, 32))
    truth2 = torch.complex(torch.randn(1, 32, 32), torch.randn(1, 32, 32))
    y1, y2 = operator.forward_masked(truth1, mask1), operator.forward_masked(truth2, mask2)
    model = _model(mask1).eval()
    with torch.no_grad():
        out1, out2, normal = model(y1, y2, operator, mask1=mask1, mask2=mask2,
                                  return_diagnostics=True)
        swap2, swap1, swapped = model(y2, y1, operator, mask1=mask2, mask2=mask1,
                                     return_diagnostics=True)
    torch.testing.assert_close(out1, swap1, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(out2, swap2, rtol=2e-6, atol=2e-6)
    for state, other in zip(normal['stages'], swapped['stages']):
        assert state['features'].shape == (1, 12, 32, 32)
        assert state['gate'].shape == (1, 2, 32, 32)
        torch.testing.assert_close(state['features'], other['features'], rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(state['gate'], other['gate'], rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(state['anchor_gamma'], normal['stages'][0]['anchor_gamma'])
    assert model.interactions[0].endpoint.net[0].in_channels == 4


@pytest.mark.parametrize('checkpointing', [False, True])
def test_revised_backpropagation_trains_each_router_and_keeps_mask_frozen(checkpointing):
    torch.manual_seed(926)
    operator = _operator()
    mask, _ = _masks()
    model = _model(mask, checkpointing=checkpointing).train()
    truth1 = torch.complex(torch.randn(1, 32, 32), torch.randn(1, 32, 32))
    truth2 = torch.complex(torch.randn(1, 32, 32), torch.randn(1, 32, 32))
    out1, out2 = model(operator.forward_masked(truth1, mask),
                       operator.forward_masked(truth2, mask), operator)
    loss = (out1 - truth1).abs().square().mean() + (out2 - truth2).abs().square().mean()
    loss.backward()
    assert model.fixed_mask.requires_grad is False
    assert model.fixed_mask.grad is None
    torch.testing.assert_close(model.fixed_mask, mask)
    for router in model.utility_routers:
        gradients = [p.grad for p in router.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum().item() for g in gradients) > 0
    assert model.phase_refiner[-1].weight.grad is not None


def test_independent_control_removes_cross_epoch_dependence():
    torch.manual_seed(927)
    operator = _operator()
    mask, _ = _masks()
    model = _model(mask).eval()
    y1 = operator.forward_masked(torch.complex(torch.randn(1, 32, 32), torch.randn(1, 32, 32)), mask)
    y2a = operator.forward_masked(torch.complex(torch.randn(1, 32, 32), torch.randn(1, 32, 32)), mask)
    y2b = operator.forward_masked(torch.complex(torch.randn(1, 32, 32), torch.randn(1, 32, 32)), mask)
    with torch.no_grad():
        out_a, _ = model(y1, y2a, operator, coupling_override='independent')
        out_b, _ = model(y1, y2b, operator, coupling_override='independent')
    torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)


@pytest.mark.parametrize('name', ['config.json', 'config.full.json'])
def test_public_configs_select_the_revised_four_term_objective(name):
    config = json.loads((Path(__file__).resolve().parents[1] / name).read_text())
    model = pipeline.build_model(config, torch.device('cpu'))
    assert isinstance(model, SplitPhaseUtilityUnfolding256)
    assert model.signed_phase_features
    assert model.utility_routers[0].network[0].in_channels == 12
    assert model.utility_routers[0].network[-1].out_channels == 2
    assert config['model']['initial_probability'] == 0.5
    assert set(config['loss']) == {'complex_weight', 'amplitude_weight',
                                   'raw_phase_weight', 'differential_phase_weight'}


def test_training_validation_and_checkpoint_roundtrip_use_no_auxiliary_loss(tmp_path, monkeypatch):
    base = Path(__file__).resolve().parents[1]
    config = json.loads((base / 'config.json').read_text())
    mask, _ = _masks()
    config['sampling'] = {'aperture_candidates': 24, 'sample_count': 9,
                          'mask_indices': mask.nonzero().flatten().tolist()}
    config['model'].update(stages=2, hidden_channels=8, gradient_checkpointing=True)
    config['output_directory'] = 'outputs'
    config['training']['batch_size'] = 1
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))

    class TinyDataset:
        records = {'train': [0, 1], 'validation': [0, 1]}

        def batch(self, split, indices, device, snr_db, noise_state):
            generator = torch.Generator().manual_seed(noise_state)
            shape = (len(indices), 32, 32)
            def random_complex(shape=shape):
                return torch.complex(torch.randn(shape, generator=generator),
                                     torch.randn(shape, generator=generator)).to(device)
            return {'x1': random_complex(), 'x2': random_complex(),
                    'noise1': random_complex((len(indices), 24, 20)),
                    'noise2': random_complex((len(indices), 24, 20)),
                    'weight': torch.ones(shape, device=device),
                    'snr': torch.as_tensor(snr_db, device=device).expand(len(indices))}

    def forbidden(*args, **kwargs):
        raise AssertionError('The revised pipeline must use only reconstruction loss')

    monkeypatch.setattr(pipeline, 'SimulatedSARDataset', lambda root: TinyDataset())
    monkeypatch.setattr(pipeline, 'MatrixFreeChunkedWidebandSAROperator', _operator)
    monkeypatch.setattr(SplitPhaseUtilityUnfolding256, 'gate_regularization', forbidden)
    monkeypatch.setattr(SplitPhaseUtilityUnfolding256, 'utility_supervision_loss', forbidden)
    metrics = pipeline.train(config_path, steps=1, device_name='cpu')
    checkpoint = tmp_path / 'outputs' / 'best.pt'
    assert checkpoint.exists()
    payload = torch.load(checkpoint, weights_only=False)
    assert payload['model_type'] == 'split_phase'
    assert any(k.startswith('utility_routers.') for k in payload['model_state'])
    replay = pipeline.validate_checkpoint(config_path, checkpoint, device_name='cpu')
    assert replay == metrics
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
