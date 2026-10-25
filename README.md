<div align="center">
  <h1>DP-JMRNet</h1>
  <p><strong>Joint complex reconstruction for bitemporal SAR with discontinuous aperture sampling</strong></p>
</div>

## Overview

Discontinuous aperture sampling leaves gaps in the SAR phase history and makes high-quality reconstruction difficult. In a bitemporal setting, reconstructing the two acquisitions independently can further disturb their shared structure and amplify phase errors, reducing the reliability of the differential phase used for change analysis.

DP-JMRNet reconstructs both complex SAR images jointly with a five-stage deep-unfolding network. The revised model uses signed relative-phase features and learned spatial gates for the common and differential phase updates. It combines measurement-consistency evidence, local complex correlation, and candidate changes to decide how much of each phase update to retain, while preserving the joint candidate amplitudes.

This repository provides the reconstruction model, a matrix-free wideband SAR operator, training and validation scripts, numerical tests, a small example dataset, and the code required to reproduce the complete simulated dataset.

## Method

The model takes two incomplete complex phase histories acquired at different epochs and reconstructs the corresponding complex SAR images together. The known SAR acquisition geometry and aperture sampling pattern are used directly in the reconstruction rather than being approximated by a generic image-degradation model.

Each of the five unfolding stages performs four steps:

1. apply pre-DC and the shared real/imaginary residual prior;
2. construct exchange-equivariant interaction corrections from signed relative-phase features;
3. apply the same post-DC step to a base candidate and an interaction-aware joint candidate for each epoch;
4. split the candidate phase increments into common and differential components, then route them with two learned spatial gates.

Each stage has a 12-channel observation-state input and a small 12→8→2 convolutional router. The features include fixed initial and current coherence, normalized DC-gradient evidence, candidate changes, amplitude discrepancies, and the common/differential phase-increment magnitudes. The router inputs are detached, while its parameters are learned jointly with the backbone. Both gates start at 0.5 in the supplied configurations.

A gate value of one retains the corresponding joint phase increment; zero restores the corresponding base phase component. The joint amplitudes are retained at both endpoints. Consequently, setting both phase gates to zero is different from selecting the fully independent reconstruction branch.

After the five stages, the same phase-only refiner is applied to both epochs. Training uses the final complex, amplitude, individual circular-phase, and differential circular-phase losses. There is no auxiliary utility loss or gate smoothness/anti-collapse penalty in the revised training objective. The tests check epoch exchange numerically, including swaps of the observations and masks.

![One unfolding stage of revised DP-JMRNet](assets/architecture.svg)

See [architecture and routing features](docs/architecture.md) for module details and the meaning of all 12 router inputs.

The default public model is <code>SplitPhaseUtilityUnfolding256</code>. The previous <code>FixedMaskExchangeEquivariantSelectiveUnfolding256</code> remains available for legacy comparisons. Version 1 checkpoints use different endpoint inputs and gates and cannot be loaded directly into the revised model; use the matching legacy code or train a revised checkpoint.

## Repository structure

```text
.
├── configs/data/                   simulated-dataset specification
├── data/simulated_dataset/         bundled 32/8/8 example dataset
├── scripts/generate_dataset.py     complete dataset generator
├── scripts/train.py                training entry point
├── scripts/validate.py             checkpoint evaluation entry point
├── src/coherent_sar/               revised/legacy models, SAR operator, and pipeline
├── docs/architecture.md            phase-routing mechanism and feature definitions
├── assets/                        revised architecture diagrams
├── tests/                          numerical and architectural tests
├── config.json                     quick example configuration
└── config.full.json                complete-dataset training configuration
```

## Installation

Python 3.10 or newer is required. Install a PyTorch build suitable for your CPU or CUDA environment, then install the project:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Run the tests to verify the installation:

```bash
python -m pytest
```

## Dataset

The bundled example dataset contains 32 training samples, 8 validation samples, and 8 held-out samples. It uses the same format as the complete dataset and is intended for testing the installation and training pipeline.

The complete simulated dataset contains 4,000 bitemporal samples:

| Split | Samples | Parent scenes |
|:--|--:|--:|
| Train | 3,000 | 188 |
| Validation | 500 | 32 |
| Test | 500 | 32 |

Each sample contains two native `256 × 256` complex SAR images, wrapped differential-phase ground truth, phase-reliability weights, a validity mask, and generation metadata. Parent scenes are assigned to train, validation, and test before patches are cropped, preventing parent-scene overlap between splits.

Generate the complete dataset with:

```bash
python scripts/generate_dataset.py --mode full
```

The output is written to `data/final_sar_dataset_256/`. The generator also creates manifests, split statistics, integrity checks, per-sample hashes, and deterministic noise-seed banks.

For a smaller generator check, run:

```bash
python scripts/generate_dataset.py --mode dry
```

A custom output directory can be specified with `--output`:

```bash
python scripts/generate_dataset.py --mode full --output /path/to/dataset
```

## Training

Run a one-step end-to-end check with the bundled example dataset:

```bash
python scripts/train.py --steps 1
```

Standard example training:

```bash
python scripts/train.py --config config.json
```

After generating the complete dataset, run the complete-data configuration with:

```bash
python scripts/train.py --config config.full.json
```

Training writes `best.pt`, `last.pt`, and `metrics.json` to the configured output directory. A CUDA-capable GPU is recommended for training the full five-stage `256 × 256` model.

## Validation

Validate a checkpoint trained with the example configuration:

```bash
python scripts/validate.py --config config.json --checkpoint outputs/best.pt
```

For the complete-data configuration:

```bash
python scripts/validate.py --config config.full.json --checkpoint outputs/full_k104/best.pt
```

The validation script reports weighted differential-phase RMSE, amplitude NMSE, and complex NMSE.

## Reproducibility

- The dataset is generated from a versioned configuration and a fixed base seed.
- Train, validation, and test splits are separated at the parent-scene level.
- Manifests record the generation metadata and numerical truth hashes of every sample.
- Noise seeds are deterministic and stored independently for each split and epoch.
- The tests cover the complex adjoint relation, operator gradients, signed-feature exchange equivariance, phase-gate endpoints, amplitude preservation, detached evidence, router gradients, and the training/checkpoint pipeline.

## Scope

The provided dataset is a controlled coherent complex SAR simulation intended for reproducible method development. Evaluation on independently acquired SAR data is still necessary when assessing performance across sensors, terrain types, and acquisition conditions.

## License

Released under the [MIT License](LICENSE).
