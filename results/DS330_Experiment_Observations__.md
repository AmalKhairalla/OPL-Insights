# DS330 — Experiment Observations

## Project overview

This report documents the baseline models and all 30 controlled hyperparameter experiments conducted by Member A (A01–A15) and Member B (B01–B15) for the DS330 deep-learning project.

The task is to predict athletes' best deadlift performance using OpenPowerlifting competition data. The experiments investigate how training and neural-network configuration choices affect validation performance in a tabular regression task.

## Scope

- Linear regression baseline.
- Early-stopping neural-network baseline.
- Member A: learning rate, optimizer, and loss-function experiments (A01–A15).
- Member B: batch-size, network-depth, and network-width experiments (B01–B15).

## Methodology

All experiments use the same name-grouped train/validation split and training-only preprocessing. The neural-network baseline uses SGD with learning rate 0.01, batch size 64, three hidden layers of 64 units each, and MSE loss.

The controlled experiments change only the setting assigned to that experiment while keeping the remaining baseline configuration unchanged. Training uses a maximum of 100 epochs, early-stopping patience of 10, and a minimum meaningful validation MSE improvement (`min_delta`) of 0.1 kg². The random seed is 42.

MSE is evaluated on the original target scale and is reported in kg². For each run, the checkpoint with the lowest validation MSE is selected. `Best epoch` identifies that checkpoint; `Epochs trained` is the number of epochs actually completed. `Stop reason` records whether the run stopped through early stopping or reached the maximum-epoch limit.

The validation set is used for model selection. The test set must remain untouched until the final model has been selected.

## Baseline results

| Run | Model / setting | Train MSE (kg²) | Validation MSE (kg²) |
|---|---|---:|---:|
| BASE_LINEAR | Linear regression | 341.94 | 358.80 |
| BASE_NN_EARLY_STOPPING | SGD; lr = 0.01; batch size = 64; hidden = (64, 64, 64); MSE | 305.53 | 320.41 |

The baseline neural network reduced validation MSE from 358.80 kg² to 320.41 kg² compared with linear regression, an improvement of approximately 10.70%.

---

## Member A — Learning Rate, Optimizer, and Loss Function

Member A evaluated 15 configurations. Each experiment changed the learning rate, optimizer/momentum, or loss function while retaining the other baseline settings.

| Run | Topic | Setting | Best validation MSE (kg²) | Best epoch | Epochs trained | Stop reason | Observation |
|---|---|---|---:|---:|---:|---|---|
| A01 | Learning rate | lr = 0.0001 | 348.22 | 100 | 100 | maximum_epochs | Very small learning rate; still improving at the 100-epoch limit, with high validation MSE. Consistent with slow convergence under the training budget. |
| A02 | Learning rate | lr = 0.0003 | 329.68 | 99 | 100 | maximum_epochs | Improved substantially over A01, but best performance was near the 100-epoch cap and remained worse than baseline; suggests relatively slow convergence. |
| A03 | Learning rate | lr = 0.001 | 324.39 | 94 | 95 | early_stopping | Approached baseline validation performance, but required 94 epochs to reach its best checkpoint, indicating slower progress than the baseline learning rate. |
| A04 | Learning rate | lr = 0.003 | 321.48 | 85 | 95 | early_stopping | Validation MSE came close to baseline but best checkpoint was reached later (epoch 85); lower learning rate may need more epochs to settle. |
| A05 | Learning rate | lr = 0.03 | 321.23 | 18 | 28 | early_stopping | Reached near-baseline validation MSE by epoch 18 and stopped at 28; faster progress, with slightly worse best MSE than baseline. |
| A06 | Learning rate | lr = 0.1 | 321.60 | 10 | 20 | early_stopping | Reached its best validation score very early (epoch 10); faster apparent convergence, but slightly worse MSE than baseline. |
| A07 | Learning rate | lr = 0.3 | 329.17 | 18 | 28 | early_stopping | Larger learning rate produced worse validation MSE than moderate rates; oversized updates may have limited optimization quality. Confirm stability from the loss curve. |
| A08 | Optimizer | SGD momentum = 0.5 | 320.71 | 26 | 36 | early_stopping | Momentum 0.5 achieved validation performance very close to baseline at an earlier best epoch (26 vs 41). The small MSE difference is not conclusive. |
| A09 | Optimizer | SGD momentum = 0.9 | 320.70 | 21 | 31 | early_stopping | Momentum 0.9 reached near-baseline validation performance at epoch 21, suggesting faster convergence; result was virtually tied with momentum 0.5. |
| A10 | Optimizer | SGD momentum = 0.99 | 328.17 | 16 | 26 | early_stopping | High momentum gave worse validation MSE; excessive accumulated updates may impair stability at the fixed learning rate. Inspect the curve before claiming instability. |
| A11 | Optimizer | Adam (lr = 0.01) | 324.60 | 6 | 16 | early_stopping | Adam reached its best checkpoint quickly (epoch 6) but underperformed SGD at the same learning rate. This does not show Adam is generally worse at its own tuned rate. |
| A12 | Optimizer | AdamW (lr = 0.01) | 324.95 | 6 | 16 | early_stopping | AdamW reached best validation performance by epoch 6, but its MSE was higher than baseline; optimizer-specific learning rate/weight decay may matter. |
| A13 | Loss function | L1 loss | 322.44 | 24 | 34 | early_stopping | Training with L1 achieved a slightly worse validation MSE than training with MSE. All runs were still evaluated using original-scale MSE. |
| A14 | Loss function | Huber loss; delta = 0.5 | 321.65 | 62 | 63 | early_stopping | Huber delta 0.5 achieved validation MSE close to baseline, with its best checkpoint at epoch 62. Delta was applied to the standardized target. |
| A15 | Loss function | Huber loss; delta = 2.0 | 321.28 | 53 | 63 | early_stopping | Huber delta 2.0 was the strongest alternative loss configuration among A13–A15, though still slightly worse than baseline MSE training. |

### Member A — Key observations

- All 15 Member A experiments completed and their metrics were recorded.
- The baseline neural network (320.41 kg²) performed slightly better than every Member A tuning run.
- The strongest Member A results were A09 (320.70 kg²) and A08 (320.71 kg²); the difference is too small to interpret as a dependable ranking without repeated trials.
- Learning-rate results show a speed/performance trade-off under the common early-stopping policy.
- Adam and AdamW were tested at learning rate 0.01. Their results should not be generalized to other learning rates.
- The alternative loss functions did not outperform the baseline MSE configuration in these runs.

---

## Member B — Batch Size, Network Depth, and Network Width

Member B evaluated 15 configurations across three topics. Batch-size experiments changed only the batch size. Depth experiments changed the number of hidden layers while keeping 64 units per layer. Width experiments changed the number of units while keeping three hidden layers.

### B.1 Batch-size experiments

| Run | Topic | Setting | Best validation MSE (kg²) | Best epoch | Epochs trained | Stop reason | Observation |
|---|---|---|---:|---:|---:|---|---|
| B01 | Batch size | batch size = 16 | 320.11 | 22 | 32 | early_stopping | Batch size 16 achieved the best validation MSE among the batch-size experiments (320.11 kg²), slightly outperforming the baseline. Its best checkpoint was reached at epoch 22. |
| B02 | Batch size | batch size = 32 | 320.42 | 22 | 32 | early_stopping | Batch size 32 produced validation performance almost identical to the baseline (320.42 vs. 320.41 kg²). Its best checkpoint was reached at epoch 22. |
| B03 | Batch size | batch size = 128 | 320.21 | 90 | 100 | early_stopping | Batch size 128 achieved validation MSE close to the baseline, but its best checkpoint occurred late at epoch 90. Training continued through 100 epochs before stopping. |
| B04 | Batch size | batch size = 256 | 321.88 | 95 | 100 | early_stopping | Increasing the batch size to 256 resulted in worse validation MSE than the baseline. The best checkpoint occurred late, at epoch 95. |
| B05 | Batch size | batch size = 512 | 323.66 | 95 | 100 | maximum_epochs | Batch size 512 produced higher validation error than smaller batch sizes and reached the 100-epoch training limit. |
| B06 | Batch size | batch size = 1024 | 325.56 | 97 | 100 | maximum_epochs | Batch size 1024 further increased validation MSE. The best checkpoint was near the end of training, and the run reached the 100-epoch limit. |
| B07 | Batch size | batch size = 2048 | 329.33 | 99 | 100 | maximum_epochs | The largest batch size tested (2048) had the worst validation MSE among the batch-size experiments. Its best checkpoint occurred at epoch 99, and training reached the 100-epoch limit. |

**Batch-size interpretation:** The smaller tested batch sizes (16 and 32) gave the strongest validation results in this group. Validation MSE generally became worse as batch size increased beyond 128, with batch sizes 512–2048 reaching the maximum training budget. This pattern applies to the tested configurations and does not establish that small batches are always better.

### B.2 Network-depth experiments

| Run | Topic | Setting | Best validation MSE (kg²) | Best epoch | Epochs trained | Stop reason | Observation |
|---|---|---|---:|---:|---:|---|---|
| B08 | Depth | hidden layers = (64) | 321.86 | 50 | 60 | early_stopping | A single hidden layer produced worse validation MSE than the three-layer baseline. The best checkpoint was reached at epoch 50, so reducing network depth did not improve performance in this run. |
| B09 | Depth | hidden layers = (64, 64) | 320.34 | 39 | 49 | early_stopping | Two hidden layers achieved validation MSE very close to the baseline (320.34 vs. 320.41 kg²). Reducing depth from three layers caused no meaningful improvement. |
| B10 | Depth | hidden layers = (64, 64, 64, 64) | 319.77 | 32 | 42 | early_stopping | Four hidden layers achieved the best validation MSE among the depth experiments (319.77 kg²), slightly outperforming the baseline. The best checkpoint was reached at epoch 32. |
| B11 | Depth | hidden layers = (64, 64, 64, 64, 64, 64) | 321.60 | 21 | 31 | early_stopping | Increasing depth to six hidden layers worsened validation MSE compared with the baseline and the four-layer model. Although the best checkpoint occurred early, greater depth did not improve validation performance. |

**Depth interpretation:** The four-hidden-layer configuration (B10) was the strongest depth setting and the best experiment overall. A single layer and six layers performed worse than the baseline, while two layers were very close to it. The results suggest that adding depth helped up to four layers in the tested configurations, but further increasing depth did not help.

### B.3 Network-width experiments

| Run | Topic | Setting | Best validation MSE (kg²) | Best epoch | Epochs trained | Stop reason | Observation |
|---|---|---|---:|---:|---:|---|---|
| B12 | Width | hidden layers = (16, 16, 16) | 324.11 | 27 | 37 | early_stopping | Using 16 neurons per hidden layer produced the worst validation MSE among the width experiments, indicating that this narrower network performed less effectively than the baseline. |
| B13 | Width | hidden layers = (32, 32, 32) | 322.03 | 29 | 39 | early_stopping | Increasing the width to 32 neurons per layer improved validation MSE compared with the 16-neuron model, but it remained worse than the baseline. |
| B14 | Width | hidden layers = (128, 128, 128) | 321.13 | 17 | 27 | early_stopping | A width of 128 neurons per layer reached its best checkpoint quickly (epoch 17) and improved on the narrower networks, but its validation MSE remained slightly above the baseline. |
| B15 | Width | hidden layers = (256, 256, 256) | 319.83 | 21 | 26 | early_stopping | The 256-neuron network achieved the best validation MSE among the width experiments (319.83 kg²), slightly better than the baseline. It reached its best checkpoint at epoch 21 and was the second-best Member B configuration, behind B10. |

**Width interpretation:** The narrowest networks (16 and 32 units per layer) had higher validation error. Increasing width to 128 improved the result, and 256 units per layer produced the strongest width result (319.83 kg²). However, B15 remained slightly behind B10 (319.77 kg²).

### Member B — Key observations

- All 15 Member B experiments completed and their metrics were recorded.
- B10 (four hidden layers of 64 units) achieved the best Member B validation MSE: 319.77 kg².
- B15 (three hidden layers of 256 units) was the second-best Member B result: 319.83 kg².
- B01 (batch size 16) was the strongest batch-size configuration at 320.11 kg².
- In the tested batch-size configurations, larger batches generally produced higher validation MSE.
- For depth, four hidden layers performed best; increasing to six layers did not improve the result.
- For width, the 256-unit configuration performed best among the tested widths, while the narrower 16- and 32-unit configurations performed worse.
- These findings describe the tested settings and should not be generalized beyond them without additional trials.

---

## Combined results and key findings

### Best results by experiment group

| Group | Best run | Setting | Validation MSE (kg²) |
|---|---|---|---:|
| Baseline neural network | BASE_NN_EARLY_STOPPING | hidden = (64, 64, 64); batch = 64 | 320.41 |
| Member A | A09 | SGD momentum = 0.9 | 320.70 |
| Member B | B10 | hidden = (64, 64, 64, 64) | 319.77 |
| Overall tuning experiments | B10 | Four hidden layers of 64 units | 319.77 |

### Overall interpretation

- All 30 controlled hyperparameter experiments (A01–A15 and B01–B15) were completed and included in the combined results.
- B10 achieved the lowest validation MSE among the 30 tuning experiments (319.77 kg²), narrowly outperforming B15 (319.83 kg²).
- The best tuning result, B10, also improved on the baseline neural network validation MSE (320.41 kg²) and the linear-regression baseline (358.80 kg²).
- Member B produced the strongest tuning result in this set. Member A's strongest result was A09 at 320.70 kg².
- The differences between the top neural-network configurations are small. Because the experiments use one fixed random seed, these small differences should not be treated as conclusive evidence of a robust ranking without repeated runs.
- Validation performance is used for model selection only. The test set must remain untouched until the final model and configuration are chosen.
- Training stability or overfitting should be discussed only after inspecting the corresponding saved learning curves; validation MSE and epoch counts alone are not sufficient to prove either.

## Sources

- Project dataset: https://data.openpowerlifting.org/
- OpenPowerlifting: https://www.openpowerlifting.org/
- PyTorch optimization tutorial: https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- PyTorch model checkpoint guidance: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
