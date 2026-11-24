# DS330 — Experiment Observations

**Scope:** Linear regression baseline, early-stopping neural-network baseline, and Member A experiments A01–A15.

**Methodology:** Same name-grouped train/validation split and training-only preprocessing; MSE measured on the original target scale. Neural networks use up to 100 epochs, patience 10, minimum meaningful validation improvement 0.1 kg²; lowest validation MSE determines the saved checkpoint.

**Note:** Observations are interpretations of the metrics and epoch summaries supplied in the conversation. Confirm statements about training stability or overfitting against saved learning curves. Only the baseline and A01 training-MSE figures were supplied, so other training MSEs are not included.

| Run | Topic | Setting | Best validation MSE (kg²) | Best epoch | Epochs | Observation |
|---|---|---|---:|---:|---:|---|
| BASE_LINEAR | Baseline | Linear regression | 358.80 | — | — | Reference model: validation MSE 358.80 kg². Used as the comparison benchmark for all neural networks. |
| BASE_NN_EARLY_STOPPING | Baseline | SGD; lr 0.01; batch 64; hidden (64,64,64); MSE | 320.41 | 41 | 51 | Baseline NN reduced validation MSE compared with linear regression. Best checkpoint was epoch 41; early stopping ended training at epoch 51. |
| A01 | Learning rate | lr = 0.0001 | 348.22 | 100 | 100 | Very small learning rate; still improving at the 100-epoch limit, with high validation MSE. Consistent with slow convergence under the training budget. |
| A02 | Learning rate | lr = 0.0003 | 329.68 | 99 | 100 | Improved substantially over A01, but best performance was near the 100-epoch cap and remained worse than baseline; suggests relatively slow convergence. |
| A03 | Learning rate | lr = 0.001 | 324.39 | 94 | 95 | Approached baseline validation performance, but required 94 epochs to reach its best checkpoint, indicating slower progress than the baseline learning rate. |
| A04 | Learning rate | lr = 0.003 | 321.48 | 85 | 95 | Validation MSE came close to baseline but best checkpoint was reached later (epoch 85); lower learning rate may need more epochs to settle. |
| A05 | Learning rate | lr = 0.03 | 321.23 | 18 | 28 | Reached near-baseline validation MSE by epoch 18 and stopped at 28; faster progress, with slightly worse best MSE than baseline. |
| A06 | Learning rate | lr = 0.1 | 321.60 | 10 | 20 | Reached its best validation score very early (epoch 10); faster apparent convergence, but slightly worse MSE than baseline. |
| A07 | Learning rate | lr = 0.3 | 329.17 | 18 | 28 | Larger learning rate produced worse validation MSE than moderate rates; oversized updates may have limited optimization quality. Confirm stability from the loss curve. |
| A08 | Optimizer | SGD momentum = 0.5 | 320.71 | 26 | 36 | Momentum 0.5 achieved validation performance very close to baseline at an earlier best epoch (26 vs 41). The small MSE difference is not conclusive. |
| A09 | Optimizer | SGD momentum = 0.9 | 320.70 | 21 | 31 | Momentum 0.9 reached near-baseline validation performance at epoch 21, suggesting faster convergence; result was virtually tied with momentum 0.5. |
| A10 | Optimizer | SGD momentum = 0.99 | 328.17 | 16 | 26 | High momentum gave worse validation MSE; excessive accumulated updates may impair stability at the fixed learning rate. Inspect the curve before claiming instability. |
| A11 | Optimizer | Adam (lr = 0.01) | 324.60 | 6 | 16 | Adam reached its best checkpoint quickly (epoch 6) but underperformed SGD at the same learning rate. This does not show Adam is generally worse at its own tuned rate. |
| A12 | Optimizer | AdamW (lr = 0.01) | 324.95 | 6 | 16 | AdamW reached best validation performance by epoch 6, but its MSE was higher than baseline; optimizer-specific learning rate/weight decay may matter. |
| A13 | Loss function | L1 loss | 322.44 | 24 | 34 | Training with L1 achieved a slightly worse validation MSE than training with MSE. All runs were still evaluated using original-scale MSE. |
| A14 | Loss function | Huber loss; delta = 0.5 | 321.65 | 62 | 63 | Huber delta 0.5 achieved validation MSE close to baseline, with its best checkpoint at epoch 62. Delta was applied to the standardized target. |
| A15 | Loss function | Huber loss; delta = 2.0 | 321.28 | 53 | 63 | Huber delta 2.0 was the strongest alternative loss configuration among A13–A15, though still slightly worse than baseline MSE training. |

## Key findings

- All 15 Member A experiments were reported as completed and saved.
- The baseline neural network (320.41 kg²) was slightly better than every Member A tuning run.
- The strongest Member A results were A09 (320.70 kg²) and A08 (320.71 kg²); this difference is too small to interpret as a dependable ranking without repeated trials.
- Learning-rate results show a speed/performance trade-off under the common early-stopping policy.
- Adam and AdamW were tested at learning rate 0.01 to change only the optimizer; their performance should not be generalized to other learning rates.
- Do not select the final model until Member B results and the shared baseline are compared. The test set must remain untouched until then.

## Sources

- Project dataset: https://data.openpowerlifting.org/
- PyTorch optimization: https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- PyTorch model checkpoints: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
