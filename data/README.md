# Dataset

## OpenPowerlifting Dataset

This project uses the OpenPowerlifting dataset.

### Data Source

The complete dataset is available from
the official OpenPowerlifting website:

[Download OpenPowerlifting Dataset](https://www.openpowerlifting.org/)

### Dataset Information

- Source: OpenPowerlifting
- Format: CSV
- Data Type: Tabular Data
- Task: Regression
- Target Variable: TotalKg
## Final Dataset Selection — FPR Single-ply

Following exploratory analysis of the OpenPowerlifting dataset, we selected competition records from the Powerlifting Federation of Russia (FPR), specifically the Single-ply equipment category and SBD events.

**Selected dataset:**
- Federation: FPR
- Equipment: Single-ply
- Event: SBD
- Valid records: 81,957
- Input features: BodyweightKg, Best3SquatKg, Best3BenchKg
- Target: Best3DeadliftKg

The dataset was selected because it provides a sufficiently large number of valid observations for training and evaluating deep neural networks.

Preliminary comparisons showed that polynomial regression reduced validation MSE by 7.91% compared with linear regression, while a basic PyTorch neural network achieved a 10.98% reduction.

These exploratory results suggest that nonlinear modeling may be beneficial for predicting deadlift performance.

The selected cleaned dataset was saved separately for reuse in the main DS330 project. The formal neural-network training and hyperparameter experiments will be conducted in a new notebook.
