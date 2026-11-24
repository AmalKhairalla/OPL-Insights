# Powerlifting Performance Analytics

### Predicting Deadlift Performance Using Deep Learning

**DS330 — Deep Learning | Project Part 1: Train and Tune a Deep Neural Network**

This project investigates how different neural network training configurations affect predictive performance on a real-world tabular regression dataset. Using OpenPowerlifting competition data, we predict an athlete's best successful deadlift based on physical characteristics and results from the squat and bench press.

The project compares linear regression with a deep neural network implemented in PyTorch and evaluates 30 controlled hyperparameter experiments to study convergence, training stability, and generalization.

## Main Notebook

**[Open the Project Notebook in Google Colab](notebooks/DS330_P1_Team[Rahaf_Amal].ipynb)**

The notebook contains data preparation, exploratory data analysis, preprocessing, baseline models, hyperparameter experiments, learning curves, model selection, and final test evaluation.

*Note: Update the notebook filename in this link if the final submitted filename differs.*

## Dataset

**Source:** [OpenPowerlifting](https://data.openpowerlifting.org/)

The dataset contains 81,957 competition records selected from the Powerlifting Federation of Russia (FPR), restricted to Single-ply equipment and full-power (SBD) competitions.

**Prediction target:** `Best3DeadliftKg` — the athlete's best successful deadlift in kilograms.

**Input features:**

| Feature | Description |
|---|---|
| Age | Athlete's age |
| Sex | Athlete's recorded sex category |
| BodyweightKg | Athlete's body weight in kilograms |
| Best3SquatKg | Best successful squat in kilograms |
| Best3BenchKg | Best successful bench press in kilograms |

The model estimates deadlift performance using the other lifting results from the same competition record. It is therefore a performance-estimation task rather than a pre-competition forecasting model.

## Data Preparation

The dataset was divided into training, validation, and test subsets using group-based splitting by normalized recorded athlete name, reducing direct overlap of repeated athlete names between subsets.

Preprocessing includes:

- Median imputation for missing Age values.
- One-hot encoding of Sex.
- Standardization of numerical input features.
- Standardization of the training target for neural network optimization.
- Conversion of predictions back to kilograms when evaluating model performance.

Preprocessing parameters were fitted using training data only. The test set was reserved for final model evaluation.

## Baseline Models

Two baseline models were established:

1. **Linear Regression:** A reference regression model without hidden layers.
2. **Deep Neural Network:** Three hidden layers of 64 neurons each, ReLU activation, SGD optimizer, learning rate 0.01, batch size 64, and MSE loss.

The neural network uses validation-based early stopping with:

| Parameter | Value |
|---|---|
| Maximum epochs | 100 |
| Patience | 10 |
| Minimum meaningful improvement | 0.1 kg² |
| Checkpoint selection | Lowest validation MSE |
| Random seed | 42 |

### Baseline Results

| Model | Validation MSE (kg²) |
|---|---:|
| Linear Regression | 358.80 |
| Baseline Neural Network | 320.41 |

The neural-network baseline reduced validation MSE by approximately **10.70%** compared with linear regression.

## Hyperparameter Experiments

A total of **30 controlled hyperparameter experiments** were conducted, with 15 experiments completed by each team member.

Each experiment modifies one setting relative to the neural-network baseline while preserving the shared dataset, preprocessing rules, random-seed policy, and early-stopping settings.

| Experiment topic | Number of runs |
|---|---:|
| Learning rate | 7 |
| Optimizer | 5 |
| Loss function | 3 |
| Batch size | 7 |
| Network depth | 4 |
| Network width | 4 |
| **Total** | **30** |

The experiments investigate how hyperparameter choices affect validation MSE, convergence, training duration, and possible overfitting or underfitting.

For each experiment, the notebook records the training and validation MSE, best checkpoint epoch, total epochs trained, runtime, and stopping reason. Training histories and model checkpoints are saved to Google Drive.

## Results and Final Model

The best configuration was selected using validation MSE without using the test set for hyperparameter selection.

**Selected model: Experiment B10**

| Hyperparameter | Selected setting |
|---|---|
| Hidden layers | [64, 64, 64, 64] |
| Activation | ReLU |
| Optimizer | SGD |
| Learning rate | 0.01 |
| Batch size | 64 |
| Training loss | MSE |
| Early stopping | Patience 10, maximum 100 epochs |

### Final Results

| Metric | Result |
|---|---:|
| Best validation MSE | **319.77 kg²** |
| Improvement over linear regression on validation MSE | **10.88%** |
| Final test MSE | **314.28 kg²** |
| Final test MAE | **13.60 kg** |
| Test records | **12,478** |

The selected model achieved the lowest validation MSE among the tested configurations. The differences between the leading models were relatively small, indicating that additional architectural complexity did not necessarily produce large improvements.

## Reproducibility

The project is implemented using Python, PyTorch, scikit-learn, pandas, NumPy, Matplotlib, and Google Colab.

To reproduce the workflow:

1. Open the main notebook in Google Colab.
2. Run the dataset loading and preprocessing cells.
3. Train the baseline models and execute the controlled experiments.
4. Review the results tables and learning curves.
5. Select the best configuration using validation results.
6. Perform the final test evaluation only after model selection.

Completed experiment metrics, learning histories, and checkpoints can be saved to Google Drive to avoid unnecessary retraining during development. Access to saved Drive files requires the corresponding Google account or shared permissions.

## Repository Organization

```text
OPL-Insights/
├── data/             # Dataset and fixed split
├── notebooks/        # Final Colab notebook
│   └── archive/      # Earlier exploratory and baseline notebooks
├── results/          # Experiment observations and results
├── reports/          # Project report
├── requirements.txt
└── README.md
```

## Project Deliverables

The DS330 project includes:

- One executable Google Colab notebook with saved outputs.
- A results table containing at least 30 controlled experiments.
- Training and validation learning curves.
- Validation-based model selection and one final test evaluation.
- A project report of no more than three pages.
- An explanation of experimental choices, findings, and limitations.

## Acknowledgments and AI Usage

The dataset was obtained from [OpenPowerlifting](https://www.openpowerlifting.org/).

AI assistance, including ChatGPT, was used to support code development, debugging, experiment organization, and documentation. The team is responsible for reviewing, executing, understanding, and explaining the implementation and results.

## Limitations

Repeated competition entries are grouped using recorded athlete names, which may not perfectly identify unique individuals. The dataset represents FPR Single-ply SBD competitions, so results may not generalize to other federations or equipment categories. Additionally, the model uses squat and bench results from the same competition, limiting its applicability to pre-competition prediction.
