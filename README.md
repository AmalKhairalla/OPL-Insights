# DS330 Shared Experiment Code

These files extract the working FPR Single-ply deadlift baseline into reusable modules.
They **do not** create a new dataset split or evaluate the test set.


## Colab quick start

```python
!git clone https://github.com/AmalKhairalla/OPL-Insights.git /content/OPL-Insights
%cd /content/OPL-Insights

from copy import deepcopy
from src.config import BASELINE_CONFIG
from src.data import load_fixed_split, prepare_data
from src.train import run_experiment
from src.results import save_result

train_df, val_df = load_fixed_split('data/FPR_SinglePly_fixed_split.csv.gz')
data = prepare_data(train_df, val_df)
config = deepcopy(BASELINE_CONFIG)
result = run_experiment(config, data)
print('Baseline validation MSE:', result['val_mse'])

# IMPORTANT: choose a persistent directory, e.g. Google Drive:
# from google.colab import drive
# drive.mount('/content/drive')
# save_result('A01', 'Learning rate', result,
#             '/content/drive/MyDrive/DS330_results/A')

# Example controlled experiment: only learning rate changes
config = deepcopy(BASELINE_CONFIG)
config['learning_rate'] = 0.001
result = run_experiment(config, data)
print(result['val_mse'])
```

## Notes

- Standardization and imputation are **fit only on the training split**.
- Sex uses one-hot encoding, and the target is standardized during NN training.
- Reported metrics are **MSE in original kg² units**, consistent with the baseline notebook.
- Each run initializes a fresh network, so changing a setting does not reuse weights.
- The default reference is 3×64 ReLU, SGD, learning rate 0.01, batch 64, MSE loss, 30 epochs, seed 42.
- Results include both final-epoch and best-epoch validation MSE; declare in advance which is used for comparing the 30 trials (recommend final-epoch to match the baseline notebook).
- The same code and split should be used by both teammates. Test baseline equivalence before launching the 15 runs per person. Exact numbers can differ slightly between CPU and GPU, library versions, and initialization/shuffling differences.
- For the final submission, ensure Colab can obtain the exact shared source version, not a moving GitHub branch.
- For two-category Sex, the encoder gives two one-hot columns, as it did in your original notebook.
- Experiment notebooks load only train and validation, so test values remain reserved.
- The saved split validates normalized `Name` group isolation, but names are imperfect athlete identifiers.

## Reproducibility

Record `torch.__version__`, `sklearn.__version__`, device, Git commit SHA, and the fixed dataset file checksum alongside final results. Do not change shared training code mid-experiment.
