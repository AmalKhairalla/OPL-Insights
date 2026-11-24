## Dataset

### Original Dataset — OpenPowerlifting

This project uses the [OpenPowerlifting dataset](https://data.openpowerlifting.org/), a publicly available dataset containing real-world powerlifting competition results.

The original dataset contains approximately **4.07 million records and 42 columns**, including athlete information (age, sex, and bodyweight), competition details (federation, equipment, event, and date), and performance measurements (squat, bench press, deadlift, and total weight lifted).

Due to its large size and variety of competition categories, the dataset was explored and filtered to select a suitable subset for our regression task.

### Selected Dataset — FPR Single-ply

Following exploratory analysis, we selected records from the **Powerlifting Federation of Russia (FPR)**, restricted to Single-ply equipment and SBD (Squat, Bench Press, Deadlift) competitions.

The final dataset contains **81,957 valid competition records**, with the following variables:

- **Target:** `Best3DeadliftKg`
- **Input features:** `Age`, `Sex`, `BodyweightKg`, `Best3SquatKg`, `Best3BenchKg`

The objective is to predict an athlete's best successful deadlift based on their physical characteristics and squat and bench press performance.

### Data Preprocessing

The selected dataset was divided into approximately **70% training, 15% validation, and 15% testing**, using group-based splitting by recorded athlete name to reduce data leakage from repeated competition records.

Missing age values were handled using median imputation, `Sex` was one-hot encoded, and numerical features were standardized using training-set statistics only. The target variable was also standardized for neural network training, with predictions converted back to kilograms for evaluation.

The finalized dataset and fixed split were saved for consistent and reproducible experiments.
