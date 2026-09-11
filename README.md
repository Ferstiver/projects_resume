# Titanic Survival Prediction

A student machine learning project using Python, Pandas, and scikit-learn to predict passenger survival on the Titanic.

## What I did

- Inspected passenger data and handled missing ages and embarkation values.
- Created `FamilySize` and `IsAlone` features.
- Used a pipeline with numerical scaling and one-hot encoding.
- Trained a Random Forest classifier.
- Compared six parameter combinations using five-fold cross-validation.
- Evaluated test accuracy and predicted survival for an example passenger.

## Run the project

1. Install the packages:

   ```sh
   python -m pip install -r requirements.txt
   ```

2. Put your labeled Titanic dataset in this folder as `titanic.csv`.
3. Run:

   ```sh
   python titanic.py
   ```

The CSV must contain `Survived`, `Pclass`, `Sex`, `Age`, `Fare`, `Embarked`, `SibSp`, `Parch`, and `Cabin`. The unlabeled competition test file cannot be used for training this script.

The dataset is not included and CSV files are gitignored. The exact source of the author's CSV has not been supplied; add its original source link and applicable terms before sharing the project as a complete portfolio entry.

## Understanding the output

- `Best`: the parameters selected by cross-validation.
- `Score`: mean cross-validation accuracy for the selected parameters on the training split.
- `Test accuracy`: accuracy on the held-out 20% split.
- The final arrays show the example passenger's predicted class and model probabilities. Probability columns follow `grid.classes_` (normally 0 = did not survive, 1 = survived).

No measured accuracy is claimed here because the script has not been run with the author's CSV.

## Learning notes

The code is preserved from my original project. Two points remain to improve:

- Missing ages are filled using the whole dataset's mean before splitting. This lets information from the test data influence preprocessing. A better evaluation fits age imputation inside the pipeline using only each training fold.
- The Random Forest has no fixed random seed, so results can vary between runs even though the train/test split is fixed.

StandardScaler is retained from the original code, although Random Forest does not require scaling. The example prediction is a model estimate, not a certainty.
