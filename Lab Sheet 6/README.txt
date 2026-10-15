MCA201P-AS07 — Lab Sheet 06
Mini Projects on Supervised Learning Applications

Contents
--------
1. student_performance.py
2. student_performance_dataset.csv
3. house_price.py
4. house_price_dataset.csv
5. requirements.txt

How to run in VS Code
---------------------
1. Extract this ZIP into a folder.
2. Open that folder in VS Code.
3. Open Terminal > New Terminal.
4. Install dependencies:
   pip install -r requirements.txt
5. Run either project:
   python student_performance.py
   python house_price.py

Each program compares Linear Regression and Random Forest Regression,
prints MAE/MSE/RMSE/R2, creates a plot PNG, saves a trained model .pkl,
and predicts an example record. Generated plot/model files appear in the
current working directory after running the script.

Important note
--------------
The included datasets are small illustrative examples created for practice,
not real-world datasets. The metrics are not reliable estimates of real-world
performance. For a stronger project, use a larger, authentic dataset and
document its source. House prices are expressed in lakh rupees.
