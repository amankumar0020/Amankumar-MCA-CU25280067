"""Train and compare classifiers for the synthetic loan approval dataset."""
from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "data/loan_data.csv").drop_duplicates()
X = df.drop(columns="Loan_Status")
y = df["Loan_Status"].astype(int)
num = X.select_dtypes(include="number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()
pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1500),
    "Decision Tree": DecisionTreeClassifier(max_depth=8,random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=250,max_depth=10,random_state=42,class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}
results=[]; fitted={}
for name, clf in models.items():
    pipe=Pipeline([("preprocessor",pre),("classifier",clf)])
    pipe.fit(X_train,y_train)
    pred=pipe.predict(X_test)
    metrics={"model":name,"accuracy":accuracy_score(y_test,pred),
             "precision":precision_score(y_test,pred,zero_division=0),
             "recall":recall_score(y_test,pred,zero_division=0),
             "f1_score":f1_score(y_test,pred,zero_division=0)}
    results.append(metrics); fitted[name]=pipe
best_name=max(results,key=lambda r:r["f1_score"])["model"]
best=fitted[best_name]
(BASE/"models").mkdir(exist_ok=True)
joblib.dump(best,BASE/"models/loan_approval_model.joblib")
pd.DataFrame(results).to_csv(BASE/"reports/model_comparison.csv",index=False)
pred=best.predict(X_test)
(BASE/"reports/classification_report.txt").write_text(
    f"Selected model: {best_name}\\n\\n"+classification_report(y_test,pred,zero_division=0))
fig,ax=plt.subplots(figsize=(5,4))
ConfusionMatrixDisplay.from_predictions(y_test,pred,ax=ax,cmap="Blues")
ax.set_title(f"Loan Approval — {best_name}")
fig.tight_layout(); fig.savefig(BASE/"reports/confusion_matrix.png",dpi=150); plt.close(fig)
print(pd.DataFrame(results).to_string(index=False))
print("Saved:",BASE/"models/loan_approval_model.joblib")
