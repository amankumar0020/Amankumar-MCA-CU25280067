"""Train and compare symptom classifiers on a synthetic educational dataset."""
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

BASE=Path(__file__).resolve().parent
df=pd.read_csv(BASE/"data/disease_symptoms.csv").drop_duplicates()
X=df.drop(columns="disease")
y=df["disease"].astype(str)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
models={
 "Decision Tree":DecisionTreeClassifier(max_depth=12,random_state=42),
 "Random Forest":RandomForestClassifier(n_estimators=300,max_depth=14,random_state=42),
 "Extra Trees":ExtraTreesClassifier(n_estimators=300,max_depth=14,random_state=42),
 "Gradient Boosting":GradientBoostingClassifier(random_state=42)
}
results=[]; fitted={}
for name,clf in models.items():
    pipe=Pipeline([("imputer",SimpleImputer(strategy="median")),("classifier",clf)])
    pipe.fit(X_train,y_train)
    pred=pipe.predict(X_test)
    results.append({"model":name,"accuracy":accuracy_score(y_test,pred),
                    "precision_macro":precision_score(y_test,pred,average="macro",zero_division=0),
                    "recall_macro":recall_score(y_test,pred,average="macro",zero_division=0),
                    "f1_macro":f1_score(y_test,pred,average="macro",zero_division=0)})
    fitted[name]=pipe
best_name=max(results,key=lambda r:r["f1_macro"])["model"]
best=fitted[best_name]
(BASE/"models").mkdir(exist_ok=True)
joblib.dump(best,BASE/"models/disease_prediction_model.joblib")
pd.DataFrame(results).to_csv(BASE/"reports/model_comparison.csv",index=False)
pred=best.predict(X_test)
(BASE/"reports/classification_report.txt").write_text(
    f"Selected model: {best_name}\\n\\n"+classification_report(y_test,pred,zero_division=0))
fig,ax=plt.subplots(figsize=(9,7))
ConfusionMatrixDisplay.from_predictions(y_test,pred,ax=ax,xticks_rotation=45,cmap="Purples")
ax.set_title(f"Symptom Classifier — {best_name}")
fig.tight_layout(); fig.savefig(BASE/"reports/confusion_matrix.png",dpi=150); plt.close(fig)
print(pd.DataFrame(results).to_string(index=False))
print("Saved:",BASE/"models/disease_prediction_model.joblib")
