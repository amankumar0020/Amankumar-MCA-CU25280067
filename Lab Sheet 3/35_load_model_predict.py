
# Q35: Load saved model and predict new data
import joblib
from sklearn.datasets import load_diabetes
data = load_diabetes()
loaded_model = joblib.load("regression_model.pkl")
new_data = [data.data[0]]
prediction = loaded_model.predict(new_data)
print("Predicted Output:", prediction[0])
