import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

X, y = load_diabetes(return_X_y=True)

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.joblib")
print("Model saved successfully!")
