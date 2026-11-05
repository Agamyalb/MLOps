import joblib
import numpy as np

model = joblib.load("model.joblib")

def test_prediction_type():
    pred = model.predict(np.zeros((1, 10)))
    assert isinstance(pred[0], (float, np.floating))

def test_output_shape():
    assert model.predict(np.zeros((5, 10))).shape == (5,)
