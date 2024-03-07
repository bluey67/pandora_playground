from fastapi.testclient import TestClient

from app.main import predict, UnseenExample

from app.main import app

client = TestClient(app)


def test_predict_main():
    with TestClient(app) as client:
        response = client.post("/predict", json={"feat1": 12.6,"feat2": 4})
    assert response.status_code == 200
    assert response.json() == {
        "Prediction": 23.6
    }

#def test_predict():
    #unseen_example = UnseenExample(feat1=12.6, feat2=4)
    #assert predict(unseen_example) == {"Prediction": 23.6}

