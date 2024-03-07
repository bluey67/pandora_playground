from main import predict, UnseenExample

def test_predict():
    unseen_example = UnseenExample(feat1=12.6, feat2=4)
    assert predict(unseen_example) == {"Prediction": 23.6}
