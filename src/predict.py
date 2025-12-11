# Demo prediction for house price model
import joblib
from sklearn.datasets import fetch_california_housing

def main():
    model = joblib.load('model.joblib')
    data = fetch_california_housing(as_frame=True)
    X = data.data
    sample = X.iloc[:5]
    preds = model.predict(sample)
    print('Predictions for first 5 samples:', preds)

if __name__ == '__main__':
    main()
