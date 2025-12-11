# Train a house-price regression model using California housing dataset
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def main():
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print('MSE:', mean_squared_error(y_test, preds))
    print('R2:', r2_score(y_test, preds))

    joblib.dump(model, 'model.joblib')
    print('Saved model to model.joblib')

if __name__ == '__main__':
    main()
