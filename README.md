# 📘 **README — House Price Prediction (house_price_prediction)**

# House Price Prediction (California Housing Dataset)

This project demonstrates a regression model trained on the **California Housing** dataset from scikit-learn.  
A RandomForestRegressor is used to predict median house prices based on geographic & demographic features.

---

## 🚀 Features
- End-to-end regression workflow  
- RandomForest model with strong baseline performance  
- Outputs MSE and R² metrics  
- Includes prediction script with sample outputs  

---

## 📂 Project Structure
```text
house_price_prediction/
├── src/
│ ├── train.py
│ └── predict.py
├── requirements.txt
└── README.md
```


---

## 🔧 Installation
```bash
python -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install -r requirements.txt

```

---

## 🧠 Train the Model
```bash
python src/train.py
```

---

## 🔍 Predict Sample Values
```bash
python src/predict.py
```