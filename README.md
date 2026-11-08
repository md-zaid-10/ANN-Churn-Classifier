# ANN Project: Customer Churn & Salary Prediction

Two artificial neural networks (ANNs) built with TensorFlow/Keras on the [Bank Customer Churn dataset](Churn_Modelling.csv), each served through an interactive Streamlit app:

1. **Churn classification** predicts whether a bank customer is likely to leave (`Exited`).
2. **Salary regression** predicts a customer's `EstimatedSalary`.

**🚀 Live demos**

- Churn prediction: https://ann-churn-classifier-fbm4rkdtphwwchu47e57xf.streamlit.app/
- Estimated salary prediction: https://ann-churn-classifier-1.streamlit.app/

## Project structure

| File | Description |
| --- | --- |
| `Churn_Modelling.csv` | Dataset: 10,000 bank customers with demographics, account details and churn label |
| `experiments.ipynb` | Preprocessing and training for the churn classification model |
| `prediction.ipynb` | Loading the saved churn model and encoders to predict on a single example |
| `salaryRegression.ipynb` | Preprocessing, training and evaluation for the salary regression model |
| `app.py` | Streamlit app for churn prediction |
| `streamlit_regression.py` | Streamlit app for salary prediction |
| `model.h5` | Trained churn classification model |
| `regression_model.h5` | Trained salary regression model |
| `label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`, `scaler.pkl` | Fitted preprocessors for the churn model |
| `label_encoder_gender_reg.pkl`, `onehot_encoder_geo_reg.pkl`, `scaler_reg.pkl` | Fitted preprocessors for the regression model |
| `requirements.txt` | Pinned Python dependencies |

## Data preprocessing

Both notebooks apply the same pipeline:

- Drop identifier columns: `RowNumber`, `CustomerId`, `Surname`
- Label-encode `Gender`
- One-hot encode `Geography` (France, Germany, Spain)
- 80/20 train-test split (`random_state=42`)
- Standardize features with `StandardScaler`

The fitted encoders and scalers are pickled so the Streamlit apps can transform user input exactly as the training data was transformed.

## Models

Both networks share the same architecture: `Dense(64, relu)` → `Dense(32, relu)` → output layer, trained for up to 100 epochs with early stopping (`patience=10`, best weights restored) and TensorBoard logging.

| | Churn classifier | Salary regressor |
| --- | --- | --- |
| Target | `Exited` (0/1) | `EstimatedSalary` |
| Output layer | `Dense(1, sigmoid)` | `Dense(1)` (linear) |
| Loss | Binary cross-entropy | Mean absolute error |
| Optimizer | Adam (lr = 0.01) | Adam (default lr) |
| Result | ~86% validation accuracy | ~50,458 test MAE |

> Note: the salary regressor's MAE is high relative to the salary range, since `EstimatedSalary` in this dataset is only weakly related to the other features.

## Getting started

Requires Python 3.11 (TensorFlow 2.15 does not support newer versions).

```bash
# create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

### Run the apps

```bash
# churn prediction
streamlit run app.py

# salary prediction
streamlit run streamlit_regression.py
```

Each app opens in your browser at `http://localhost:8501`. Adjust the customer's details (geography, gender, age, balance, credit score, tenure, products, etc.) and the prediction updates automatically:

- **Churn app**: shows the churn probability and says the customer is likely to churn when it is above 0.5.
- **Salary app**: shows the predicted estimated salary.

### Retrain the models

Run `experiments.ipynb` (churn) or `salaryRegression.ipynb` (salary) top to bottom. This regenerates the `.h5` model and `.pkl` preprocessor files. Training curves can be viewed with TensorBoard:

```bash
tensorboard --logdir logs/fit             # churn model
tensorboard --logdir regressionlogs/fit   # salary model
```

## Tech stack

TensorFlow / Keras · scikit-learn · pandas · NumPy · Streamlit · Matplotlib
