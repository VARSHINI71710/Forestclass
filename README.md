Forest Cover Type Prediction 🌲

App link- (https://forestclass-0-6.streamlit.app/)

This project predicts the forest cover type (classes 0–6) based on environmental and geographical features using XGBoost. The model is trained on a tabular dataset and deployed with Streamlit for interactive prediction.

📌 Features Used

Numerical Features:

Elevation

Aspect

Slope

Horizontal Distance to Hydrology

Vertical Distance to Hydrology

Horizontal Distance to Roadways

Hillshade 9AM, Noon, 3PM

Horizontal Distance to Fire Points

Categorical Features (One-hot encoded):

Wilderness Area (1–4)

Soil Type (1–5)

📈 Model

Algorithm: XGBoost Classifier

Objective: Multi-class classification (multi:softmax)

Classes: 0–6 (Forest Cover Types)

Hyperparameter Tuning: GridSearchCV

max_depth: 3, 5, 7

learning_rate: 0.01, 0.1, 0.2

n_estimators: 100, 200

subsample: 0.8, 1.0

colsample_bytree: 0.8, 1.0

Evaluation Metrics:

Accuracy

Classification Report

💾 Files

train_model.py – Trains the XGBoost model and saves it as xgb_forest_model.pkl

xgb_forest_model.pkl – Trained model file

app.py – Streamlit app for interactive prediction

README.md – Project documentation

🚀 How to Run

Install required packages:

pip install pandas scikit-learn xgboost streamlit matplotlib


Train the model (optional if using pre-trained .pkl):

python train_model.py


Run the Streamlit app:

streamlit run app.py


Input features in the web app:

Numerical values using sliders/number inputs

Categorical features using checkboxes (converted to 0/1 internally)

Click “Predict” to get:

Predicted Forest Cover Type (0–6)

Prediction probabilities for all classes
