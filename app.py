import streamlit as st
import pandas as pd
import pickle


with open('xgb_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Forest Cover Type Prediction 🌲")
st.write("Enter the forest features below:")


Elevation = st.number_input("Elevation", min_value=0, max_value=5000, value=2785)
Aspect = st.number_input("Aspect", min_value=0, max_value=360, value=155)
Slope = st.number_input("Slope", min_value=0, max_value=90, value=18)
Horizontal_Distance_To_Hydrology = st.number_input("Horizontal Distance To Hydrology", value=242)
Vertical_Distance_To_Hydrology = st.number_input("Vertical Distance To Hydrology", value=118)
Horizontal_Distance_To_Roadways = st.number_input("Horizontal Distance To Roadways", value=3090)
Hillshade_9am = st.number_input("Hillshade 9AM", value=238)
Hillshade_Noon = st.number_input("Hillshade Noon", value=238)
Hillshade_3pm = st.number_input("Hillshade 3PM", value=122)
Horizontal_Distance_To_Fire_Points = st.number_input("Horizontal Distance To Fire Points", value=6211)
Wilderness_Area1 = st.number_input("Wilderness Area 1") # True → 1, False → 0
Soil_Type3 = st.number_input("Soil Type 3",value=1)    

feature_names = model.feature_names_in_
sample_data = {col: [0] for col in feature_names}

sample_data.update({
    'Elevation': [Elevation],
    'Aspect': [Aspect],
    'Slope': [Slope],
    'Horizontal_Distance_To_Hydrology': [Horizontal_Distance_To_Hydrology],
    'Vertical_Distance_To_Hydrology': [Vertical_Distance_To_Hydrology],
    'Horizontal_Distance_To_Roadways': [Horizontal_Distance_To_Roadways],
    'Hillshade_9am': [Hillshade_9am],
    'Hillshade_Noon': [Hillshade_Noon],
    'Hillshade_3pm': [Hillshade_3pm],
    'Horizontal_Distance_To_Fire_Points': [Horizontal_Distance_To_Fire_Points],
    'Wilderness_Area1': [int(Wilderness_Area1)],
    'Soil_Type3': [int(Soil_Type3)],
   
})

sample_df = pd.DataFrame(sample_data)[feature_names]

# ---- Prediction ----
if st.button("Predict"):
    prediction = model.predict(sample_df)[0]
    proba = model.predict_proba(sample_df)[0]

    st.success(f"Predicted Forest Cover Type (0-6): {prediction}")
    st.write("Prediction Probabilities per class:")
    st.write(proba)
