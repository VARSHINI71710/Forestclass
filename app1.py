import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

with open('rf_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


st.title("Forest Cover Type Prediction 🌲")
st.write("Enter the forest features below:")


df=pd.read_csv("c:/Users/varsh/Downloads/sampled_data.csv")
X = df.drop("Cover_Type", axis=1)
y = df["Cover_Type"].astype(int)
y = y - y.min()   

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestClassifier(
    n_estimators=200,     
    max_depth=None,       
    random_state=42,
    class_weight="balanced"   
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

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
Wilderness_Area1 = st.number_input("Wilderness Area 1") 
Wilderness_Area2 = st.number_input("Wilderness Area 2") 
Wilderness_Area3 = st.number_input("Wilderness Area 3") 
Wilderness_Area4 = st.number_input("Wilderness Area 4") 
Soil_Type4 = st.number_input("Soil Type 4",value=1)   
Soil_Type10 = st.number_input("Soil Type 10",value=1)   
Soil_Type11 = st.number_input("Soil Type 11",value=1)   
Soil_Type12 = st.number_input("Soil Type 12",value=1)   
Soil_Type13 = st.number_input("Soil Type 13",value=1)   
Soil_Type20 = st.number_input("Soil Type 20",value=1)   
Soil_Type22 = st.number_input("Soil Type 22",value=1)   
Soil_Type23 = st.number_input("Soil Type 23",value=1)   
Soil_Type24 = st.number_input("Soil Type 24",value=1)   
Soil_Type29 = st.number_input("Soil Type 29",value=1)  
Soil_Type30 = st.number_input("Soil Type 30",value=1)   
Soil_Type31 = st.number_input("Soil Type 31",value=1)   
Soil_Type32 = st.number_input("Soil Type 32",value=1)   
Soil_Type33 = st.number_input("Soil Type 33",value=1)   
Soil_Type38 = st.number_input("Soil Type 38",value=1)   
Soil_Type39 = st.number_input("Soil Type 39",value=1)  


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
    'Wilderness_Area2': [int(Wilderness_Area2)],
    'Wilderness_Area3': [int(Wilderness_Area3)],
    'Wilderness_Area4': [int(Wilderness_Area4)],
    'Soil_Type4': [Soil_Type4],
    'Soil_Type10': [Soil_Type10],
    'Soil_Type11': [Soil_Type11],
    'Soil_Type12': [Soil_Type12],
    'Soil_Type13': [Soil_Type13],
    'Soil_Type20': [Soil_Type20],
    'Soil_Type22': [Soil_Type22],
    'Soil_Type23': [Soil_Type23],
    'Soil_Type24': [Soil_Type24],
    'Soil_Type29': [Soil_Type29],
    'Soil_Type30': [Soil_Type30],
    'Soil_Type31': [Soil_Type31],
    'Soil_Type32': [Soil_Type32],
    'Soil_Type33': [Soil_Type33],
    'Soil_Type38': [Soil_Type38],
    'Soil_Type39': [Soil_Type39]
})

sample_df = pd.DataFrame(sample_data)[feature_names]

if st.button("Predict"):
    model=rf.fit(X_train, y_train)

    prediction = model.predict(sample_df)[0]
    proba = model.predict_proba(sample_df)[0]

    st.success(f"Predicted Forest Cover Type (0-6): {prediction}")
    st.write("Prediction Probabilities per class:")
    st.write(proba)

  



