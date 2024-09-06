import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def set_white_background():
    background_color = """
        <style>
            body {
                background-color: #ffffff; /* Set the background color to white */
                font-family: Arial, sans-serif;
            }
            .sidebar .sidebar-content {
                background-color: #f0f0f0;
            }
            .css-1d391kg {
                margin: 0 auto;
            }
        </style>
    """
    st.markdown(background_color, unsafe_allow_html=True)

# Load the dataset
df = pd.read_csv("dataset.csv")

# Set the background color
set_white_background()

# Sidebar user input
def user_report():
    st.sidebar.header("User Input Parameters")
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider("Glucose", 0, 200, 120)
    bp = st.sidebar.slider("Blood Pressure", 0, 122, 70)
    skinthickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0, 846, 79)
    bmi = st.sidebar.slider("BMI", 0, 67, 20)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.4, 0.47)
    age = st.sidebar.slider("Age", 21, 88, 33)

    user_report = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "Blood Pressure": bp,
        'Skin Thickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'Diabetes Pedigree Function': dpf,
        'Age': age
    }
    report_data = pd.DataFrame(user_report, index=[0])
    return report_data

user_data = user_report()
# Prepare the model
x = df.drop(["Outcome"], axis=1)
y = df["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Display accuracy
st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy_score(y_test, rf.predict(x_test)) * 100:.2f}%")

# Display user result
user_result = rf.predict(user_data.values)
st.subheader("Your Diabetes Report")
if user_result[0] == 0:
    st.success("You are not diabetic")
else:
    st.error("You have Diabetes")

# Title and description
st.title("Diabetes Checkup Streamlit App")
st.write("The Diabetes Checkup Streamlit App is designed to provide users with insights into their potential risk of diabetes based on input parameters. It utilizes a machine learning model, specifically a RandomForestClassifier, trained on a dataset containing information related to diabetes outcomes.")

# Display training data
st.subheader("Training Data")
st.write(df.describe())

# Display data visualization
st.subheader("Visualization")
st.bar_chart(df)



import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
df1 = pd.read_csv("chancess.csv")

# Streamlit app
st.title("Diabetes Checkup")
st.subheader("Training Data")
st.write(df1.describe())



# Visualization
st.subheader("Visualization")
st.bar_chart(df1)

# Split the data into features (X) and target variable (y)
X = df1.drop(["Outcomes"], axis=1)
y = df1["Outcomes"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model (using RandomForestRegressor for continuous outcomes)
rf = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)
rf.fit(X_train_scaled, y_train)

# Check test set performance
y_pred_test = rf.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

st.write(f"Test Set Mean Squared Error: {mse:.2f}")
st.write(f"Test Set R-squared Score: {r2:.2f}")

# Feature importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
st.write("Feature Importance:")
st.bar_chart(feature_importance.set_index('feature'))

# User input form
st.title("Predict the Chance of Diabetes Based on Previous Generation Details")
with st.form(key='user_form'):
    father_diabetes = st.radio("Your Father Has Diabetes", ["Yes", "No"])
    mother_diabetes = st.radio("Your Mother Has Diabetes", ["Yes", "No"])
    grandfather_diabetes = st.radio("Your Grandfather Has Diabetes", ["Yes", "No"])
    grandmother_diabetes = st.radio("Your Grandmother Has Diabetes", ["Yes", "No"])
    blood_pressure = st.slider("Blood Pressure", 60, 200, 150)
    skin_thickness = st.slider("Skin Thickness (mm)", 1, 50, 28)
    pregnancies = st.slider("Number of Pregnancies", 0, 20, 8)
    age = st.slider("Age", 10, 100, 60)
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    user_data = pd.DataFrame({
        'father_diabetes': [1 if father_diabetes == "Yes" else 0],
        'mother_diabetes': [1 if mother_diabetes == "Yes" else 0],
        'grandfather_diabetes': [1 if grandfather_diabetes == "Yes" else 0],
        'grandmother_diabetes': [1 if grandmother_diabetes == "Yes" else 0],
        'blood_pressure': [blood_pressure],
        'skin_thickness': [skin_thickness],
        'pregnancies': [pregnancies],
        'age': [age]
    })
    
    user_data_scaled = scaler.transform(user_data)
    user_result = rf.predict(user_data_scaled)[0]
    
    # Ensure the prediction is between 0 and 100
    user_result = np.clip(user_result, 0, 100)

    st.subheader("Your Report:")
    if user_result > 70:
        st.markdown(f"<p style='color:red;'>The estimated chance of diabetes is: {user_result:.2f}%</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='color:green;'>The estimated chance of diabetes is: {user_result:.2f}%</p>", unsafe_allow_html=True)

# Adding the GitHub icon with hyperlink
github_url = "https://github.com/chandankumarm55"
st.markdown(
    f"""
    <div style="text-align: center;margin-top:10px">
        <a href="{github_url}" target="_blank">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="30" height="30" alt="GitHub" style="background-color: black; padding: 5px; border-radius: 50%;">
        </a>
    </div>
    """, 
    unsafe_allow_html=True
)