import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set the background color using HTML and CSS
def set_white_background():
    background_color = """
        <style>
            body {
                background-color: #ffffff; /* Set the background color to white */
            }
        </style>
    """
    st.markdown(background_color, unsafe_allow_html=True)

# Load the dataset
df = pd.read_csv("dataset.csv")

st.title("Hey How are you....?")
st.write("The Diabetes Checkup Streamlit App is designed to provide users with insights into their potential risk of diabetes based on input parameters. It utilizes a machine learning model, specifically a RandomForestClassifier, trained on a dataset containing information related to diabetes outcomes.")

st.subheader("Training Data")
st.write(df.describe())

st.subheader("Visualization")
st.bar_chart(df)

# Split the data into features (x) and target variable (y)
x = df.drop(["Outcome"], axis=1)
y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Sidebar user input
def user_report():
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
set_white_background()

rf = RandomForestClassifier()
rf.fit(x_train, y_train)

st.subheader("Accuracy: ")
st.write(f"{accuracy_score(y_test, rf.predict(x_test)) * 100:.2f}%")

user_result = rf.predict(user_data.values)
st.subheader("Your Diabetes Report")

# Determine the output based on the prediction
if user_result[0] == 0:
    st.success("You are not diabetic")
else:
    st.error("You have Diabetes")

# Load the second dataset
df2 = pd.read_csv("chancess.csv")

# Display training data summary
st.title("Diabetes Checkup")
st.subheader("Training Data")
st.write(df2.describe())

# Visualization
st.subheader("Visualization")
st.bar_chart(df2)

# Split the data into features (x) and target variable (y)
x2 = df2.drop(["Outcomes"], axis=1)
y2 = df2["Outcomes"]

# Split the data into training and testing sets
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2, random_state=0)

# Sidebar with user input using a form
st.title("Your Details")
with st.form(key='user_form'):
    father_diabetes = st.radio("Your Mother or Father Has Diabetes", ["Yes", "No"])
    mother_diabetes = st.radio("Your Mother Has Diabetes", ["Yes", "No"])
    grandfather_diabetes = st.radio("Your Grandfather Has Diabetes", ["Yes", "No"])
    grandmother_diabetes = st.radio("Your Grandmother Has Diabetes", ["Yes", "No"])
    blood_pressure = st.slider("Blood Pressure", 80, 400)
    skin_thickness = st.text_input("Skin Thickness")
    pregnancies = st.text_input("Pregnancies")
    age = st.text_input("Age")
    submit_button = st.form_submit_button(label='Submit')

# Check if the form is submitted
if submit_button:
    # Prepare input data for prediction
    user_report_data = {
        'father_diabetes': [1 if father_diabetes == "Yes" else 0],
        'mother_diabetes': [1 if mother_diabetes == "Yes" else 0],
        'grandfather_diabetes': [1 if grandfather_diabetes == "Yes" else 0],
        'grandmother_diabetes': [1 if grandmother_diabetes == "Yes" else 0],
        'blood_pressure': [float(blood_pressure)],
        'skin_thickness': [float(skin_thickness)],
        'pregnancies': [float(pregnancies)],
        'age': [float(age)]
    }

    # Create a DataFrame for user input
    user_data2 = pd.DataFrame(user_report_data)

    # Train the model
    rf2 = RandomForestClassifier()
    rf2.fit(x2_train, y2_train)

    # Make probability prediction
    user_result_prob = rf2.predict_proba(user_data2)[0, 1]  # Probability of class 1 (diabetes)
    user_result_percentage = user_result_prob * 100

    # Display the prediction as a percentage
    st.subheader("Your Report:")
    st.write(f"The estimated chance of diabetes is: {user_result_percentage:.2f}%")

# Adding the GitHub icon with hyperlink in the center with a black background
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
