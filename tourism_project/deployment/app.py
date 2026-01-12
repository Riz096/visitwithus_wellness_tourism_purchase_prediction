import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="Rizwan9/visit-with-us-wellness-tourism-purchase-prediction", filename="best_visit-with-us-wellness-tourism-purchase-prediction_model.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Visit with us: Wellness tourism purchase prediction model")
st.write("""
This model predicts whether a customer will purchase a tourism package (specifically the Wellness Tourism Package) before the company contacts them.
""")

# User Input (Tourism Dataset)

age = st.number_input("Age",min_value=18,max_value=100,value=35)
typeof_contact = st.selectbox("Type of Contact",["Self Enquiry", "Company Invited"])
city_tier = st.selectbox("City Tier",[1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)",min_value=1,max_value=60,value=10)
occupation = st.selectbox("Occupation",["Salaried", "Free Lancer", "Small Business"])
gender = st.selectbox("Gender",["Male", "Female"])
num_person_visiting = st.number_input("Number of People Visiting",min_value=1,max_value=10,value=2)
num_followups = st.number_input("Number of Follow-ups",min_value=0,max_value=10,value=2)
product_pitched = st.selectbox("Product Pitched",["Basic", "Standard", "Deluxe"])
preferred_property_star = st.selectbox("Preferred Property Star",[1, 2, 3, 4, 5])
marital_status = st.selectbox("Marital Status",["Single", "Married", "Divorced"])
num_trips = st.number_input("Number of Trips",min_value=0,max_value=50,value=2)
passport = st.selectbox("Passport Available",[0, 1])
pitch_satisfaction = st.slider("Pitch Satisfaction Score",min_value=1,max_value=5,value=3)
own_car = st.selectbox("Own a Car",[0, 1])
num_children = st.number_input("Number of Children Visiting",min_value=0,max_value=5,value=0)
designation = st.selectbox("Designation",["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income",min_value=5000,max_value=200000,value=30000,step=1000)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_person_visiting,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])

# Predict button
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Customer is **LIKELY** to purchase the Wellness Tourism Package.")
    else:
        st.warning("Customer is **UNLIKELY** to purchase the Wellness Tourism Package.")
