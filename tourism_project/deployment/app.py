
import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

# Title of the Streamlit app
st.title("Wellness Tourism Package Purchase Prediction")

# Hugging Face model details
repo_id = "SantoshS23/tourism_package_prediction_model"
filename = "best_tourism_package_prediction_model_v1.joblib"

# Get Hugging Face token from environment variables
hf_token = os.getenv("HF_TOKEN")

@st.cache_resource
def load_model(repo_id, filename, token):
    """Caches the model loading process."""
    if not token:
        st.error("HF_TOKEN environment variable not set. Cannot load model.")
        st.info("Please set the HF_TOKEN environment variable with your Hugging Face access token.")
        st.stop()
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from Hugging Face Hub: {e}")
        st.info("Please ensure you have set the HF_TOKEN environment variable with a valid token.")
        st.stop()

model = load_model(repo_id, filename, hf_token)

if model:
    st.success("Model loaded successfully from Hugging Face Hub!")
else:
    st.error("Model could not be loaded. Please check your HF_TOKEN and repo details.")
    st.stop()


# Input features for the model
st.header("Customer Information")

# Define options for select boxes (must match alphabetical order for LabelEncoding simulation)
typeofcontact_options = ["Company Invited", "Self Inquiry"]
occupation_options = ["Freelancer", "Government Sector", "Large Business", "Private Sector", "Salaried", "Small Business", "Unemployed"]
gender_options = ["Female", "Male"]
productpitched_options = ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"]
maritalstatus_options = ["Divorced", "Married", "Single"]
designation_options = ["Director", "Entry Level", "Executive", "Manager", "Senior Manager", "VP"]
passport_options = ["No", "Yes"]
owncar_options = ["No", "Yes"]

# Collect inputs
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, key='age_input')
        typeofcontact = st.selectbox("Type of Contact", typeofcontact_options, key='typeofcontact_input')
        citytier = st.selectbox("City Tier", [1, 2, 3], key='citytier_input')
        occupation = st.selectbox("Occupation", occupation_options, key='occupation_input')
        gender = st.selectbox("Gender", gender_options, key='gender_input')
        numberofpersonvisiting = st.number_input("Number of People Visiting", min_value=1, max_value=10, value=1, key='num_person_input')

    with col2:
        preferredpropertystar = st.selectbox("Preferred Property Star", [3, 4, 5], key='prop_star_input')
        maritalstatus = st.selectbox("Marital Status", maritalstatus_options, key='marital_status_input')
        numberoftrips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=5, key='num_trips_input')
        passport = st.selectbox("Has Passport?", passport_options, key='passport_input')
        owncar = st.selectbox("Owns a Car?", owncar_options, key='owncar_input')
        numberofchildrenvisiting = st.number_input("Number of Children Visiting (below 5)", min_value=0, max_value=5, value=0, key='num_children_input')

    with col3:
        designation = st.selectbox("Designation", designation_options, key='designation_input')
        monthlyincome = st.number_input("Monthly Income", min_value=0, value=50000, step=1000, key='monthly_income_input')
        pitchsatisfactionscore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5], key='pitch_score_input')
        productpitched = st.selectbox("Product Pitched", productpitched_options, key='product_pitched_input')
        numberoffollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2, key='num_followups_input')
        durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=10, key='pitch_duration_input')

    submitted = st.form_submit_button("Predict Purchase")

    if submitted:
        # Create a dictionary from inputs
        input_data_dict = {
            'Age': age,
            'TypeofContact': typeofcontact,
            'CityTier': citytier,
            'DurationOfPitch': durationofpitch,
            'NumberOfPersonVisiting': numberofpersonvisiting,
            'NumberOfFollowups': numberoffollowups,
            'PreferredPropertyStar': preferredpropertystar,
            'NumberOfTrips': numberoftrips,
            'Passport': passport,
            'PitchSatisfactionScore': pitchsatisfactionscore,
            'OwnCar': owncar,
            'NumberOfChildrenVisiting': numberofchildrenvisiting,
            'MonthlyIncome': monthlyincome,
            'Occupation': occupation,
            'Gender': gender,
            'ProductPitched': productpitched,
            'MaritalStatus': maritalstatus,
            'Designation': designation
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data_dict])

        # Convert 'Yes'/'No' for Passport and OwnCar to 0/1 integers
        input_df['Passport'] = input_df['Passport'].map({"No": 0, "Yes": 1}).astype(int)
        input_df['OwnCar'] = input_df['OwnCar'].map({"No": 0, "Yes": 1}).astype(int)

        # Manually apply LabelEncoding for other categorical features (based on alphabetical order of defined options)
        input_df['TypeofContact'] = input_df['TypeofContact'].map({val: i for i, val in enumerate(sorted(typeofcontact_options))})
        input_df['Occupation'] = input_df['Occupation'].map({val: i for i, val in enumerate(sorted(occupation_options))})
        input_df['Gender'] = input_df['Gender'].map({val: i for i, val in enumerate(sorted(gender_options))})
        input_df['ProductPitched'] = input_df['ProductPitched'].map({val: i for i, val in enumerate(sorted(productpitched_options))})
        input_df['MaritalStatus'] = input_df['MaritalStatus'].map({val: i for i, val in enumerate(sorted(maritalstatus_options))})
        input_df['Designation'] = input_df['Designation'].map({val: i for i, val in enumerate(sorted(designation_options))})

        # Ensure column order matches training data features, excluding 'Unnamed: 0'
        expected_columns = [
            'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
            'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'Passport',
            'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome',
            'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation'
        ]
        input_df = input_df[expected_columns]


        # Make prediction
        prediction = model.predict(input_df)
        probability_of_purchase = prediction[0]

        st.subheader("Prediction Result")
        st.write(f"Predicted likelihood of purchase: **{probability_of_purchase:.2f}**")

        if probability_of_purchase > 0.5:
            st.success("This customer is likely to purchase the Wellness Tourism Package!")
        else:
            st.info("This customer is less likely to purchase the Wellness Tourism Package.")
