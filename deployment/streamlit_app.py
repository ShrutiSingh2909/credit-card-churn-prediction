import streamlit as st
import joblib
import numpy as np

model = joblib.load('xgboost_tuned.pkl')

st.set_page_config(page_title="Credit Card Churn Predictor", page_icon="💳")

st.title("💳 Credit Card Churn Predictor")
st.markdown("Fill in the customer details below to predict if they will churn.")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Personal Info")
    age = st.number_input("Customer Age", 18, 100, 45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    dependents = st.number_input("Dependent Count", 0, 10, 2)
    education = st.selectbox("Education Level", 
                    ["Uneducated", "High School", "Graduate", 
                     "Post-Graduate", "Doctorate", "College", "Unknown"])
    marital = st.selectbox("Marital Status", 
                    ["Married", "Single", "Divorced", "Unknown"])

with col2:
    st.subheader("💰 Financial Info")
    income = st.selectbox("Income Category",
                    ["Less than $40K", "$40K - $60K", "$60K - $80K",
                     "$80K - $120K", "$120K +", "Unknown"])
    card = st.selectbox("Card Category", 
                    ["Blue", "Silver", "Gold", "Platinum"])
    credit_limit = st.number_input("Credit Limit", 0.0, 35000.0, 5000.0)
    revolving_bal = st.number_input("Total Revolving Balance", 0, 3000, 800)
    utilization = st.slider("Avg Utilization Ratio", 0.0, 1.0, 0.3)

with col3:
    st.subheader("📊 Activity Info")
    months_on_book = st.number_input("Months on Book", 0, 60, 36)
    relationship_count = st.number_input("Total Relationship Count", 1, 6, 3)
    inactive_months = st.number_input("Months Inactive (12 mon)", 0, 6, 2)
    contacts_count = st.number_input("Contacts Count (12 mon)", 0, 6, 3)
    trans_amt = st.number_input("Total Transaction Amount", 0, 20000, 2500)
    trans_ct = st.number_input("Total Transaction Count", 0, 150, 40)
    amt_chng = st.number_input("Amount Change Q4-Q1", 0.0, 4.0, 0.7)
    ct_chng = st.number_input("Count Change Q4-Q1", 0.0, 4.0, 0.6)

st.divider()

# Encode inputs
gender_enc    = 1 if gender == "Male" else 0
edu_map       = {"Uneducated":0,"High School":1,"Graduate":2,"Post-Graduate":3,"Doctorate":4,"College":5,"Unknown":6}
marital_map   = {"Married":0,"Single":1,"Divorced":2,"Unknown":3}
income_map    = {"Less than $40K":0,"$40K - $60K":1,"$60K - $80K":2,"$80K - $120K":3,"$120K +":4,"Unknown":5}
card_map      = {"Blue":0,"Silver":1,"Gold":2,"Platinum":3}

if st.button("🔮 Predict Churn", use_container_width=True):
    features = np.array([[
        age, gender_enc, dependents,
        edu_map[education], marital_map[marital], income_map[income],
        card_map[card], months_on_book, relationship_count,
        inactive_months, contacts_count, credit_limit,
        revolving_bal, amt_chng, trans_amt,
        trans_ct, ct_chng, utilization
    ]])

    prediction  = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.divider()

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to CHURN!")
        st.metric("Churn Probability", f"{probability*100:.1f}%")
    else:
        st.success(f"✅ This customer is likely to STAY!")
        st.metric("Churn Probability", f"{probability*100:.1f}%")

    st.progress(float(probability))