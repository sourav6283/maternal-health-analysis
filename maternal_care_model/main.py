import pickle
import warnings
from pathlib import Path

import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

from codebase.dashboard_graphs import MaternalHealthDashboard

# All paths are built from this file's location, so the app works
# no matter which folder you launch it from.
BASE_DIR = Path(__file__).parent

st.set_page_config(page_title="MedPredict", layout="wide")


# ---------- Load models (cached so they load only once) ----------
@st.cache_resource
def load_pickle(relative_path):
    with open(BASE_DIR / relative_path, "rb") as f:
        return pickle.load(f)


maternal_model = load_pickle("model/finalized_maternal_model.sav")
fetal_model = load_pickle("model/fetal_health_classifier.sav")
scale_X = load_pickle("model/scaler_maternal_model.sav")


# ---------- Helpers ----------
def parse_floats(values):
    """Convert a list of text inputs to floats. Returns None if any is empty/invalid."""
    try:
        return [float(v) for v in values]
    except (ValueError, TypeError):
        return None


def show_result(text, color):
    st.markdown(
        f'<p style="font-weight: bold; font-size: 20px; color: {color};">{text}</p>',
        unsafe_allow_html=True,
    )


# ---------- Sidebar navigation ----------
with st.sidebar:
    st.title("MedPredict")
    st.write("Welcome to MedPredict")
    st.write("Choose an option from the menu below to get started:")

    selected = option_menu(
        "MedPredict",
        ["About us", "Pregnancy Risk Prediction", "Fetal Health Prediction", "Dashboard"],
        icons=["chat-square-text", "hospital", "capsule-pill", "clipboard-data"],
        default_index=0,
    )


# ---------- About us ----------
if selected == "About us":
    st.title("Welcome to MedPredict")
    st.write(
        "At MedPredict, our mission is to revolutionize healthcare by offering innovative solutions "
        "through predictive analysis. Our platform is specifically designed to address the intricate "
        "aspects of maternal and fetal health, providing accurate predictions and proactive risk management."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.header("1. Pregnancy Risk Prediction")
        st.write(
            "Our Pregnancy Risk Prediction feature utilizes advanced algorithms to analyze various "
            "parameters, including age, body sugar levels, blood pressure, and more. By processing this "
            "information, we provide accurate predictions of potential risks during pregnancy."
        )
        st.image(
            str(BASE_DIR / "graphics/pregnancy_risk_image.jpg"),
            caption="Pregnancy Risk Prediction",
            width="stretch",
        )
    with col2:
        st.header("2. Fetal Health Prediction")
        st.write(
            "Fetal Health Prediction is a crucial aspect of our system. We leverage cutting-edge "
            "technology to assess the health status of the fetus. Through a comprehensive analysis of "
            "factors such as ultrasound data, maternal health, and genetic factors, we deliver insights "
            "into the well-being of the unborn child."
        )
        st.image(
            str(BASE_DIR / "graphics/fetal_health_image.jpg"),
            caption="Fetal Health Prediction",
            width="stretch",
        )

    st.header("3. Dashboard")
    st.write(
        "Our Dashboard provides a user-friendly interface for monitoring and managing health data. "
        "It offers a holistic view of predictive analyses, allowing healthcare professionals and users "
        "to make informed decisions. The Dashboard is designed for ease of use and accessibility."
    )

    st.write(
        "Thank you for choosing MedPredict. We are committed to advancing healthcare through technology "
        "and predictive analytics. Feel free to explore our features and take advantage of the insights we provide."
    )


# ---------- Pregnancy Risk Prediction ----------
if selected == "Pregnancy Risk Prediction":
    st.title("Pregnancy Risk Prediction")
    content = (
        "Predicting the risk in pregnancy involves analyzing several parameters, including age, blood "
        "sugar levels, blood pressure, and other relevant factors. By evaluating these parameters, we "
        "can assess potential risks and make informed predictions regarding the pregnancy's health"
    )
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div></br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input("Age of the Person", key="age")
    with col2:
        diastolicBP = st.text_input("diastolicBP in mmHg")
    with col3:
        BS = st.text_input("Blood glucose in mmol/L")
    with col1:
        bodyTemp = st.text_input("Body Temperature in Celsius")
    with col2:
        heartRate = st.text_input("Heart rate in beats per minute")

    with col1:
        if st.button("Predict Pregnancy Risk"):
            values = parse_floats([age, diastolicBP, BS, bodyTemp, heartRate])
            if values is None:
                st.error("Please fill in all fields with valid numbers.")
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    input_scaled = scale_X.transform(np.array([values]))
                    predicted_risk = maternal_model.predict(input_scaled)

                st.subheader("Risk Level:")
                if predicted_risk[0] == 0:
                    show_result("Low Risk", "green")
                elif predicted_risk[0] == 1:
                    show_result("Medium Risk", "orange")
                else:
                    show_result("High Risk", "red")

    with col2:
        if st.button("Clear", key="clear_maternal"):
            st.rerun()


# ---------- Fetal Health Prediction ----------
if selected == "Fetal Health Prediction":
    st.title("Fetal Health Prediction")
    content = (
        "Cardiotocograms (CTGs) are a simple and cost accessible option to assess fetal health, allowing "
        "healthcare professionals to take action in order to prevent child and maternal mortality"
    )
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div></br>", unsafe_allow_html=True)

    # Order matters: it must match the feature order the model was trained on.
    fetal_fields = [
        "Baseline Value",
        "Accelerations",
        "Fetal Movement",
        "Uterine Contractions",
        "Light Decelerations",
        "Severe Decelerations",
        "Prolongued Decelerations",
        "Abnormal Short Term Variability",
        "Mean Value Of Short Term Variability",
        "Percentage Of Time With ALTV",
        "Mean Value Long Term Variability",
        "Histogram Width",
        "Histogram Min",
        "Histogram Max",
        "Histogram Number Of Peaks",
        "Histogram Number Of Zeroes",
        "Histogram Mode",
        "Histogram Mean",
        "Histogram Median",
        "Histogram Variance",
        "Histogram Tendency",
    ]

    cols = st.columns(3)
    fetal_inputs = []
    for i, label in enumerate(fetal_fields):
        with cols[i % 3]:
            fetal_inputs.append(st.text_input(label, key=f"fetal_{i}"))

    st.markdown("</br>", unsafe_allow_html=True)

    btn_col1, btn_col2, _ = st.columns(3)
    with btn_col1:
        if st.button("Predict Fetal Health"):
            values = parse_floats(fetal_inputs)
            if values is None:
                st.error("Please fill in all fields with valid numbers.")
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    predicted = fetal_model.predict([values])

                st.markdown("</br>", unsafe_allow_html=True)
                if predicted[0] == 0:
                    show_result("Result comes to be Normal", "green")
                elif predicted[0] == 1:
                    show_result("Result comes to be Suspect", "orange")
                else:
                    show_result("Result comes to be Pathological", "red")

    with btn_col2:
        if st.button("Clear", key="clear_fetal"):
            st.rerun()


# ---------- Dashboard ----------
if selected == "Dashboard":
    st.header("Dashboard")
    content = (
        "This dashboard shows maternal and child health indicators for every state and union territory "
        "of India, based on survey data. Compare antenatal care, institutional births, anaemia in pregnancy, "
        "infant mortality and more, and see how urban and rural areas differ."
    )
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div></br>", unsafe_allow_html=True)

    dashboard = MaternalHealthDashboard(BASE_DIR / "graphics" / "datafile.csv")
    if dashboard.df is None:
        st.stop()

    dashboard.show_national_summary()

    dashboard.create_bubble_chart()
    with st.expander("Show More"):
        st.markdown(dashboard.get_bubble_chart_data())

    indicator = dashboard.select_indicator()

    dashboard.create_ranking_chart(indicator)
    with st.expander("Show More"):
        st.markdown(dashboard.get_ranking_data())

    dashboard.create_urban_rural_chart(indicator)
    with st.expander("Show More"):
        st.markdown(dashboard.get_urban_rural_data())