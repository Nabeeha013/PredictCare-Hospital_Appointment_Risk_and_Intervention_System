import streamlit as st
import pandas as pd
import joblib
import base64
from pathlib import Path
from datetime import date, datetime, time
from io import BytesIO
import plotly.express as px
import altair as alt
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,auc,accuracy_score,roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from prediction_functions import prepare_uploaded_data,get_intervention,get_risk_level
from charts import age_group_chart, donut_show_noshow, sms_chart, waiting_days_chart,waiting_days_no_show_chart,sms_no_show_chart,previous_noshows_chart,appointment_day_chart
from theme import COLORS, PAGES, inject_css, kpi_card, risk_level_html

st.set_page_config(
    page_title="PredictCare",
    page_icon="🏥",
    layout="wide")

with open("PredictCare_header.png", "rb") as f:
    encoded = base64.b64encode(f.read()).decode()

st.markdown("""
<style>
.predictcare-header {
    margin-top: -140px;
    margin-bottom: -90px;
    width:100%;
}

.predictcare-header img {
    width: 100%;
    max-width: 3300px;
    height: 300px;
    object-fit: contain;
    object-position: left;
    display: block;
    margin: 0 auto;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="predictcare-header">
        <img src="data:image/png;base64,{encoded}">
    </div>
    """,
    unsafe_allow_html=True
)
#Model loading
@st.cache_resource
def load_model():
    return joblib.load("predictcare_model.pkl")
model_data = load_model()
model = model_data["model"]
THRESHOLD = model_data["threshold"]

if "prediction_results" not in st.session_state:
    st.session_state["prediction_results"] = pd.DataFrame()

if "prediction_source" not in st.session_state:
    st.session_state["prediction_source"] = None

#SIDEBAR

with st.sidebar:
     st.image("PredictCare_logo.png",width=300)
     page = st.radio(
          "Navigation",
          ["🏠Dashboard","👤Single Patient","📊Batch Upload","📝Quick Prediction","📈Model Insights","ℹ️About PredictCare"]
     )
     st.divider()
     with st.container(border=True):
        st.markdown("### 💙 Predict earlier.")
        st.markdown("**Care better.**")
        st.caption("— PredictCare")

# =================
# LIVE DASHBOARD
# =================
if page == "🏠Dashboard":
        
    st.header("🏠 Dashboard")
    st.write(
         "Live overview of appointment risk, predictions, "
         "and attendance patterns.")
    
    prediction_df = st.session_state.get(
        "prediction_results",
        pd.DataFrame()
    )

    prediction_source = st.session_state.get(
        "prediction_source",
        None
    )

    if prediction_df.empty:

        st.info(
            "📊 No prediction results available yet. "
            "Generate predictions from Batch Upload, Single Patient, "
            "or Quick Prediction to activate the live dashboard."
        )

    else:
        st.caption(
            f"Live results from: **{prediction_source}**"
        )

        # KPI CALCULATIONS

        total_patients = len(prediction_df)

        predicted_no_shows = (
            prediction_df["Prediction"]
            .eq("Predicted No Show")
            .sum()
        )

        predicted_shows = (
            prediction_df["Prediction"]
            .eq("Predicted Show")
            .sum()
        )

        high_risk_patients = (
            prediction_df["Risk Level"]
            .isin(["🟠High", "🔴Very High"])
            .sum()
        )

        average_risk = prediction_df["Risk %"].mean()

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric(
            "Total Appointments",
            f"{total_patients:,}"
        )

        col2.metric(
            "Predicted No-Shows",
            f"{predicted_no_shows:,}"
        )

        col3.metric(
            "Predicted Shows",
            f"{predicted_shows:,}"
        )

        col4.metric(
            "High / Very High Risk",
            f"{high_risk_patients:,}"
        )

        col5.metric(
            "Average Risk",
            f"{average_risk:.1f}%"
        )

        st.divider()
        
        st.subheader("🎯 Risk Distribution")

        # Clean risk-level text for reliable grouping
        risk_df = prediction_df.copy()

        risk_df["Risk Level Clean"] = (
            risk_df["Risk Level"]
            .astype(str)
            .str.replace("🟢", "", regex=False)
            .str.replace("🟡", "", regex=False)
            .str.replace("🟠", "", regex=False)
            .str.replace("🔴", "", regex=False)
            .str.strip()
        )

        # Count actual risk levels
        risk_counts = (
            risk_df["Risk Level Clean"]
            .value_counts()
            .reindex(
                ["Low", "Medium", "High", "Very High"],
                fill_value=0
            )
            .reset_index()
        )
        risk_counts.columns = [
            "Risk Level",
            "Patients"
        ]
        # Put emojis back only for display
        risk_counts["Display Risk Level"] = (
            risk_counts["Risk Level"].map({
                "Low": "🟢 Low",
                "Medium": "🟡 Medium",
                "High": "🟠 High",
                "Very High": "🔴 Very High"
            })
        )

        risk_colors = {
            "Low": "#22C55E",
            "Medium": "#FACC15",
            "High": "#F97316",
            "Very High": "#EF4444"
        }

        risk_chart = (
            alt.Chart(risk_counts)
            .mark_bar(
                cornerRadiusTopLeft=6,
                cornerRadiusTopRight=6
            )
            .encode(
                x=alt.X(
                    "Display Risk Level:N",
                    sort=[
                        "🟢 Low",
                        "🟡 Medium",
                        "🟠 High",
                        "🔴 Very High"
                    ],
                    title="Risk Level"
                ),

                y=alt.Y(
                    "Patients:Q",
                    title="Number of Patients"
                ),

                color=alt.Color(
                    "Risk Level:N",
                    scale=alt.Scale(
                        domain=[
                            "Low",
                            "Medium",
                            "High",
                            "Very High"
                        ],
                        range=[
                            risk_colors["Low"],
                            risk_colors["Medium"],
                            risk_colors["High"],
                            risk_colors["Very High"]
                        ]
                    ),
                    legend=None
                ),

                tooltip=[
                    alt.Tooltip(
                        "Display Risk Level:N",
                        title="Risk Level"
                    ),
                    alt.Tooltip(
                        "Patients:Q",
                        title="Patients"
                    )
                ]
            )
        )

        risk_labels = (
            alt.Chart(risk_counts)
            .mark_text(
                dy=-10,
                fontSize=14
            )
            .encode(
                x=alt.X(
                    "Display Risk Level:N",
                    sort=[
                        "🟢 Low",
                        "🟡 Medium",
                        "🟠 High",
                        "🔴 Very High"
                    ]
                ),
                y=alt.Y("Patients:Q"),
                text=alt.Text("Patients:Q")
            )
        )

        st.altair_chart(
            risk_chart + risk_labels,
            use_container_width=True
        )
        st.divider()

        st.subheader("🔎 Attendance Insights")

        st.write(
            "These charts show how PredictCare's predicted no-show "
            "risk varies across important patient and appointment factors "
            "in the current prediction dataset."
        )

        st.caption(
            "⚠️ These patterns describe predicted risk and should not "
            "be interpreted as proof that a factor directly causes no-shows."
        )

        
        # ============================
        # CHARTS
        # ============================
        col01,col02 = st.columns(2)
        with col01:    
            if "WaitingDays" in prediction_df.columns:
                st.markdown("### ⏳ Waiting Days vs Average Predicted Risk")
                waiting_df = prediction_df.copy()
                waiting_df["WaitingDays"] = pd.to_numeric(
                    waiting_df["WaitingDays"],
                    errors="coerce"
                )
                waiting_df = waiting_df.dropna(
                    subset=["WaitingDays", "Risk %"]
                )
                waiting_df["Waiting Group"] = pd.cut(
                    waiting_df["WaitingDays"],
                    bins=[
                        -1,
                        0,
                        7,
                        14,
                        30,
                        float("inf")
                    ],
                    labels=[
                        "Same Day",
                        "1–7 Days",
                        "8–14 Days",
                        "15–30 Days",
                        "30+ Days"
                    ]
                )
                waiting_analysis = (
                    waiting_df
                    .groupby(
                        "Waiting Group",
                        observed=False
                    )["Risk %"]
                    .mean()
                    .reset_index()
                )
                waiting_analysis.columns = [
                    "Waiting Group",
                    "Average Risk"
                ]
                waiting_chart = (
                    alt.Chart(waiting_analysis)
                    .mark_line(
                        point=True,
                        strokeWidth=3
                    )
                    .encode(
                        x=alt.X(
                            "Waiting Group:N",
                            sort=[
                                "Same Day",
                                "1–7 Days",
                                "8–14 Days",
                                "15–30 Days",
                                "30+ Days"
                            ],
                            title="Waiting Time"
                        ),
                        y=alt.Y(
                            "Average Risk:Q",
                            title="Average Predicted Risk (%)"
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "Waiting Group:N",
                                title="Waiting Time"
                            ),
                            alt.Tooltip(
                                "Average Risk:Q",
                                title="Average Risk",
                                format=".1f"
                            )
                        ]
                    )
                )
                st.altair_chart(
                    waiting_chart,
                    use_container_width=True
                )

        # SMS RECEIVED 

        with col02:
            if "SMS_received" in prediction_df.columns:

                st.markdown("### 📱 SMS Received vs Average Predicted Risk")

                sms_df = prediction_df.copy()

                sms_df["SMS Status"] = (
                    sms_df["SMS_received"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map({
                        "1": "Received",
                        "0": "Not Received",
                        "yes": "Received",
                        "no": "Not Received"
                    })
                )

                sms_df = sms_df.dropna(
                    subset=["SMS Status", "Risk %"]
                )

                sms_analysis = (
                    sms_df
                    .groupby("SMS Status")["Risk %"]
                    .mean()
                    .reindex([
                        "Received",
                        "Not Received"
                    ])
                    .reset_index()
                )

                sms_analysis.columns = [
                    "SMS Status",
                    "Average Risk"
                ]

                sms_chart = (
                    alt.Chart(sms_analysis)
                    .mark_bar(
                        cornerRadiusTopLeft=6,
                        cornerRadiusTopRight=6
                    )
                    .encode(
                        x=alt.X(
                            "SMS Status:N",
                            title="SMS Status"
                        ),

                        y=alt.Y(
                            "Average Risk:Q",
                            title="Average Predicted Risk (%)"
                        ),

                        color=alt.Color(
                            "SMS Status:N",
                            scale=alt.Scale(
                                domain=[
                                    "Received",
                                    "Not Received"
                                ],
                                range=[
                                    "#0EA5E9",
                                    "#64748B"
                                ]
                            ),
                            legend=None
                        ),

                        tooltip=[
                            alt.Tooltip(
                                "SMS Status:N",
                                title="SMS"
                            ),
                            alt.Tooltip(
                                "Average Risk:Q",
                                title="Average Risk",
                                format=".1f"
                            )
                        ]
                    )
                )

                st.altair_chart(
                    sms_chart,
                    use_container_width=True
                )

        # AGE GROUP
        
        col03,col04 = st.columns(2)
        with col03:
            if "Age" in prediction_df.columns:

                st.markdown("### 👥 Age Group vs Average Predicted Risk")

                age_df = prediction_df.copy()

                age_df["Age"] = pd.to_numeric(
                    age_df["Age"],
                    errors="coerce"
                )
                age_df = age_df.dropna(
                    subset=["Age", "Risk %"]
                )
                age_df["Age Group"] = pd.cut(
                    age_df["Age"],
                    bins=[
                        -1,
                        17,
                        30,
                        45,
                        60,
                        float("inf")
                    ],
                    labels=[
                        "0–17",
                        "18–30",
                        "31–45",
                        "46–60",
                        "61+"
                    ]
                )
                age_analysis = (
                    age_df
                    .groupby(
                        "Age Group",
                        observed=False
                    )["Risk %"]
                    .mean()
                    .reset_index()
                )
                age_analysis.columns = [
                    "Age Group",
                    "Average Risk"
                ]
                age_chart = (
                    alt.Chart(age_analysis)
                    .mark_bar(
                        cornerRadiusTopLeft=6,
                        cornerRadiusTopRight=6
                    )
                    .encode(
                        x=alt.X(
                            "Age Group:N",
                            sort=[
                                "0–17",
                                "18–30",
                                "31–45",
                                "46–60",
                                "61+"
                            ],
                            title="Age Group"
                        ),
                        y=alt.Y(
                            "Average Risk:Q",
                            title="Average Predicted Risk (%)"
                        ),
                        color=alt.value("#2563EB"),

                        tooltip=[
                            alt.Tooltip(
                                "Age Group:N",
                                title="Age Group"
                            ),
                            alt.Tooltip(
                                "Average Risk:Q",
                                title="Average Risk",
                                format=".1f"
                            )
                        ]
                    )
                )
                st.altair_chart(
                    age_chart,
                    use_container_width=True
                )

        # PREVIOUS NO-SHOWS
        
        with col04:
            if "PreviousNoShows" in prediction_df.columns:

                st.markdown(
                    "### 🔁 Previous No-Shows vs Average Predicted Risk"
                )

                previous_df = prediction_df.copy()

                previous_df["PreviousNoShows"] = pd.to_numeric(
                    previous_df["PreviousNoShows"],
                    errors="coerce"
                )

                previous_df = previous_df.dropna(
                    subset=["PreviousNoShows", "Risk %"]
                )

                previous_df["Previous No-Show Group"] = pd.cut(
                    previous_df["PreviousNoShows"],
                    bins=[
                        -1,
                        0,
                        1,
                        2,
                        float("inf")
                    ],
                    labels=[
                        "0",
                        "1",
                        "2",
                        "3+"
                    ]
                )
                previous_analysis = (
                    previous_df
                    .groupby(
                        "Previous No-Show Group",
                        observed=False
                    )["Risk %"]
                    .mean()
                    .reset_index()
                )
                previous_analysis.columns = [
                    "Previous No-Show Group",
                    "Average Risk"
                ]
                previous_chart = (
                    alt.Chart(previous_analysis)
                    .mark_line(
                        point=True,
                        strokeWidth=3
                    )
                    .encode(
                        x=alt.X(
                            "Previous No-Show Group:N",
                            sort=[
                                "0",
                                "1",
                                "2",
                                "3+"
                            ],
                            title="Previous No-Shows"
                        ),

                        y=alt.Y(
                            "Average Risk:Q",
                            title="Average Predicted Risk (%)"
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "Previous No-Show Group:N",
                                title="Previous No-Shows"
                            ),
                            alt.Tooltip(
                                "Average Risk:Q",
                                title="Average Risk",
                                format=".1f"
                            )
                        ]
                    )
                )
                st.altair_chart(
                    previous_chart,
                    use_container_width=True
                )
        # NEIGHBOURHOOD 

        if "Neighbourhood" in prediction_df.columns:
            st.markdown(
                "### 📍 Highest-Risk Neighbourhoods"
            )
            neighbourhood_df = prediction_df.copy()
            neighbourhood_df["Neighbourhood"] = (
                neighbourhood_df["Neighbourhood"]
                .astype(str)
                .str.strip()
            )
            # Remove empty / invalid neighbourhoods
            neighbourhood_df = neighbourhood_df[
                neighbourhood_df["Neighbourhood"].notna()
                & (neighbourhood_df["Neighbourhood"] != "")
                & (neighbourhood_df["Neighbourhood"] != "nan")
            ]
            neighbourhood_analysis = (
                neighbourhood_df
                .groupby("Neighbourhood")["Risk %"]
                .agg(
                    Average_Risk="mean",
                    Patients="count"
                )
                .reset_index()
            )
            neighbourhood_analysis = (
                neighbourhood_analysis
                .sort_values(
                    "Average_Risk",
                    ascending=False
                )
                .head(10)
            )
            if neighbourhood_analysis.empty:

                st.info(
                    "📍 No neighbourhood data is available "
                    "for the current predictions."
                )

            else:
                neighbourhood_chart = (
                    alt.Chart(neighbourhood_analysis)
                    .mark_bar(
                        cornerRadiusTopRight=6,
                        cornerRadiusBottomRight=6
                    )
                    .encode(
                        x=alt.X(
                            "Average_Risk:Q",
                            title="Average Predicted Risk (%)"
                        ),
                        y=alt.Y(
                            "Neighbourhood:N",
                            sort="-x",
                            title="Neighbourhood"
                        ),
                        color=alt.value("#7C3AED"),
                        tooltip=[
                            alt.Tooltip(
                                "Neighbourhood:N",
                                title="Neighbourhood"
                            ),
                            alt.Tooltip(
                                "Average_Risk:Q",
                                title="Average Risk",
                                format=".1f"
                            ),
                            alt.Tooltip(
                                "Patients:Q",
                                title="Appointments"
                            )
                        ]
                    )
                )
                st.altair_chart(
                    neighbourhood_chart,
                    use_container_width=True
                )

        # WHAT PREDICTCARE IS SEEING

        st.divider()

        st.subheader("💡 What PredictCare is Seeing")

        st.caption(
            "Automatically generated observations from the current prediction dataset."
        )
        insights = []

        # WAITING DAYS INSIGHT
    
        if "WaitingDays" in prediction_df.columns:

            waiting_df = prediction_df.copy()
            waiting_df["WaitingDays"] = pd.to_numeric(
                waiting_df["WaitingDays"],
                errors="coerce"
            )
            waiting_df["Waiting Group"] = pd.cut(
                waiting_df["WaitingDays"],
                bins=[
                    -1,
                    0,
                    7,
                    14,
                    30,
                    float("inf")
                ],
                labels=[
                    "Same Day",
                    "1–7 Days",
                    "8–14 Days",
                    "15–30 Days",
                    "30+ Days"
                ]
            )
            waiting_insight = (
                waiting_df
                .groupby(
                    "Waiting Group",
                    observed=False
                )["Risk %"]
                .mean()
                .dropna()
            )

            if not waiting_insight.empty:

                highest_waiting_group = waiting_insight.idxmax()
                highest_waiting_risk = waiting_insight.max()

                insights.append(
                    f"⏳ **Waiting time:** "
                    f"**{highest_waiting_group}** has the highest "
                    f"average predicted risk at **{highest_waiting_risk:.1f}%**."
                )

        # SMS INSIGHT

        if "SMS_received" in prediction_df.columns:

            sms_df = prediction_df.copy()

            sms_df["SMS Status"] = (
                sms_df["SMS_received"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({
                    "1": "Received",
                    "0": "Not Received",
                    "yes": "Received",
                    "no": "Not Received"
                })
            )
            sms_insight = (
                sms_df
                .groupby("SMS Status")["Risk %"]
                .mean()
                .dropna()
            )
            if len(sms_insight) >= 2:

                highest_sms_group = sms_insight.idxmax()
                lowest_sms_group = sms_insight.idxmin()

                difference = (
                    sms_insight.max()
                    - sms_insight.min()
                )
                insights.append(
                    f"📱 **SMS pattern:** "
                    f"Patients marked **{highest_sms_group}** have "
                    f"the higher average predicted risk. "
                    f"The difference between the groups is "
                    f"**{difference:.1f} percentage points**."
                )

        # PREVIOUS NO-SHOW INSIGHT

        if "PreviousNoShows" in prediction_df.columns:

            previous_df = prediction_df.copy()

            previous_df["PreviousNoShows"] = pd.to_numeric(
                previous_df["PreviousNoShows"],
                errors="coerce"
            )

            previous_df["Previous No-Show Group"] = pd.cut(
                previous_df["PreviousNoShows"],
                bins=[
                    -1,
                    0,
                    1,
                    2,
                    float("inf")
                ],
                labels=[
                    "0",
                    "1",
                    "2",
                    "3+"
                ]
            )

            previous_insight = (
                previous_df
                .groupby(
                    "Previous No-Show Group",
                    observed=False
                )["Risk %"]
                .mean()
                .dropna()
            )

            if not previous_insight.empty:

                highest_previous_group = (
                    previous_insight.idxmax()
                )

                highest_previous_risk = (
                    previous_insight.max()
                )

                insights.append(
                    f"🔁 **Previous attendance:** "
                    f"Patients with **{highest_previous_group} previous "
                    f"no-show(s)** have the highest average predicted "
                    f"risk at **{highest_previous_risk:.1f}%**."
                )

        # AGE INSIGHT

        if "Age" in prediction_df.columns:

            age_df = prediction_df.copy()

            age_df["Age"] = pd.to_numeric(
                age_df["Age"],
                errors="coerce"
            )

            age_df["Age Group"] = pd.cut(
                age_df["Age"],
                bins=[
                    -1,
                    17,
                    30,
                    45,
                    60,
                    float("inf")
                ],
                labels=[
                    "0–17",
                    "18–30",
                    "31–45",
                    "46–60",
                    "61+"
                ]
            )

            age_insight = (
                age_df
                .groupby(
                    "Age Group",
                    observed=False
                )["Risk %"]
                .mean()
                .dropna()
            )

            if not age_insight.empty:

                highest_age_group = age_insight.idxmax()
                highest_age_risk = age_insight.max()

                insights.append(
                    f"👥 **Age pattern:** "
                    f"The **{highest_age_group}** age group has the "
                    f"highest average predicted risk at "
                    f"**{highest_age_risk:.1f}%**."
                )

        # NEIGHBOURHOOD INSIGHT

        if "Neighbourhood" in prediction_df.columns:

            neighbourhood_insight = (
                prediction_df
                .groupby("Neighbourhood")["Risk %"]
                .agg(
                    Average_Risk="mean",
                    Patients="count"
                )
                .reset_index()
            )

            neighbourhood_insight = (
                neighbourhood_insight[
                    neighbourhood_insight["Patients"] >= 5
                ]
                .sort_values(
                    "Average_Risk",
                    ascending=False
                )
            )

            if not neighbourhood_insight.empty:

                top_neighbourhood = (
                    neighbourhood_insight.iloc[0]
                )

                insights.append(
                    f"📍 **Neighbourhood pattern:** "
                    f"**{top_neighbourhood['Neighbourhood']}** has the "
                    f"highest average predicted risk among neighbourhoods "
                    f"with at least 5 appointments "
                    f"(**{top_neighbourhood['Average_Risk']:.1f}%**)."
                )


        # ============================================================
        # DISPLAY INSIGHTS
        # ============================================================

        if insights:

            for insight in insights:

                st.info(insight)

        else:

            st.info(
                "Not enough information is available to generate "
                "automatic insights for this prediction dataset."
            )

REQUIRED_COLUMNS = [
        "PatientID",
        "AppointmentID",
        "Gender",
        "ScheduledDay",
        "AppointmentDay",
        "Age",
        "Neighbourhood",
        "Scholarship",
        "Hypertension",
        "Diabetes",
        "Alcoholism",
        "Handicap",
        "SMS_received",
        "PreviousAppointments",
        "PreviousNoShows"]
REQUIRED_NON_NULL = [
        "PatientID",
        "AppointmentID",
        "Gender",
        "ScheduledDay",
        "AppointmentDay",
        "Age",
        "Neighbourhood",]
  
NUMERIC_COLUMNS = [
        "Age",
        "PreviousAppointments",
        "PreviousNoShows"]
display_columns = [
        "PatientID",
        "AppointmentID",
        "Gender",
        "Age",
        "Neighbourhood",
        "AppointmentDay",
        "Risk %",
        "Risk Level",
        "Prediction",
        "Recommended Action"]
MODEL_COLUMN_NAMES = {
        "patient_id": "PatientID",
        "appointment_id": "AppointmentID",
        "scheduled_day": "ScheduledDay",
        "appointment_day": "AppointmentDay",
        "sms_received": "SMS_received",
        "age":"Age",
        "scholarship":"Scholarship",
        "hypertension":"Hypertension",
        "alcoholism":"Alcoholism",
        "handicap":"Handicap",
        "previousappointments":"PreviousAppointments",
        "previousnoshows":"PreviousNoShows",
        "neighbourhood":"Neighbourhood",
        "gender":"Gender",
        "diabetes":"Diabetes"
        }
if page =="📊Batch Upload":
    st.header("📊Multiple Appointments Prediction")
    with open("PredictCare_CSV_Template.csv", "rb") as file:
        st.download_button(
        label="📥 Download Template",
        data=file,
        file_name="PredictCare_CSV_Template.csv",
        mime="text/csv"
    )
    uploaded_file = st.file_uploader("Upload Appointments File",type=["csv","xlsx"])
   
    if uploaded_file is not None:
            dataset_valid='True'
            if uploaded_file.name.endswith(".csv"):
                uploaded_df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                 uploaded_df = pd.read_excel(uploaded_file)
            # Normalize column names
            uploaded_df.columns = (
                uploaded_df.columns
                .str.strip()
                .str.lower()
                .str.replace(r"[\s\-]+", "_", regex=True))

            # Handle common variations
            COLUMN_MAPPING = {
                "patientid": "patient_id",
                "appointmentid": "appointment_id",
                "scheduledday": "scheduled_day",
                "appointmentday": "appointment_day",
                "smsreceived": "sms_received"
            }

            uploaded_df = uploaded_df.rename(
                columns=COLUMN_MAPPING
            )
            uploaded_df = uploaded_df.rename(
            columns=MODEL_COLUMN_NAMES)
            missing_columns = [
                col for col in REQUIRED_COLUMNS
                if col not in uploaded_df.columns]
            if missing_columns:
                st.error("❌ Dataset cannot be processed.")

                st.write("Missing required columns:")

                for col in missing_columns:
                    st.write(f"- {col}")

                st.stop()
        
            # COMPREHENSIVE DATA VALIDATION

            issues = []

            # 1. Missing values

            for col in REQUIRED_COLUMNS:

                if col in uploaded_df.columns:

                    missing_rows = uploaded_df[
                        uploaded_df[col].isna()
                    ].index

                    for row in missing_rows:

                        issues.append({
                            "Row": row + 2,
                            "Column": col,
                            "Issue": "Missing value"
                        })

            # 2. Invalid dates

            DATE_COLUMNS = [
                "ScheduledDay",
                "AppointmentDay"
            ]

            for col in DATE_COLUMNS:

                if col in uploaded_df.columns:

                    converted_dates = pd.to_datetime(
                        uploaded_df[col],
                        errors="coerce"
                    )

                    invalid_rows = uploaded_df[
                        uploaded_df[col].notna() &
                        converted_dates.isna()
                    ].index

                    for row in invalid_rows:

                        issues.append({
                            "Row": row + 2,
                            "Column": col,
                            "Issue": "Invalid date"
                        })

            # 3. Invalid numeric values

            for col in NUMERIC_COLUMNS:

                if col in uploaded_df.columns:

                    converted = pd.to_numeric(
                        uploaded_df[col],
                        errors="coerce"
                    )

                    invalid_rows = uploaded_df[
                        uploaded_df[col].notna() &
                        converted.isna()
                    ].index

                    for row in invalid_rows:

                        issues.append({
                            "Row": row + 2,
                            "Column": col,
                            "Issue": "Invalid numeric value"
                        })

            # 4. Invalid Age

            if "Age" in uploaded_df.columns:

                age_numeric = pd.to_numeric(
                    uploaded_df["Age"],
                    errors="coerce"
                )

                invalid_age_rows = uploaded_df[
                    uploaded_df["Age"].notna() &
                    (
                        age_numeric.isna() |
                        (age_numeric < 0) |
                        (age_numeric > 120)
                    )
                ].index

                for row in invalid_age_rows:

                    issues.append({
                        "Row": row + 2,
                        "Column": "Age",
                        "Issue": "Invalid age — must be between 0 and 120"
                    })

            # 5. Invalid Gender

            if "Gender" in uploaded_df.columns:

                valid_gender = ["male", "female","m","f"]

                gender_values = (
                    uploaded_df["Gender"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )

                invalid_gender_rows = uploaded_df[
                    uploaded_df["Gender"].notna() &
                    ~gender_values.isin(valid_gender)
                ].index

                for row in invalid_gender_rows:

                    issues.append({
                        "Row": row + 2,
                        "Column": "Gender",
                        "Issue": "Invalid value — expected Male\\M or Female\\F"
                    })

            # 6. Invalid Yes/No columns

            YES_NO_COLUMNS = [
                "Scholarship",
                "Hypertension",
                "Diabetes",
                "Alcoholism",
                "Handicap",
                "SMS_received"
            ]

            for col in YES_NO_COLUMNS:

                if col in uploaded_df.columns:

                    values = (
                        uploaded_df[col]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                    )

                    invalid_rows = uploaded_df[
                        uploaded_df[col].notna() &
                        ~values.isin(["yes", "no","y","n","0","1"])
                    ].index

                    for row in invalid_rows:

                        issues.append({
                            "Row": row + 2,
                            "Column": col,
                            "Issue": "Invalid value — expected Yes/Y/1 or No/N/0"
                        })

            # 7. Invalid PreviousAppointments

            if "PreviousAppointments" in uploaded_df.columns:

                previous_apps = pd.to_numeric(
                    uploaded_df["PreviousAppointments"],
                    errors="coerce"
                )

                invalid_rows = uploaded_df[
                    uploaded_df["PreviousAppointments"].notna() &
                    (
                        previous_apps.isna() |
                        (previous_apps < 0)
                    )
                ].index

                for row in invalid_rows:

                    issues.append({
                        "Row": row + 2,
                        "Column": "PreviousAppointments",
                        "Issue": "Invalid value — must be 0 or greater"
                    })

            # 8. Invalid PreviousNoShows

            if "PreviousNoShows" in uploaded_df.columns:

                previous_no_shows = pd.to_numeric(
                    uploaded_df["PreviousNoShows"],
                    errors="coerce"
                )

                invalid_rows = uploaded_df[
                    uploaded_df["PreviousNoShows"].notna() &
                    (
                        previous_no_shows.isna() |
                        (previous_no_shows < 0)
                    )
                ].index

                for row in invalid_rows:

                    issues.append({
                        "Row": row + 2,
                        "Column": "PreviousNoShows",
                        "Issue": "Invalid value — must be 0 or greater"
                    })

            # 9. Logical validation: PreviousNoShows cannot exceed PreviousAppointments

            if "PreviousAppointments" in uploaded_df.columns and "PreviousNoShows" in uploaded_df.columns:

                previous_apps = pd.to_numeric(
                    uploaded_df["PreviousAppointments"],
                    errors="coerce"
                )

                previous_no_shows = pd.to_numeric(
                    uploaded_df["PreviousNoShows"],
                    errors="coerce"
                )

                invalid_rows = uploaded_df[
                    uploaded_df["PreviousAppointments"].notna() &
                    uploaded_df["PreviousNoShows"].notna() &
                    previous_apps.notna() &
                    previous_no_shows.notna() &
                    (previous_no_shows > previous_apps)
                ].index

                for row in invalid_rows:
                    issues.append({
                        "Row": row + 2,
                        "Column": "PreviousNoShows",
                        "Issue": (
                            "Invalid value — PreviousNoShows cannot be "
                            "greater than PreviousAppointments"
                        )
                    })

            # 10. Logical validation: AppointmentDay cannot be before ScheduledDay

            if "ScheduledDay" in uploaded_df.columns and "AppointmentDay" in uploaded_df.columns:

                scheduled_dates = pd.to_datetime(
                    uploaded_df["ScheduledDay"],
                    errors="coerce"
                )

                appointment_dates = pd.to_datetime(
                    uploaded_df["AppointmentDay"],
                    errors="coerce"
                )

                invalid_rows = uploaded_df[
                    scheduled_dates.notna() &
                    appointment_dates.notna() &
                    (appointment_dates < scheduled_dates)
                ].index

                for row in invalid_rows:
                    issues.append({
                        "Row": row + 2,
                        "Column": "AppointmentDay",
                        "Issue": "Appointment date cannot be before scheduled date"
                    })

            if issues:

                issues_df = pd.DataFrame(issues)
                issues_df = issues_df.drop_duplicates()

                st.warning(
                    f"⚠️ {len(issues_df)} data issue(s) detected."
                )

                st.dataframe(
                    issues_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.info(
                    "Please correct the issues shown above in your "
                    "CSV/Excel file and upload it again before generating predictions."
                )

                st.stop()

            else:

                st.success(
                    f"✅ Dataset validated successfully! "
                    f"{len(uploaded_df):,} appointments ready for prediction."
                )
                st.dataframe(uploaded_df)
                if st.button("🔍 Generate Predictions",type="primary",
                        use_container_width=True):
                        processed_df, model_input = prepare_uploaded_data(uploaded_df)
                        risk_probability = model.predict_proba(model_input)[:, 1]
                        processed_df["Risk %"] = risk_probability * 100
                        processed_df["Prediction"] = (
                        processed_df["Risk %"] .apply(lambda x: "Predicted No Show" if x >= THRESHOLD * 100 else "Predicted Show"))
                        processed_df["Risk Level"] = (
                        processed_df["Risk %"]
                        .apply(get_risk_level)
                        )
                
                        processed_df["Recommended Action"] = (
                        processed_df["Risk %"]
                        .apply(get_intervention)
                        )
                        st.session_state["prediction_results"] = processed_df.copy()
                        st.session_state["prediction_source"] = "📊Batch Upload"
                        display_df = processed_df[display_columns].copy()
                        display_df["Gender"] = display_df["Gender"].map({
                             1:"Female",
                             0:"Male"
                        })
                        all_results = display_df.copy()
                        predicted_no_shows = display_df[
                        display_df["Prediction"] == "Predicted No Show"
                        ].copy()
                
                        predicted_shows = display_df[
                        display_df["Prediction"] == "Predicted Show"
                        ].copy()
                        st.divider()
                        col1,col2,col3 = st.columns(3)
                        with col1:
                                st.write("Total processed:", len(processed_df))
                        with col2:
                                st.write("Predicted No-shows:", len(predicted_no_shows))      
                        with col3:
                             st.write("Predicted Shows:", len(predicted_shows))                       
                        view1, view2, view3 = st.tabs([
                        "📋 All Appointments",
                        "⚠️ Predicted No-Shows",
                        "✅ Predicted Shows"
                        ])
                        with view1:
                            st.dataframe(all_results)
                            st.download_button(label="📥Download All Predictions",
                                               data=all_results.to_csv(index=False),file_name="PredictCare_All_Predictions.csv",
                                               mime='text/csv')
                        with view2:
                            st.dataframe(predicted_no_shows)
                            st.download_button(label="📥Download Predicted No-Shows",
                                               data=predicted_no_shows.to_csv(index=False),
                                               file_name="PredictCare_Predicted_NoShows.csv",
                                               mime='text/csv')
                        with view3:
                            st.dataframe(predicted_shows)
                            st.download_button(label="📥 Download Predicted Shows",
                            data=predicted_shows.to_csv(index=False),
                            file_name="PredictCare_Predicted_Shows.csv",
                            mime="text/csv")
    else:
        st.error("File not uploaded!")

# ==============================
# SINGLE PATIENT PREDICTION
# ==============================
if page=="👤Single Patient":
     st.header("👤 Single Patient Prediction")
     st.subheader("Enter patient data:")
     PatientID = st.text_input("PatientID:")
     AppointmentID = st.text_input("AppointmentID:")
     Gender = st.selectbox("Gender:",["Male","Female"])
     ScheduledDay = st.datetime_input("ScheduledDay:")
     AppointmentDay = st.datetime_input("AppointmentDay:")
     Age = st.number_input("Age:",min_value=0)
     Neighbourhood = st.text_input("Neighbourhood:")
     Scholarship = st.selectbox("Scholarship:",["Yes","No"])
     Diabetes = st.selectbox("Diabetes:",["Yes","No"])
     Hypertension = st.selectbox("Hypertension:",["Yes","No"])
     Handicap = st.selectbox("Handicap:",["Yes","No"])
     Alcoholism = st.selectbox("Alcoholism:",["Yes","No"])
     SMS_received = st.selectbox("SMS received:",["Yes","No"])
     PreviousAppointments = st.number_input("Previous Appointments:",min_value=0)
     PreviousNoShows = st.number_input("Previous No Shows:",min_value=0)
     if st.button("🔍Generate Predictions",type="primary",
             use_container_width=True):
                                      
                                      if not Neighbourhood.strip():
                                                                    st.error("Neighbourhood is required.")
                                                                    st.stop()

                                      if PreviousNoShows > PreviousAppointments:
                                        st.error("❌ Previous No-Shows cannot be greater than Previous Appointments.")
                                        st.stop()

                                      if ScheduledDay>AppointmentDay:
                                        st.error("❌ Appointment Date cannot be earlier than Scheduled Date!")
                                        st.stop()
                        
                                      uploaded_df = pd.DataFrame({
                                                            "PatientID":[PatientID],
                                                            "AppointmentID":[AppointmentID],
                                                            "Gender":[Gender],
                                                            "ScheduledDay":[ScheduledDay],
                                                            "AppointmentDay":[AppointmentDay],
                                                            "Age":[Age],
                                                            "Neighbourhood":[Neighbourhood],
                                                            "Scholarship":[Scholarship],
                                                            "Hypertension":[Hypertension],
                                                            "Diabetes":[Diabetes],
                                                            "Alcoholism":[Alcoholism],
                                                            "Handicap":[Handicap],
                                                            "SMS_received":[SMS_received],
                                                            "PreviousAppointments":[PreviousAppointments],
                                                            "PreviousNoShows":[PreviousNoShows]})
                                      processed_df, model_input = prepare_uploaded_data(uploaded_df)
                                      risk_probability = model.predict_proba(model_input)[:, 1]
                                      processed_df["Risk %"] = risk_probability * 100
                                      processed_df["Prediction"] = (
                                      processed_df["Risk %"] .apply(lambda x: "Predicted No Show" if x >= THRESHOLD * 100 else "Predicted Show"))
                                      processed_df["Risk Level"] = (processed_df["Risk %"]
                                                            .apply(get_risk_level))
                                      processed_df["Recommended Action"] = (processed_df["Risk %"]
                                      .apply(get_intervention))
                                      st.session_state["prediction_results"] = processed_df.copy()
                                      st.session_state["prediction_source"] = "👤Single Patient"
                                      result=processed_df[display_columns]
                                      result["Gender"]=result["Gender"].map({
                                           0:"Male",
                                           1:"Female"
                                      })
                                      st.dataframe(result)
# ====================
# QUICK PREDICTION
# ====================
if page=="📝Quick Prediction":
    st.header("📝 Quick Prediction")
    st.write(
        "Enter appointment information directly below. "
        "PredictCare will calculate the no-show risk and "
        "recommend an appropriate intervention."
    )
    empty_df = pd.DataFrame(columns=REQUIRED_COLUMNS)

    edited_df = st.data_editor(
    empty_df,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "PatientID": st.column_config.TextColumn(
            "Patient ID"
        ),

        "AppointmentID": st.column_config.TextColumn(
            "Appointment ID"
        ),

        "Gender": st.column_config.SelectboxColumn(
            "Gender",
            options=["Male", "Female"]
        ),

        "ScheduledDay": st.column_config.DatetimeColumn(
            "Scheduled Date",
            format="DD/MM/YYYY HH:mm"
        ),

        "AppointmentDay": st.column_config.DatetimeColumn(
            "Appointment Date",
            format="DD/MM/YYYY HH:mm"
        ),

        "Age": st.column_config.NumberColumn(
            "Age",
            min_value=0,
            max_value=120,
            step=1
        ),

        "Neighbourhood": st.column_config.TextColumn(
            "Neighbourhood"
        ),

        "Scholarship": st.column_config.SelectboxColumn(
            "Scholarship",
            options=["Yes", "No"]
        ),

        "Hypertension": st.column_config.SelectboxColumn(
            "Hypertension",
            options=["Yes", "No"]
        ),

        "Diabetes": st.column_config.SelectboxColumn(
            "Diabetes",
            options=["Yes", "No"]
        ),

        "Alcoholism": st.column_config.SelectboxColumn(
            "Alcoholism",
            options=["Yes", "No"]
        ),

        "Handicap": st.column_config.SelectboxColumn(
            "Handicap",
            options=["Yes", "No"]
        ),

        "SMS_received": st.column_config.SelectboxColumn(
            "SMS Received",
            options=["Yes", "No"]
        ),
        "PreviousAppointments": st.column_config.NumberColumn(
                    "Previous Appointments",
                    min_value=0,
                    max_value=120,
                    step=1
        ),
        "PreviousNoShows": st.column_config.NumberColumn(
                    "Previous No Shows",
                    min_value=0,
                    max_value=120,
                    step=1
        ),
    }
)


# PREDICTIONS

    if st.button(
        "🔮 Generate Predictions",
        type="primary",
        use_container_width=True
    ):
         if edited_df.empty:
         
                     st.warning(
                         "⚠️ Please enter at least one appointment."
                     )
                     st.stop()
         issues=[]
         for col in REQUIRED_COLUMNS:
                        if col in edited_df.columns:
        
                            missing_rows = edited_df[
                                edited_df[col].isna()
                            ].index
        
                            for row in missing_rows:
        
                                issues.append({
                                    "Row": row + 1,
                                    "Column": col,
                                    "Issue": "Missing value"})
         for row in edited_df.index:
                scheduled_day = edited_df.loc[row, "ScheduledDay"]
                appointment_day = edited_df.loc[row, "AppointmentDay"]

                if pd.notna(scheduled_day) and pd.notna(appointment_day):

                    if scheduled_day > appointment_day:

                        issues.append({
                            "Row": row + 1,
                            "Column": "ScheduledDay / AppointmentDay",
                            "Issue": "Invalid dates - Appointment Date cannot be earlier than Scheduled Date"
                        }) 
         for row in edited_df.index:
                          previous_no_shows = edited_df.loc[row, "PreviousNoShows"]
                          previous_appointments = edited_df.loc[row, "PreviousAppointments"]
          
                          if pd.notna(previous_no_shows) and pd.notna(previous_appointments):
          
                              if previous_no_shows > previous_appointments:
          
                                  issues.append({
                                      "Row": row + 1,
                                      "Column": "Previous Appointments / Previous No Shows",
                                      "Issue": "Previous No Shows cannot be greater than Previous Appointments!"
                                  }) 
         if issues:
                        issues_df = pd.DataFrame(issues)
                        issues_df = issues_df.drop_duplicates()
        
                        st.warning(
                            f"⚠️ {len(issues_df)} issues(s) detected."
                        )
        
                        st.dataframe(
                            issues_df,
                            use_container_width=True,
                            hide_index=True
                        )
        
                        st.info(
                            "Please fix this issues listed above to continue!")
        
                        st.stop()

         prediction_df = edited_df.copy()

         processed_df, model_input = prepare_uploaded_data(
            prediction_df
        )

         risk_probability = (
            model.predict_proba(model_input)[:, 1]
        )

         processed_df["Risk %"] = (
            risk_probability * 100
        )

         processed_df["Prediction"] = (
            processed_df["Risk %"]
            .apply(
                lambda x:
                "Predicted No Show"
                if x >= THRESHOLD * 100
                else "Predicted Show"
            )
        )

         processed_df["Risk Level"] = (
            processed_df["Risk %"]
            .apply(get_risk_level)
        )

         processed_df["Recommended Action"] = (
            processed_df["Risk %"]
            .apply(get_intervention)
        ) 
         processed_df["Gender"]=processed_df["Gender"].map({
            1:"Female",
            0:"Male"
        })
         display = processed_df.copy()
         display = display[display_columns]
         st.session_state["prediction_results"] = processed_df.copy()
         st.session_state["prediction_source"] = "📝Quick Prediction"

        # DISPLAY

         st.subheader("📊 Prediction Results")

         st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )
         
         st.download_button(
            label="📥 Download Predictions",
            data=display.to_csv(index=False),
            file_name="PredictCare_Quick_Predictions.csv",
            mime="text/csv"
        )

# =================
# MODEL INSIGHTS
# =================

if page == "📈Model Insights":

    st.header("📈 Model Insights")

    st.write(
        "Understand how PredictCare's machine learning model evaluates "
        "appointment no-show risk and supports proactive intervention."
    )

    st.divider()

    st.subheader("🔮 Prediction Model")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Model",
            "HistGradientBoosting"
        )

    with col2:
        st.metric(
            "Decision Threshold",
            f"{THRESHOLD:.2f}"
        )

    with col3:
        st.metric(
            "Prediction Type",
            "No-Show Risk"
        )

    st.info(
        f"""
        PredictCare uses a **HistGradientBoostingClassifier** to estimate
        the probability that an appointment will result in a no-show.

        The current decision threshold is **{THRESHOLD:.2f}**.

        Appointments with a predicted no-show probability at or above
        this threshold are classified as **Predicted No-Show**.
        """
    )

    st.divider()

    model_comparison = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Threshold": 0.50,
        "Accuracy": 80.01,
        "Precision (No-Show)": 55.3,
        "Recall (No-Show)": 5.6,
        "F1 (No-Show)": 10.1,
        "ROC-AUC": 52.2
    },
    {
        "Model": "Logistic Regression",
        "Threshold": 0.25,
        "Accuracy": 66.2,
        "Precision (No-Show)": 33.0,
        "Recall (No-Show)": 65.6,
        "F1 (No-Show)": 43.9,
        "ROC-AUC": 66.0
    },

    {
        "Model": "Decision Tree",
        "Threshold": 0.50,
        "Accuracy": 72.91,
        "Precision (No-Show)": 32.9,
        "Recall (No-Show)": 33.2,
        "F1 (No-Show)": 33.1,
        "ROC-AUC": 58.0
    },
    {
        "Model": "Decision Tree",
        "Threshold": 0.25,
        "Accuracy": 72.9,
        "Precision (No-Show)": 32.9,
        "Recall (No-Show)": 33.2,
        "F1 (No-Show)": 33.1,
        "ROC-AUC": 58.0
    },

    {
        "Model": "Random Forest",
        "Threshold": 0.50,
        "Accuracy": 80.3,
        "Precision (No-Show)": 54.3,
        "Recall (No-Show)": 15.2,
        "F1 (No-Show)": 23.8,
        "ROC-AUC": 56.0
    },
    {
        "Model": "Random Forest",
        "Threshold": 0.25,
        "Accuracy": 68.6,
        "Precision (No-Show)": 34.8,
        "Recall (No-Show)": 63.5,
        "F1 (No-Show)": 44.9,
        "ROC-AUC": 66.7
    },

    {
        "Model": "Hist Gradient Boosting",
        "Threshold": 0.50,
        "Accuracy": 80.4,
        "Precision (No-Show)": 62.0, 
        "Recall (No-Show)": 7.2,     
        "F1 (No-Show)": 12.9,         
        "ROC-AUC": 53.1
    },
    {
        "Model": "Hist Gradient Boosting",
        "Threshold": 0.25,
        "Accuracy": 68.0,
        "Precision (No-Show)": 34.7,
        "Recall (No-Show)": 66.4,
        "F1 (No-Show)": 45.6,
        "ROC-AUC": 67.4
    }
    ])

    st.subheader("🧠 Why HistGradientBoostingClassifier?")

    st.write("PredictCare tested four models — Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting, " \
    "to determine which was most suitable for no-show prediction. Since identifying potential no-shows is the main objective, models were compared using accuracy, precision, recall, F1-score, and ROC-AUC at both the default 0.50 and tuned 0.25 thresholds.")
    st.subheader("Models Tested:")
    st.dataframe(
        model_comparison,
        use_container_width=True,
        hide_index=True
    )
    st.subheader("🎯 Why Threshold = 0.25?")

    st.write(
        "We use a lower threshold to catch more potential no-shows "
        "(higher recall)."
    )

    threshold_50, threshold_25 = st.columns(2)

    with threshold_50:
        with st.container(border=True):

            st.markdown("#### Threshold 0.50")

            st.write("**Precision**")
            st.write("62.0%")

            st.write("**Recall**")
            st.write("7.2%")

            st.write("**F1 Score**")
            st.write("12.9%")


    with threshold_25:
        with st.container(border=True):

            st.markdown("#### Threshold 0.25")

            st.write("**Precision**")
            st.write("34.7%")

            st.write("**Recall**")
            st.write("66.4%")

            st.write("**F1 Score**")
            st.write("45.6%")


    st.success(
        "✅ Better for hospitals → identifies more at-risk "
        "patients for early intervention."
)

    st.subheader("📊 Model Performance")

    accuracy = 0.68
    roc_auc = 67.4
    precision = 0.35
    recall = 0.66
    f1_score = 0.46
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )
    with col2:
        st.metric(
            "ROC-AUC",
            f"{roc_auc:.3f}%"
        )
    with col3:
        st.metric(
            "Precision",
            f"{precision * 100:.1f}%"
        )
        st.caption("For No-show")
    with col4:
        st.metric(
            "Recall",
            f"{recall * 100:.1f}%"
        )
        st.caption("For No-show")
    with col5:
        st.metric(
            "F1 Score",
            f"{f1_score * 100:.1f}%"
        )
        st.caption("For No-show")

    st.caption(
        "Performance values are based on the model evaluation performed "
        "during development."
    )
    st.divider()

    st.subheader("🎯 Classification Performance")
    st.markdown("### Confusion Matrix")

    cm = pd.DataFrame(
        [
            [18106, 8357],
            [2246, 4448]
        ],
        index=[
            "Actual Show",
            "Actual No-Show"
        ],
        columns=[
            "Predicted Show",
            "Predicted No-Show"
        ]
    )

    st.dataframe(
        cm,
        use_container_width=True
    )

    st.divider()

    st.subheader("📈 ROC-AUC")

    st.write(
        """
        ROC-AUC measures how well the model separates appointments that
        are likely to be attended from appointments that are likely to
        result in a no-show.

        A value closer to **1.0** indicates stronger discrimination,
        while **0.5** represents performance similar to random guessing.
        """
    )

    roc_data = pd.DataFrame({
        "Metric": ["ROC-AUC"],
        "Score": [roc_auc]
    })

    st.bar_chart(
        roc_data,
        y="Score"
    )

    st.subheader("🧩 Features Used by PredictCare")

    feature_data = pd.DataFrame({
        "Features": [
            "Gender",
            "Age",
            "Neighbourhood",
            "Scholarship",
            "Hypertension",
            "Diabetes",
            "Alcoholism",
            "Handicap",
            "SMS Received",
            "Waiting Days",
            "Appointment Day",
            "Scheduled Day",
            "Scheduled Hour",
            "Appointment Month",
            "Previous Appointments",
            "Previous No-Shows",
            "Previous No-Show Rate",
            "Same-Day Appointment"
        ],
        "Purpose": [
            "Patient demographic information",
            "Patient age",
            "Appointment location",
            "Scholarship status",
            "Hypertension status",
            "Diabetes status",
            "Alcoholism status",
            "Handicap status",
            "Whether an SMS reminder was received",
            "Time between scheduling and appointment",
            "Day of the appointment",
            "Day the appointment was scheduled",
            "Time at which appointment was scheduled",
            "Appointment month",
            "Previous appointment history",
            "Previous no-show history",
            "Historical no-show proportion",
            "Whether appointment was scheduled for the same day"
        ]
    })

    st.dataframe(
        feature_data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("⚠️ Important Interpretation")

    st.warning(
        """
        PredictCare produces **risk estimates**, not guaranteed outcomes.

        A high-risk prediction does not mean that a patient will definitely
        miss their appointment. It indicates that the appointment has
        characteristics associated with a higher estimated probability
        of no-show.

        The system should therefore be used as a **decision-support tool**
        to help hospital staff prioritize reminders and follow-ups.
        """
    )

    st.success(
        "💡 The goal is not to replace hospital staff decisions — "
        "it is to help them identify appointments that may deserve "
        "attention earlier."
    )
# ========================
# ABOUT PREDICTCARE
# ========================

if page == "ℹ️About PredictCare":

    st.header("ℹ️ About PredictCare")

    st.write(
        "A machine-learning powered hospital appointment risk and "
        "intervention system."
    )

    st.divider()

    st.subheader("🏥 What is PredictCare?")

    st.write(
        """
        **PredictCare** is a hospital appointment management and
        no-show prediction system designed to help healthcare
        organizations identify appointments that may have a higher
        risk of patient non-attendance.

        Instead of waiting until an appointment is missed, PredictCare
        analyzes appointment and patient-related information to estimate
        no-show risk before the appointment takes place.

        The system then converts that risk into an understandable
        recommendation so hospital staff can decide whether additional
        follow-up may be appropriate.
        """
    )

    st.divider()

    st.subheader("🎯 The Problem")

    st.write(
        """
        Missed appointments can create several operational problems
        for healthcare organizations.

        They may result in:

        • Unused appointment slots  
        • Reduced staff and resource utilization  
        • Longer waiting times for other patients  
        • Lost opportunities to provide care  
        • Additional administrative workload  

        PredictCare aims to help hospitals move from a reactive approach
        toward a more proactive appointment management process.
        """
    )

    st.divider()

    st.subheader("⚙️ How PredictCare Works")

    step1, step2, step3, step4 = st.columns(4)

    with step1:
        st.markdown("### 1️⃣")
        st.markdown("**Enter Data**")
        st.write(
            "Appointment and patient information is entered manually "
            "or uploaded in bulk."
        )

    with step2:
        st.markdown("### 2️⃣")
        st.markdown("**Process Data**")
        st.write(
            "PredictCare validates the information and creates the "
            "features required by the model."
        )

    with step3:
        st.markdown("### 3️⃣")
        st.markdown("**Predict Risk**")
        st.write(
            "The machine-learning model estimates the probability "
            "of appointment non-attendance."
        )

    with step4:
        st.markdown("### 4️⃣")
        st.markdown("**Take Action**")
        st.write(
            "Risk levels are converted into recommended intervention "
            "strategies."
        )

    st.divider()

    st.subheader("✨ Key Features")

    features = pd.DataFrame({
        "Feature": [
            "✅ Live Dashboard",
            "✅ Single Patient Prediction",
            "✅ Batch Prediction",
            "✅ Quick Prediction",
            "✅ Risk Classification",
            "✅ Recommended Intervention",
            "✅ Attendance Insights",
            "✅ Prediction Downloads",
            "✅ Data Validation"
        ],
        "Description": [
            "Overview of appointment risk and attendance patterns.",
            "Generate a prediction for an individual appointment.",
            "Process multiple appointments from CSV or Excel files.",
            "Enter multiple appointments directly through an interactive table.",
            "Appointments are categorized according to predicted risk.",
            "Provides an appropriate follow-up recommendation based on risk.",
            "Helps identify factors associated with predicted no-show risk.",
            "Export prediction results for further hospital use.",
            "Detect missing, invalid, or incorrectly formatted data before prediction."
        ]
    })

    st.dataframe(
        features,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("🚦 Risk & Intervention System")

    risk_data = pd.DataFrame({
        "Risk Level": [
            "🟢Low",
            "🟡Medium",
            "🟠High",
            "🔴Very High"
        ],
        "Risk Range": [
            "< 30%",
            "30% – < 60%",
            "60% – < 80%",
            "≥ 80%"
        ],
        "Suggested Approach": [
            "Automated reminder",
            "Strong reminder + confirmation request",
            "Confirmation + closer follow-up",
            "Staff follow-up"
        ]
    })

    st.dataframe(
        risk_data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("💻 Machine Learning")

    st.write(
        """
        PredictCare uses a **HistGradientBoostingClassifier**.

        The model receives appointment and patient-related features,
        processes categorical information through encoding, and uses
        the resulting feature representation to estimate the probability
        of a no-show.

        The predicted probability is then converted into a risk percentage
        and risk category.
        """
    )

    st.divider()

    st.subheader("🌱 Project Vision")

    st.write(
        """
        PredictCare is designed around a simple principle:

        ### **Predict Earlier. Care Better.**

        By identifying potentially high-risk appointments before they
        occur, healthcare staff can focus their attention where it may
        have the greatest operational value.

        The long-term vision is to develop PredictCare into a more
        comprehensive appointment intelligence platform capable of
        supporting hospital scheduling, patient communication and
        resource planning.
        """
    )

    st.divider()

    st.subheader("⚠️ Disclaimer")

    st.warning(
        """
        PredictCare is a machine-learning based decision-support system.

        Predictions are estimates generated from the information supplied
        to the system and should not be treated as certain outcomes.

        Hospital staff should use professional judgment when deciding
        how to respond to a prediction.
        """
    )

    st.divider()

    st.markdown(
        """
        <div style="text-align:center; padding:20px 0;">
            <h3>🏥 PredictCare</h3>
            <p><i>Predict Earlier. Care Better.</i></p>
            <p>Hospital Appointment Risk & Intervention System</p>
        </div>
        """,
        unsafe_allow_html=True
    )


