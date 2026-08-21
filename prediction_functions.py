import pandas as pd

def get_risk_level(risk):
    if risk < 30:
        return "🟢Low"
    elif risk < 60:
        return "🟡Medium"
    elif risk < 80:
        return "🟠High"
    else:
        return "🔴Very High"
def get_intervention(risk):
    if risk < 30:
        return "Automated reminder"
    elif risk < 60:
        return "Strong reminder + confirmation request"
    elif risk < 80:
        return "Confirmation required + closer follow-up"
    else:
        return "Staff follow-up"

def prepare_uploaded_data(uploaded_df):

    df = uploaded_df.copy()
    df.columns = df.columns.str.strip()

    # Convert Gender
    df["Gender"] = (
        df["Gender"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "male": 0,
            "female": 1,
            "f":1,
            "m":0
        })
    )

    # 3. Convert Yes/No columns
    binary_columns = [
        "Scholarship",
        "Hypertension",
        "Diabetes",
        "Alcoholism",
        "Handicap",
        "SMS_received"
    ]

    for col in binary_columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "n":0,
                "no": 0,
                "yes": 1,
                "y":1,
                "0":0,
                "1":1
            })
        )

    # Convert dates
    df["ScheduledDay"] = pd.to_datetime(
        df["ScheduledDay"],
        errors="coerce"
    )

    df["AppointmentDay"] = pd.to_datetime(
        df["AppointmentDay"],
        errors="coerce"
    )

    # Feature engineering

    # Waiting days
    df["WaitingDays"] = (
        df["AppointmentDay"].dt.normalize()
        - df["ScheduledDay"].dt.normalize()
    ).dt.days

    # Appointment day of week
    df["AppointmentDayOfWeek"] = (
        df["AppointmentDay"].dt.dayofweek
    )

    # Scheduled day of week
    df["ScheduledDayOfWeek"] = (
        df["ScheduledDay"].dt.dayofweek
    )

    # Scheduled hour
    df["ScheduledHour"] = (
        df["ScheduledDay"].dt.hour
    )

    # Appointment month
    df["AppointmentMonth"] = (
        df["AppointmentDay"].dt.month
    )

    # Appointment day of month
    df["AppointmentDayOfMonth"] = (
        df["AppointmentDay"].dt.day
    )

    # Previous shows
    df["PreviousShows"] = (
        df["PreviousAppointments"]
        - df["PreviousNoShows"]
    )

    # Previous no-show rate
    df["PreviousNoShowRate"] = (
        df["PreviousNoShows"]
        / df["PreviousAppointments"]
    ).fillna(0)

    # Same-day appointment
    df["SameDayAppointment"] = (
        df["WaitingDays"] == 0
    ).astype(int)

    model_columns = [
        "Gender",
        "Age",
        "Neighbourhood",
        "Scholarship",
        "Hypertension",
        "Diabetes",
        "Alcoholism",
        "Handicap",
        "SMS_received",
        "WaitingDays",
        "AppointmentDayOfWeek",
        "ScheduledDayOfWeek",
        "ScheduledHour",
        "AppointmentMonth",
        "AppointmentDayOfMonth",
        "PreviousAppointments",
        "PreviousNoShows",
        "PreviousNoShowRate",
        "SameDayAppointment"
    ]

    model_data = df[model_columns].copy()

    return df, model_data
