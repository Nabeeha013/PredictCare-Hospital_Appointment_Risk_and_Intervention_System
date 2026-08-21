import plotly.graph_objects as go
import pandas as pd

from theme import COLORS
import pandas as pd
import plotly.express as px


def waiting_days_no_show_chart(df):
    data = df.copy()

    if "No_show" not in data.columns:
        return None

    data["WaitingDays"] = pd.to_numeric(
        data["WaitingDays"], errors="coerce"
    )

    data = data.dropna(subset=["WaitingDays", "No_show"])

    bins = [-1, 0, 3, 7, 14, 30, float("inf")]
    labels = [
        "Same Day",
        "1–3 Days",
        "4–7 Days",
        "8–14 Days",
        "15–30 Days",
        "30+ Days"
    ]

    data["Waiting Group"] = pd.cut(
        data["WaitingDays"],
        bins=bins,
        labels=labels
    )

    summary = (
        data.groupby("Waiting Group", observed=False)["No_show"]
        .mean()
        .reset_index()
    )

    summary["No-Show Rate"] = summary["No_show"] * 100

    fig = px.bar(
        summary,
        x="Waiting Group",
        y="No-Show Rate",
        text="No-Show Rate",
        title="Waiting Days vs No-Show Rate"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="No-Show Rate (%)",
        xaxis_title="Waiting Period",
        yaxis_range=[0, max(summary["No-Show Rate"].max() * 1.2, 10)]
    )

    return fig


def sms_no_show_chart(df):
    data = df.copy()

    if "No_show" not in data.columns or "SMS_received" not in data.columns:
        return None

    data["SMS_received"] = (
        data["SMS_received"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "yes": "SMS Received",
            "no": "No SMS"
        })
    )

    data = data.dropna(subset=["SMS_received", "No_show"])

    summary = (
        data.groupby("SMS_received")["No_show"]
        .mean()
        .reset_index()
    )

    summary["No-Show Rate"] = summary["No_show"] * 100

    fig = px.bar(
        summary,
        x="SMS_received",
        y="No-Show Rate",
        text="No-Show Rate",
        title="SMS Received vs No-Show Rate"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="SMS Status",
        yaxis_title="No-Show Rate (%)"
    )

    return fig


def previous_noshows_chart(df):
    data = df.copy()

    if "No_show" not in data.columns:
        return None

    data["PreviousNoShows"] = pd.to_numeric(
        data["PreviousNoShows"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["PreviousNoShows", "No_show"]
    )

    data["Previous No-Show Group"] = data[
        "PreviousNoShows"
    ].apply(
        lambda x:
        "0" if x == 0
        else "1" if x == 1
        else "2" if x == 2
        else "3+"
    )

    order = ["0", "1", "2", "3+"]

    summary = (
        data.groupby(
            "Previous No-Show Group",
            observed=False
        )["No_show"]
        .mean()
        .reindex(order)
        .reset_index()
    )

    summary["No-Show Rate"] = summary["No_show"] * 100

    fig = px.bar(
        summary,
        x="Previous No-Show Group",
        y="No-Show Rate",
        text="No-Show Rate",
        title="Previous No-Shows vs Current No-Show Rate"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="Previous No-Shows",
        yaxis_title="Current No-Show Rate (%)"
    )

    return fig


def appointment_day_chart(df):
    data = df.copy()

    if "No_show" not in data.columns:
        return None

    data["No_show"] = pd.to_numeric(
        data["No_show"],
        errors="coerce"
    )

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    if "AppointmentDayOfWeek" in data.columns:

        if pd.api.types.is_numeric_dtype(
            data["AppointmentDayOfWeek"]
        ):
            data["Day"] = data[
                "AppointmentDayOfWeek"
            ].map({
                0: "Monday",
                1: "Tuesday",
                2: "Wednesday",
                3: "Thursday",
                4: "Friday",
                5: "Saturday",
                6: "Sunday"
            })

        else:
            data["Day"] = data[
                "AppointmentDayOfWeek"
            ]

    else:
        return None

    data = data.dropna(subset=["Day", "No_show"])

    summary = (
        data.groupby("Day")["No_show"]
        .mean()
        .reindex(day_order)
        .dropna()
        .reset_index()
    )

    summary["No-Show Rate"] = summary["No_show"] * 100

    fig = px.bar(
        summary,
        x="Day",
        y="No-Show Rate",
        text="No-Show Rate",
        title="Appointment Day vs No-Show Rate"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="Appointment Day",
        yaxis_title="No-Show Rate (%)"
    )

    return fig


def _base_layout(title="", height=300):
    return dict(
        title=dict(text=title, font=dict(size=13, color=COLORS["navy"])),
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5, xanchor="center"),
    )


def donut_show_noshow(show_count, noshow_count):
    total = show_count + noshow_count
    show_pct = show_count / total * 100
    noshow_pct = noshow_count / total * 100

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Showed Up", "No-Show"],
                values=[show_count, noshow_count],
                hole=0.62,
                marker=dict(colors=[COLORS["blue"], COLORS["coral"]]),
                textinfo="label+percent",
                textfont=dict(size=12),
                hovertemplate="%{label}: %{value:,}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        **_base_layout(height=310),
        annotations=[
            dict(
                text=f"<b>{total:,}</b><br>Total",
                x=0.5,
                y=0.5,
                font=dict(size=14, color=COLORS["navy"]),
                showarrow=False,
            )
        ],
    )
    fig.data[0].textposition = "outside"
    return fig, show_pct, noshow_pct


def age_group_chart(dataset):
    age_data = dataset.copy()
    age_data["AgeGroup"] = pd.cut(
        age_data["Age"],
        bins=[-1, 18, 30, 45, 60, 200],
        labels=["0-18", "19-30", "31-45", "46-60", "61+"],
    )
    rates = (
        age_data.groupby("AgeGroup", observed=False)["No_show"]
        .mean()
        .mul(100)
        .reset_index()
    )
    rates.columns = ["AgeGroup", "Rate"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=rates["AgeGroup"],
            y=rates["Rate"],
            name="No-Show Rate",
            marker_color=COLORS["blue"],
            opacity=0.85,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rates["AgeGroup"],
            y=rates["Rate"],
            mode="lines+markers",
            name="Trend",
            line=dict(color=COLORS["coral"], width=2.5),
            marker=dict(size=8),
        )
    )
    fig.update_layout(
        **_base_layout(height=310),
        yaxis_title="No-Show Rate (%)",
        xaxis_title="",
        barmode="overlay",
    )
    return fig


def sms_chart(dataset):
    sms = (
        dataset.groupby("SMS_received")["No_show"]
        .mean()
        .mul(100)
        .reset_index()
    )
    labels = ["No SMS" if x == 0 else "SMS Received" for x in sms["SMS_received"]]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=sms["No_show"],
                marker_color=[COLORS["coral"], COLORS["blue"]],
                text=[f"{v:.1f}%" for v in sms["No_show"]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(**_base_layout(height=310), yaxis_title="No-Show Rate (%)")
    return fig


def waiting_days_chart(dataset):
    wait_data = dataset.copy()
    wait_data["WaitingGroup"] = pd.cut(
        wait_data["WaitingDays"],
        bins=[-1, 3, 7, 14, 1000],
        labels=["0-3", "4-7", "8-14", "15+"],
    )
    rates = (
        wait_data.groupby("WaitingGroup", observed=False)["No_show"]
        .mean()
        .mul(100)
        .reset_index()
    )

    fig = go.Figure(
        data=[
            go.Scatter(
                x=rates["WaitingGroup"].astype(str),
                y=rates["No_show"],
                mode="lines+markers",
                fill="tozeroy",
                fillcolor="rgba(74, 144, 217, 0.15)",
                line=dict(color=COLORS["blue"], width=3),
                marker=dict(size=9, color=COLORS["teal"]),
            )
        ]
    )
    fig.update_layout(
        **_base_layout(height=310),
        xaxis_title="Waiting Days",
        yaxis_title="No-Show Rate (%)",
        showlegend=False,
    )
    return fig
