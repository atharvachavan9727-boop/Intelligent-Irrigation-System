
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from fuzzy_model import predict_demand

st.set_page_config(
    page_title="Intelligent Irrigation System",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Intelligent Irrigation System Using Fuzzy Logic")
st.caption(
    "Soft Computing Project | Fuzzy irrigation-demand decision support"
)

DATA_PATH = "irrigation_dashboard_data.csv"

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["ts"] = pd.to_datetime(data["ts"], errors="coerce")
    data["month"] = data["ts"].dt.to_period("M").astype(str)
    return data

try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Dataset file not found. Place "
        "irrigation_dashboard_data.csv beside app.py."
    )
    st.stop()

required = [
    "soil_moisture",
    "weather_temp",
    "weather_humidity",
    "fuzzy_demand_score",
    "fuzzy_demand_level"
]

missing = [col for col in required if col not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Sidebar navigation
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Live Prediction",
        "Historical Explorer",
        "About the Model"
    ]
)

# Common category colors for text labels
def demand_label(score):
    if score <= 30:
        return "Low"
    elif score <= 70:
        return "Medium"
    return "High"


if page == "Overview":
    st.header("Dashboard Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", f"{len(df):,}")
    c2.metric(
        "Average Demand Score",
        f"{df['fuzzy_demand_score'].mean():.2f}/100"
    )
    c3.metric(
        "Mean Soil Moisture",
        f"{df['soil_moisture'].mean():.2f}"
    )
    c4.metric(
        "Mean Temperature",
        f"{df['weather_temp'].mean():.1f} °C"
    )

    st.subheader("Demand-Level Distribution")
    counts = (
        df["fuzzy_demand_level"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
    )
    st.bar_chart(counts)

    st.subheader("Monthly Trends")
    monthly = (
        df.groupby("month", as_index=False)
        .agg(
            mean_soil_moisture=("soil_moisture", "mean"),
            mean_demand_score=("fuzzy_demand_score", "mean"),
            mean_temperature=("weather_temp", "mean"),
            mean_humidity=("weather_humidity", "mean")
        )
    )

    st.line_chart(
        monthly.set_index("month")[
            ["mean_soil_moisture", "mean_demand_score"]
        ]
    )

    st.caption(
        "Soil moisture uses the dataset's relative scale. "
        "Demand scores are fuzzy-model outputs, not water-volume estimates."
    )


elif page == "Live Prediction":
    st.header("Interactive Fuzzy Prediction")
    st.write(
        "Adjust the sensor inputs to estimate the irrigation-demand score."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        moisture = st.slider(
            "Soil Moisture",
            min_value=0.0,
            max_value=80.0,
            value=25.0,
            step=1.0
        )

    with c2:
        temp = st.slider(
            "Air Temperature (°C)",
            min_value=0.0,
            max_value=50.0,
            value=30.0,
            step=1.0
        )

    with c3:
        humidity = st.slider(
            "Relative Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            step=1.0
        )

    score = predict_demand(moisture, temp, humidity)
    label = demand_label(score)

    st.divider()
    a, b = st.columns(2)
    a.metric("Fuzzy Demand Score", f"{score:.2f}/100")
    b.metric("Demand Category", label)

    st.progress(int(round(score)))

    st.info(
        "This is a decision-support score. It does not directly specify "
        "irrigation duration, water volume, or a verified watering schedule."
    )


elif page == "Historical Explorer":
    st.header("Historical Data Explorer")

    months = sorted(df["month"].dropna().unique().tolist())
    selected_months = st.multiselect(
        "Select month(s)",
        options=months,
        default=months
    )

    filtered = df[df["month"].isin(selected_months)].copy()

    if filtered.empty:
        st.warning("Select at least one month to display data.")
    else:
        st.write(f"Showing {len(filtered):,} observations")

        st.subheader("Sensor and Demand Trends")
        chart_data = (
            filtered.set_index("ts")[
                [
                    "soil_moisture",
                    "weather_temp",
                    "weather_humidity",
                    "fuzzy_demand_score"
                ]
            ]
            .sort_index()
        )
        st.line_chart(chart_data)

        st.subheader("Observations")
        display_columns = [
            col for col in [
                "ts",
                "soil_moisture",
                "weather_temp",
                "weather_humidity",
                "fuzzy_demand_score",
                "fuzzy_demand_level",
                "irrigation_duration_minutes"
            ]
            if col in filtered.columns
        ]
        st.dataframe(
            filtered[display_columns],
            use_container_width=True
        )

        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered data as CSV",
            data=csv_bytes,
            file_name="filtered_irrigation_data.csv",
            mime="text/csv"
        )


else:
    st.header("About the Model")

    st.markdown("""
    **Model type:** Mamdani-style fuzzy inference system

    **Inputs**
    - Soil moisture: dataset-relative scale, 0–80
    - Air temperature: 0–50 °C
    - Relative humidity: 0–100%

    **Output**
    - Fuzzy irrigation-demand score: 0–100
    - Low, Medium, and High categories for dashboard presentation

    **Validation completed**
    - 2,057 grid combinations tested without failures
    - 1,000 random input combinations tested without failures
    - Representative scenarios and moisture-demand trends checked
    """)

    st.warning(
        "The system has been tested for computational coverage and "
        "expected behavior. Its agronomic recommendations have not been "
        "independently validated against crop outcomes or optimal water use."
    )

st.sidebar.divider()
st.sidebar.caption("Intelligent Irrigation System | Soft Computing")
