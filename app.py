
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import csv
import random
from datetime import datetime

# TensorFlow is required for LSTM prediction
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Energy AI",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# CONSTANTS AND FILE PATHS
# ============================================================

DATA_FOLDER = "data"
MODEL_FOLDER = "models"

ENERGY_FILE = os.path.join(DATA_FOLDER, "energy_data.csv")
HISTORICAL_FILE = os.path.join(DATA_FOLDER, "historical_energy.csv")
CUSTOM_APPLIANCES_FILE = os.path.join(
    DATA_FOLDER,
    "custom_appliances.json"
)

MODEL_FILE = os.path.join(
    MODEL_FOLDER,
    "energy_lstm_model.keras"
)

SCALER_FILE = os.path.join(
    MODEL_FOLDER,
    "energy_scaler.pkl"
)

POWER_LIMIT = 3500
ELECTRICITY_RATE = 8.0
SEQUENCE_LENGTH = 7

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)


# ============================================================
# DEFAULT APPLIANCES
# ============================================================

DEFAULT_APPLIANCES = {
    "Home Mode": {
        "Refrigerator": 150,
        "Washing Machine": 500,
        "Fan": 75,
        "Light": 20,
        "Television": 120,
        "Heater": 2000,
        "Oven": 1800,
        "Microwave": 1200,
        "Electric Iron": 1000,
        "Water Pump": 750,
        "Laptop": 65,
        "Mixer Grinder": 500
    },

    "College Mode": {
        "Light": 20,
        "Fan": 75,
        "Projector": 300,
        "Computer": 200,
        "Printer": 400,
        "Air Conditioner": 1500,
        "Water Dispenser": 500
    }
}


# ============================================================
# APPLIANCE CUSTOMIZATION FUNCTIONS
# ============================================================

def load_custom_appliances():
    """
    Load customized appliances from JSON.
    If the file does not exist, use default appliances.
    """

    if not os.path.exists(CUSTOM_APPLIANCES_FILE):
        return {
            mode: dict(appliances)
            for mode, appliances in DEFAULT_APPLIANCES.items()
        }

    try:
        with open(
            CUSTOM_APPLIANCES_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            saved_data = json.load(file)

        all_appliances = {}

        for mode, defaults in DEFAULT_APPLIANCES.items():

            mode_data = saved_data.get(mode, defaults)

            cleaned_data = {}

            for name, power in mode_data.items():

                try:
                    power = float(power)

                    if str(name).strip() and power > 0:
                        cleaned_data[str(name)] = power

                except (ValueError, TypeError):
                    continue

            if not cleaned_data:
                cleaned_data = dict(defaults)

            all_appliances[mode] = cleaned_data

        return all_appliances

    except (json.JSONDecodeError, OSError, TypeError):
        return {
            mode: dict(appliances)
            for mode, appliances in DEFAULT_APPLIANCES.items()
        }


def save_custom_appliances(all_appliances):
    """
    Save customized appliances to JSON.
    """

    with open(
        CUSTOM_APPLIANCES_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_appliances,
            file,
            indent=4
        )


# ============================================================
# ENERGY CALCULATION FUNCTIONS
# ============================================================

def calculate_energy(power_watts, hours):
    """
    Energy = Power in watts × Time in hours ÷ 1000
    """

    return (power_watts * hours) / 1000


def calculate_cost(energy_kwh, rate):
    """
    Cost = Energy in kWh × Electricity rate
    """

    return energy_kwh * rate


# ============================================================
# DATA STORAGE FUNCTIONS
# ============================================================

def save_energy_record(
    mode,
    selected_appliances,
    total_power,
    total_energy,
    total_cost
):
    """
    Save energy usage to CSV.
    """

    file_exists = os.path.exists(ENERGY_FILE)

    record = {
        "date_time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "mode": mode,
        "appliances": ", ".join(selected_appliances),
        "total_power_w": round(total_power, 2),
        "total_energy_kwh": round(total_energy, 3),
        "cost_inr": round(total_cost, 2)
    }

    fieldnames = [
        "date_time",
        "mode",
        "appliances",
        "total_power_w",
        "total_energy_kwh",
        "cost_inr"
    ]

    with open(
        ENERGY_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        if not file_exists or os.path.getsize(ENERGY_FILE) == 0:
            writer.writeheader()

        writer.writerow(record)


def load_energy_records():
    """
    Load saved energy records.
    """

    if not os.path.exists(ENERGY_FILE):
        return pd.DataFrame()

    try:
        data = pd.read_csv(ENERGY_FILE)

        required_columns = [
            "date_time",
            "mode",
            "appliances",
            "total_power_w",
            "total_energy_kwh",
            "cost_inr"
        ]

        if not all(
            column in data.columns
            for column in required_columns
        ):
            return pd.DataFrame()

        return data

    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def load_historical_data():
    """
    Load historical energy dataset.
    """

    if not os.path.exists(HISTORICAL_FILE):
        return pd.DataFrame()

    try:
        data = pd.read_csv(HISTORICAL_FILE)

        required_columns = [
            "date",
            "total_energy_kwh",
            "cost_inr"
        ]

        if not all(
            column in data.columns
            for column in required_columns
        ):
            return pd.DataFrame()

        return data

    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


# ============================================================
# REAL-TIME MONITORING FUNCTIONS
# ============================================================

def simulate_sensor_readings(appliances):
    """
    Simulate ON/OFF readings for appliances.
    """

    return {
        appliance: random.choice([True, False])
        for appliance in appliances
    }


def calculate_live_power(sensor_status, appliances):
    """
    Calculate power consumption based on ON appliances.
    """

    total_power = 0

    for appliance, is_on in sensor_status.items():

        if is_on:
            total_power += appliances.get(appliance, 0)

    return total_power


# ============================================================
# LSTM PREDICTION FUNCTION
# ============================================================

def get_lstm_prediction(historical_data):
    """
    Predict the next day's energy usage using the trained LSTM.
    """

    if not TENSORFLOW_AVAILABLE:
        return None, None, "TensorFlow is not installed."

    if not os.path.exists(MODEL_FILE):
        return None, None, "LSTM model file not found."

    if not os.path.exists(SCALER_FILE):
        return None, None, "Scaler file not found."

    if len(historical_data) < SEQUENCE_LENGTH:
        return (
            None,
            None,
            "At least 7 historical records are required."
        )

    try:
        import pickle

        model = load_model(MODEL_FILE)

        with open(SCALER_FILE, "rb") as file:
            scaler = pickle.load(file)

        energy_values = historical_data[
            "total_energy_kwh"
        ].values.reshape(-1, 1)

        scaled_values = scaler.transform(energy_values)

        last_sequence = scaled_values[
            -SEQUENCE_LENGTH:
        ]

        last_sequence = last_sequence.reshape(
            1,
            SEQUENCE_LENGTH,
            1
        )

        prediction_scaled = model.predict(
            last_sequence,
            verbose=0
        )

        prediction = scaler.inverse_transform(
            prediction_scaled
        )

        predicted_energy = max(
            0,
            float(prediction[0][0])
        )

        predicted_cost = calculate_cost(
            predicted_energy,
            ELECTRICITY_RATE
        )

        return (
            predicted_energy,
            predicted_cost,
            None
        )

    except Exception as error:
        return None, None, str(error)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚡ Smart Energy AI")

mode = st.sidebar.selectbox(
    "Select Mode",
    [
        "Home Mode",
        "College Mode"
    ]
)

all_appliances = load_custom_appliances()

current_appliances = all_appliances.get(
    mode,
    DEFAULT_APPLIANCES[mode]
)

page = st.sidebar.radio(
    "Select Page",
    [
        "📊 Dashboard",
        "⚡ Real-Time Monitoring",
        "⚠️ Live Overload Alert",
        "💰 Cost Calculator",
        "🤖 Deep Learning Prediction",
        "📈 Analytics",
        "💡 Smart Recommendations",
        "⚙️ Customize Appliances"
    ]
)

st.sidebar.divider()

st.sidebar.write("Current Mode:")
st.sidebar.info(mode)

st.sidebar.write("Power Limit:")
st.sidebar.warning(f"{POWER_LIMIT} W")


# ============================================================
# CUSTOMIZE APPLIANCES
# ============================================================

if page == "⚙️ Customize Appliances":

    st.title("⚙️ Customize Appliances")

    st.write(
        f"Customize the appliances available in **{mode}**."
    )

    st.info(
        "You can add new appliances, edit power ratings, "
        "or remove appliances you do not use."
    )

    st.subheader("Current Appliances")

    if current_appliances:

        appliance_table = pd.DataFrame(
            [
                {
                    "Appliance": name,
                    "Power (Watts)": power
                }
                for name, power in current_appliances.items()
            ]
        )

        st.dataframe(
            appliance_table,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning("No appliances available.")

    st.divider()

    # --------------------------------------------------------
    # ADD APPLIANCE
    # --------------------------------------------------------

    st.subheader("➕ Add Appliance")

    with st.form("add_appliance_form"):

        new_name = st.text_input(
            "Appliance Name",
            placeholder="Example: Laptop"
        )

        new_power = st.number_input(
            "Power Consumption (Watts)",
            min_value=1.0,
            max_value=10000.0,
            value=100.0,
            step=10.0
        )

        add_button = st.form_submit_button(
            "Add Appliance"
        )

    if add_button:

        clean_name = new_name.strip()

        existing_names = [
            name.lower()
            for name in current_appliances
        ]

        if not clean_name:
            st.error("Please enter an appliance name.")

        elif clean_name.lower() in existing_names:
            st.error("This appliance already exists.")

        else:

            all_appliances[mode][clean_name] = new_power

            save_custom_appliances(all_appliances)

            st.success(
                f"{clean_name} was added to {mode}."
            )

            st.rerun()

    st.divider()

    # --------------------------------------------------------
    # EDIT APPLIANCE
    # --------------------------------------------------------

    st.subheader("✏️ Edit Appliance")

    if current_appliances:

        selected_appliance = st.selectbox(
            "Select Appliance",
            list(current_appliances.keys()),
            key="edit_appliance"
        )

        edited_power = st.number_input(
            "Updated Power (Watts)",
            min_value=1.0,
            max_value=10000.0,
            value=float(
                current_appliances[selected_appliance]
            ),
            step=10.0,
            key="edited_power"
        )

        if st.button("Update Appliance"):

            all_appliances[mode][
                selected_appliance
            ] = edited_power

            save_custom_appliances(all_appliances)

            st.success(
                f"{selected_appliance} was updated."
            )

            st.rerun()

    st.divider()

    # --------------------------------------------------------
    # DELETE APPLIANCE
    # --------------------------------------------------------

    st.subheader("🗑️ Delete Appliance")

    if current_appliances:

        appliance_to_delete = st.selectbox(
            "Select Appliance to Delete",
            list(current_appliances.keys()),
            key="delete_appliance"
        )

        if st.button("Delete Appliance"):

            del all_appliances[mode][
                appliance_to_delete
            ]

            save_custom_appliances(all_appliances)

            st.success(
                f"{appliance_to_delete} was deleted."
            )

            st.rerun()


# ============================================================
# DASHBOARD
# ============================================================

elif page == "📊 Dashboard":

    st.title("📊 Smart Energy Dashboard")

    st.write(
        f"Monitor energy usage in **{mode}**."
    )

    energy_records = load_energy_records()

    if not energy_records.empty:

        mode_records = energy_records[
            energy_records["mode"] == mode
        ].copy()

        if not mode_records.empty:

            today = datetime.now().strftime("%Y-%m-%d")

            mode_records["date_time"] = pd.to_datetime(
                mode_records["date_time"],
                errors="coerce"
            )

            today_records = mode_records[
                mode_records["date_time"].dt.strftime(
                    "%Y-%m-%d"
                ) == today
            ]

            current_power = float(
                mode_records["total_power_w"].iloc[-1]
            )

            today_energy = float(
                today_records["total_energy_kwh"].sum()
            )

            today_cost = float(
                today_records["cost_inr"].sum()
            )

        else:

            current_power = 0
            today_energy = 0
            today_cost = 0

    else:

        current_power = 0
        today_energy = 0
        today_cost = 0

    historical_data = load_historical_data()

    predicted_energy = 0
    predicted_cost = 0

    if not historical_data.empty:

        prediction, cost, error = get_lstm_prediction(
            historical_data
        )

        if prediction is not None:
            predicted_energy = prediction * 30
            predicted_cost = cost * 30

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Current Power",
        f"{current_power:.0f} W"
    )

    col2.metric(
        "Today's Energy",
        f"{today_energy:.2f} kWh"
    )

    col3.metric(
        "Today's Cost",
        f"₹{today_cost:.2f}"
    )

    col4.metric(
        "Monthly Predicted Energy",
        f"{predicted_energy:.2f} kWh"
    )

    col5.metric(
        "Monthly Predicted Bill",
        f"₹{predicted_cost:.2f}"
    )

    st.divider()

    st.subheader("Available Appliances")

    appliance_table = pd.DataFrame(
        [
            {
                "Appliance": name,
                "Power": f"{power:.0f} W"
            }
            for name, power in current_appliances.items()
        ]
    )

    st.dataframe(
        appliance_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# REAL-TIME MONITORING
# ============================================================

elif page == "⚡ Real-Time Monitoring":

    st.title("⚡ Real-Time Appliance Monitoring")

    st.write(
        f"Simulated monitoring for **{mode}**."
    )

    st.info(
        "This is simulated sensor data. "
        "Real hardware sensors can be connected later."
    )

    if st.button("Refresh Sensor Readings"):

        sensor_status = simulate_sensor_readings(
            current_appliances
        )

        st.session_state["sensor_status"] = sensor_status

    if "sensor_status" not in st.session_state:

        st.session_state["sensor_status"] = {
            appliance: False
            for appliance in current_appliances
        }

    sensor_status = st.session_state["sensor_status"]

    total_live_power = calculate_live_power(
        sensor_status,
        current_appliances
    )

    st.metric(
        "Current Simulated Power",
        f"{total_live_power:.0f} W"
    )

    if total_live_power > POWER_LIMIT:
        st.error(
            f"⚠️ Overload Warning: "
            f"{total_live_power:.0f} W exceeds "
            f"{POWER_LIMIT} W."
        )
    else:
        st.success("Power consumption is within the limit.")

    st.subheader("Appliance Status")

    status_rows = []

    for appliance in current_appliances:

        is_on = sensor_status.get(appliance, False)

        status_rows.append(
            {
                "Appliance": appliance,
                "Power": f"{current_appliances[appliance]:.0f} W",
                "Status": "ON" if is_on else "OFF"
            }
        )

    st.dataframe(
        pd.DataFrame(status_rows),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# LIVE OVERLOAD ALERT
# ============================================================

elif page == "⚠️ Live Overload Alert":

    st.title("⚠️ Live Overload Alert")

    st.write(
        "Select the appliances that are currently switched ON."
    )

    selected_on_appliances = []

    for appliance, power in current_appliances.items():

        is_selected = st.checkbox(
            f"{appliance} ({power:.0f} W)",
            key=f"overload_{mode}_{appliance}"
        )

        if is_selected:
            selected_on_appliances.append(appliance)

    total_power = sum(
        current_appliances[appliance]
        for appliance in selected_on_appliances
    )

    st.divider()

    st.metric(
        "Total Power Consumption",
        f"{total_power:.0f} W"
    )

    st.metric(
        "Maximum Safe Power",
        f"{POWER_LIMIT} W"
    )

    if total_power > POWER_LIMIT:

        st.error(
            "🚨 HIGH POWER USAGE! "
            "The selected appliances exceed the power limit."
        )

    else:

        remaining_power = POWER_LIMIT - total_power

        st.success(
            "✅ Power consumption is within the safe limit."
        )

        st.info(
            f"Remaining available power: "
            f"{remaining_power:.0f} W"
        )

    st.subheader("Selected Appliances")

    if selected_on_appliances:

        for appliance in selected_on_appliances:
            st.write(
                f"• {appliance}: "
                f"{current_appliances[appliance]:.0f} W"
            )

    else:
        st.write("No appliances are currently selected.")


# ============================================================
# COST CALCULATOR
# ============================================================

elif page == "💰 Cost Calculator":

    st.title("💰 Energy Cost Calculator")

    st.write(
        f"Calculate energy consumption and cost for **{mode}**."
    )

    selected_appliances = st.multiselect(
        "Select Appliances",
        options=list(current_appliances.keys())
    )

    rate = st.number_input(
        "Electricity Rate (₹ per kWh)",
        min_value=0.0,
        value=ELECTRICITY_RATE,
        step=0.5
    )

    appliance_hours = {}

    total_power = 0
    total_energy = 0

    if selected_appliances:

        st.subheader("Usage Duration")

        for appliance in selected_appliances:

            power = current_appliances[appliance]

            hours = st.number_input(
                f"{appliance} usage (hours)",
                min_value=0.0,
                max_value=24.0,
                value=1.0,
                step=0.5,
                key=f"hours_{mode}_{appliance}"
            )

            appliance_hours[appliance] = hours

            total_power += power

            total_energy += calculate_energy(
                power,
                hours
            )

        total_cost = calculate_cost(
            total_energy,
            rate
        )

        st.divider()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Combined Power",
            f"{total_power:.0f} W"
        )

        col2.metric(
            "Total Energy",
            f"{total_energy:.2f} kWh"
        )

        col3.metric(
            "Estimated Cost",
            f"₹{total_cost:.2f}"
        )

        if total_power > POWER_LIMIT:

            st.error(
                f"⚠️ Combined power is above "
                f"the {POWER_LIMIT} W limit."
            )

        else:

            st.success(
                "Power consumption is within the limit."
            )

        if st.button("Save Energy Record"):

            save_energy_record(
                mode,
                selected_appliances,
                total_power,
                total_energy,
                total_cost
            )

            st.success(
                "Energy record saved successfully!"
            )

    else:

        st.info(
            "Select one or more appliances to calculate cost."
        )


# ============================================================
# DEEP LEARNING PREDICTION
# ============================================================

elif page == "🤖 Deep Learning Prediction":

    st.title("🤖 Deep Learning Energy Prediction")

    st.write(
        "Predict future energy consumption using an LSTM model."
    )

    historical_data = load_historical_data()

    if historical_data.empty:

        st.warning(
            "Historical dataset not found. "
            "Create data/historical_energy.csv first."
        )

    else:

        st.subheader("Historical Dataset")

        st.dataframe(
            historical_data.tail(10),
            use_container_width=True,
            hide_index=True
        )

        if st.button("Predict Next Day Energy"):

            prediction, cost, error = get_lstm_prediction(
                historical_data
            )

            if error:

                st.error(error)

            else:

                col1, col2 = st.columns(2)

                col1.metric(
                    "Predicted Next Day Energy",
                    f"{prediction:.2f} kWh"
                )

                col2.metric(
                    "Predicted Next Day Cost",
                    f"₹{cost:.2f}"
                )

                st.success(
                    "Prediction completed successfully."
                )

        st.divider()

        st.subheader("Monthly Estimate")

        prediction, cost, error = get_lstm_prediction(
            historical_data
        )

        if error:

            st.info(
                "Monthly prediction unavailable: "
                + error
            )

        else:

            monthly_energy = prediction * 30
            monthly_cost = cost * 30

            col1, col2 = st.columns(2)

            col1.metric(
                "Estimated Monthly Energy",
                f"{monthly_energy:.2f} kWh"
            )

            col2.metric(
                "Estimated Monthly Bill",
                f"₹{monthly_cost:.2f}"
            )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📈 Analytics":

    st.title("📈 Energy Analytics")

    historical_data = load_historical_data()
    energy_records = load_energy_records()

    if not historical_data.empty:

        st.subheader("Historical Energy Consumption")

        historical_data["date"] = pd.to_datetime(
            historical_data["date"],
            errors="coerce"
        )

        historical_data = historical_data.dropna(
            subset=["date"]
        )

        st.line_chart(
            historical_data.set_index("date")[
                "total_energy_kwh"
            ]
        )

        st.subheader("Historical Electricity Cost")

        st.line_chart(
            historical_data.set_index("date")[
                "cost_inr"
            ]
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Energy",
            f"{historical_data['total_energy_kwh'].sum():.2f} kWh"
        )

        col2.metric(
            "Average Daily Energy",
            f"{historical_data['total_energy_kwh'].mean():.2f} kWh"
        )

        col3.metric(
            "Total Cost",
            f"₹{historical_data['cost_inr'].sum():.2f}"
        )

    else:

        st.warning(
            "Historical data is not available."
        )

    st.divider()

    st.subheader("Saved Energy Records")

    if not energy_records.empty:

        mode_records = energy_records[
            energy_records["mode"] == mode
        ]

        if not mode_records.empty:

            st.dataframe(
                mode_records,
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Saved Energy Usage")

            st.bar_chart(
                mode_records.set_index("date_time")[
                    "total_energy_kwh"
                ]
            )

            st.subheader("Most Recent Appliance Usage")

            latest_record = mode_records.iloc[-1]

            st.write(
                latest_record["appliances"]
            )

        else:

            st.info(
                f"No saved records for {mode}."
            )

    else:

        st.info(
            "No energy records available yet. "
            "Save a record from the Cost Calculator."
        )


# ============================================================
# SMART RECOMMENDATIONS
# ============================================================

elif page == "💡 Smart Recommendations":

    st.title("💡 Smart Energy Recommendations")

    st.write(
        f"Recommendations for **{mode}**."
    )

    total_available_power = sum(
        current_appliances.values()
    )

    st.subheader("Appliance Power Overview")

    st.metric(
        "Total Registered Appliance Power",
        f"{total_available_power:.0f} W"
    )

    if total_available_power > POWER_LIMIT:

        st.warning(
            "Your registered appliances have a combined "
            "power rating above the safe limit. "
            "Avoid operating all appliances together."
        )

    else:

        st.success(
            "Your registered appliances are within "
            "the configured power limit when combined."
        )

    st.divider()

    st.subheader("Recommendations")

    recommendations = [
        "Switch off lights and fans when they are not required.",
        "Avoid operating several high-power appliances simultaneously.",
        "Use energy-efficient LED lights.",
        "Run the washing machine with a full load.",
        "Turn off projectors and computers after use.",
        "Check high-power appliances regularly.",
        "Track daily energy consumption to reduce electricity cost."
    ]

    for recommendation in recommendations:

        st.write(
            f"💡 {recommendation}"
        )

    st.divider()

    st.subheader("High-Power Appliances")

    high_power_appliances = {
        name: power
        for name, power in current_appliances.items()
        if power >= 1000
    }

    if high_power_appliances:

        high_power_table = pd.DataFrame(
            [
                {
                    "Appliance": name,
                    "Power": f"{power:.0f} W"
                }
                for name, power in high_power_appliances.items()
            ]
        )

        st.dataframe(
            high_power_table,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "Use high-power appliances separately "
            "to reduce overload risk."
        )

    else:

        st.success(
            "No registered appliance has a power rating "
            "of 1000 W or more."
        )