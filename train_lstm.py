
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# ---------------------------------------
# SETTINGS
# ---------------------------------------

DATA_FILE = "data/historical_energy.csv"

MODEL_FOLDER = "models"

MODEL_FILE = os.path.join(
    MODEL_FOLDER,
    "energy_lstm_model.keras"
)

SCALER_FILE = os.path.join(
    MODEL_FOLDER,
    "energy_scaler.pkl"
)

SEQUENCE_LENGTH = 7

EPOCHS = 30

BATCH_SIZE = 8

ELECTRICITY_RATE = 8.0


# ---------------------------------------
# CREATE MODEL FOLDER
# ---------------------------------------

if not os.path.exists(MODEL_FOLDER):

    os.makedirs(MODEL_FOLDER)


# ---------------------------------------
# LOAD DATASET
# ---------------------------------------

print("Loading historical dataset...")

if not os.path.exists(DATA_FILE):

    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

data = pd.read_csv(DATA_FILE)

print("\nDataset columns:")

print(
    data.columns.tolist()
)

print("\nFirst five rows:")

print(
    data.head()
)


# ---------------------------------------
# VALIDATE DATASET
# ---------------------------------------

required_columns = [
    "date",
    "total_energy_kwh"
]

for column in required_columns:

    if column not in data.columns:

        raise ValueError(
            f"Required column missing: {column}"
        )


# ---------------------------------------
# PREPARE DATA
# ---------------------------------------

data["date"] = pd.to_datetime(
    data["date"],
    errors="coerce"
)

data["total_energy_kwh"] = pd.to_numeric(
    data["total_energy_kwh"],
    errors="coerce"
)

data = data.dropna(
    subset=[
        "date",
        "total_energy_kwh"
    ]
)

data = data.sort_values(
    by="date"
)

energy_values = data[
    "total_energy_kwh"
].values.reshape(-1, 1)


print(
    "\nTotal valid records:",
    len(energy_values)
)


# ---------------------------------------
# CHECK DATA SIZE
# ---------------------------------------

if len(energy_values) <= SEQUENCE_LENGTH:

    raise ValueError(
        "Not enough data for LSTM training. "
        "At least 8 valid records are required."
    )


# ---------------------------------------
# NORMALIZE DATA
# ---------------------------------------

print("\nNormalizing energy data...")

scaler = MinMaxScaler()

scaled_energy = scaler.fit_transform(
    energy_values
)


# ---------------------------------------
# SAVE SCALER
# ---------------------------------------

with open(
    SCALER_FILE,
    "wb"
) as file:

    pickle.dump(
        scaler,
        file
    )

print(
    "Scaler saved successfully:",
    SCALER_FILE
)


# ---------------------------------------
# CREATE LSTM SEQUENCES
# ---------------------------------------

print("\nCreating training sequences...")

X = []

y = []

for i in range(
    SEQUENCE_LENGTH,
    len(scaled_energy)
):

    sequence = scaled_energy[
        i - SEQUENCE_LENGTH:i
    ]

    target = scaled_energy[i]

    X.append(sequence)

    y.append(target)


X = np.array(X)

y = np.array(y)


print(
    "\nInput shape:",
    X.shape
)

print(
    "Target shape:",
    y.shape
)


# ---------------------------------------
# TRAIN AND TEST SPLIT
# ---------------------------------------

split_index = int(
    len(X) * 0.8
)

X_train = X[
    :split_index
]

X_test = X[
    split_index:
]

y_train = y[
    :split_index
]

y_test = y[
    split_index:
]


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ---------------------------------------
# BUILD LSTM MODEL
# ---------------------------------------

print("\nBuilding LSTM model...")

model = Sequential(
    [

        tf.keras.Input(
            shape=(
                SEQUENCE_LENGTH,
                1
            )
        ),

        LSTM(
            64,
            return_sequences=True
        ),

        Dropout(
            0.2
        ),

        LSTM(
            32
        ),

        Dropout(
            0.2
        ),

        Dense(
            16,
            activation="relu"
        ),

        Dense(
            1
        )

    ]
)


# ---------------------------------------
# COMPILE MODEL
# ---------------------------------------

model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=[
        "mae"
    ]
)


print(
    "\nModel summary:"
)

model.summary()


# ---------------------------------------
# TRAIN MODEL
# ---------------------------------------

print(
    "\nStarting LSTM training..."
)

history = model.fit(

    X_train,

    y_train,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    validation_data=(
        X_test,
        y_test
    ),

    verbose=1

)


# ---------------------------------------
# EVALUATE MODEL
# ---------------------------------------

print(
    "\nEvaluating model..."
)

test_loss, test_mae = model.evaluate(

    X_test,

    y_test,

    verbose=0

)


print(
    f"Test Loss: {test_loss:.6f}"
)

print(
    f"Test MAE: {test_mae:.6f}"
)


# ---------------------------------------
# PREDICT NEXT DAY ENERGY
# ---------------------------------------

print(
    "\nPredicting next-day energy..."
)

last_sequence = scaled_energy[
    -SEQUENCE_LENGTH:
]

last_sequence = last_sequence.reshape(

    1,

    SEQUENCE_LENGTH,

    1

)


predicted_scaled = model.predict(

    last_sequence,

    verbose=0

)


predicted_energy = scaler.inverse_transform(

    predicted_scaled

)


predicted_energy_value = float(

    predicted_energy[0][0]

)


# Prevent negative energy predictions

predicted_energy_value = max(

    0,

    predicted_energy_value

)


# ---------------------------------------
# CALCULATE PREDICTED COST
# ---------------------------------------

predicted_cost = (

    predicted_energy_value
    * ELECTRICITY_RATE

)


# ---------------------------------------
# SAVE LSTM MODEL
# ---------------------------------------

model.save(

    MODEL_FILE

)


# ---------------------------------------
# DISPLAY FINAL RESULTS
# ---------------------------------------

print(
    "\n---------------------------------------"
)

print(
    "LSTM TRAINING COMPLETED"
)

print(
    "---------------------------------------"
)

print(

    f"Predicted next-day energy: "
    f"{predicted_energy_value:.2f} kWh"

)

print(

    f"Predicted next-day electricity cost: "
    f"₹{predicted_cost:.2f}"

)

print(

    f"Model saved at: "
    f"{MODEL_FILE}"

)

print(

    f"Scaler saved at: "
    f"{SCALER_FILE}"

)

print(
    "---------------------------------------"
)