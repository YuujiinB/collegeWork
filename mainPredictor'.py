import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
df = pd.read_csv("vegetable_prices.csv", parse_dates=["Date"], index_col=["Date"])

# Display the first few rows of the dataset
print(df.head())

# Get a list of all unique vegetables in the dataset
unique_commodities = df['Commodity'].unique()

# List vegetables with numbers for selection
print("Select a vegetable by entering its corresponding number:")
for i, veg in enumerate(unique_commodities, start=1):
    print(f"{i}. {veg}")

# Ask user to select a vegetable by number
while True:
    try:
        selected_number = int(input("Enter the number of the vegetable: "))
        if 1 <= selected_number <= len(unique_commodities):
            selected_commodity = unique_commodities[selected_number - 1]
            break
        else:
            print("Invalid number. Please try again.")
    except ValueError:
        print("Invalid input. Please enter a number.")

print(f"You selected: {selected_commodity}")

# Filter dataset by the selected commodity
df_selected = df[df['Commodity'] == selected_commodity]

# Check if the filtered DataFrame is empty
if df_selected.empty:
    print(f"No data found for {selected_commodity}. Please check the dataset.")
    exit()

# Drop unnecessary column
df_selected = df_selected.drop(['Commodity'], axis=1)

# Handle missing values (if any)
df_selected = df_selected.dropna()

# Display first few rows of the selected vegetable's data
print(df_selected.head())

# Resample data to quarterly averages
df_selected_quarterly = df_selected.resample('Q').mean().dropna()

# Get the date range for the selected vegetable
first_date = df_selected_quarterly.index.min().strftime('%Y-%m-%d')
last_date = df_selected_quarterly.index.max().strftime('%Y-%m-%d')

# Combine the dates into a single string
date_range = f"{first_date} to {last_date}"
print(f"Data range: {date_range}")

# Plot the data (Actual Prices vs Date)
plt.figure(figsize=(10, 6))
plt.plot(df_selected_quarterly.index, df_selected_quarterly['Average'], label='Average Price')
plt.title(f"Quarterly Price of {selected_commodity} from {date_range}", fontsize=16)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Convert the data to a list for training
time = df_selected_quarterly.index.to_numpy()
series = df_selected_quarterly['Average'].to_numpy()

# Normalize the data to [0, 1] range
min_val = np.min(series)
max_val = np.max(series)
series_normalized = (series - min_val) / (max_val - min_val)

# Define the split time
split_ratio = 0.8  # 80% for the training set
split_time = int(len(series_normalized) * split_ratio)

# Get the train set
time_train = time[:split_time]
x_train_norm = series_normalized[:split_time]

# Get the validation set
time_valid = time[split_time:]
x_valid_norm = series_normalized[split_time:]

# Plot the train/test split
plt.figure(figsize=(15, 5))
plt.plot(time_train, x_train_norm, color='orange', label='Train Data')
plt.plot(time_valid, x_valid_norm, color='purple', label='Test Data')
plt.xlabel('Date')
plt.ylabel('Normalized Price')
plt.title(f"Train-Test Split for {selected_commodity}")
plt.legend()
plt.show()

# Parameters
window_size = 4  # 4 quarters = 1 year
batch_size = 8
shuffle_buffer_size = 1000

# Prepare dataset for TensorFlow
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Generate the dataset windows
train_dataset = windowed_dataset(x_train_norm, window_size, batch_size, shuffle_buffer_size)
val_dataset = windowed_dataset(x_valid_norm, window_size, batch_size, shuffle_buffer_size)

# Build the model (Enhanced Dense Model with Dropout)
model = Sequential([
    Dense(64, activation='relu', input_shape=[window_size], kernel_regularizer='l2'),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer='l2'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

# Learning rate scheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.96)

# Early stopping callback
early_stopping = EarlyStopping(patience=20, restore_best_weights=True)

# Compile the model
model.compile(loss=Huber(delta=0.5), optimizer=Adam(learning_rate=lr_schedule), metrics=['mae'])

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=200,
    callbacks=[early_stopping]
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Use the model to make predictions
def model_forecast(model, series, window_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(window_size))
    dataset = dataset.batch(batch_size).prefetch(1)
    forecast = model.predict(dataset)
    return forecast

# Generate predictions
forecast_series = series_normalized[split_time - window_size:]
forecast = model_forecast(model, forecast_series, window_size, batch_size)

# Drop single dimensional axis
results = forecast.squeeze()

# Denormalize predictions
results_denorm = results * (max_val - min_val) + min_val
x_valid_denorm = x_valid_norm * (max_val - min_val) + min_val

# Ensure predictions and validation data have the same length
results_denorm = results_denorm[:len(x_valid_denorm)]

# Plot Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.plot(time_valid, x_valid_denorm, label='Actual Price', color='blue')
plt.plot(time_valid, results_denorm, label='Predicted Price', color='red')
plt.title(f"Predicted vs Actual Prices for {selected_commodity}", fontsize=16)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Seasonal Analysis: Compare quarters
df_selected_quarterly['Quarter'] = df_selected_quarterly.index.quarter
df_selected_quarterly['Year'] = df_selected_quarterly.index.year

# Historical quarterly averages
historical_quarterly_avg = df_selected_quarterly.groupby('Quarter')['Average'].mean()

# Predicted quarterly averages
predicted_quarterly = pd.DataFrame({
    'Date': time_valid,
    'Predicted': results_denorm
})
predicted_quarterly['Quarter'] = predicted_quarterly['Date'].dt.quarter
predicted_quarterly_avg = predicted_quarterly.groupby('Quarter')['Predicted'].mean()

# Plot Historical vs Predicted Quarterly Averages
plt.figure(figsize=(10, 6))
plt.bar(historical_quarterly_avg.index - 0.2, historical_quarterly_avg, width=0.4, label='Historical Average')
plt.bar(predicted_quarterly_avg.index + 0.2, predicted_quarterly_avg, width=0.4, label='Predicted Average')
plt.title(f"Quarterly Price Comparison for {selected_commodity}", fontsize=16)
plt.xlabel('Quarter')
plt.ylabel('Price')
plt.xticks([1, 2, 3, 4], ['Q1', 'Q2', 'Q3', 'Q4'])
plt.legend()
plt.show()

# Evaluate model performance
def evaluate_preds(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = tf.cast(y_pred, dtype=y_true.dtype)  # Ensure matching data types
    
    # Calculate MAE
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # Calculate MSE
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Calculate RMSE
    rmse = tf.sqrt(mse)
    
    # Calculate MAPE
    mape = 100 * tf.reduce_mean(tf.abs((y_true - y_pred) / y_true))
    
    return {"mae": mae.numpy(), "mse": mse.numpy(), "rmse": rmse.numpy(), "mape": mape.numpy()}

results_eval = evaluate_preds(x_valid_denorm, results_denorm)
print(f"Evaluation results: {results_eval}")