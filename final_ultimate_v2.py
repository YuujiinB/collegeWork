import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Define image size and batch size
img_size = (128, 128)
batch_size = 32

# Create data generators for train, test, and validation
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load generators without class labels (we'll use CSV for prices)
train_generator = train_datagen.flow_from_directory(
    'Vegetable Images/train',
    target_size=img_size,
    batch_size=batch_size,
    class_mode=None,  # No labels
    shuffle=False  # Maintain order to match filenames
)

test_generator = test_datagen.flow_from_directory(
    'Vegetable Images/test',
    target_size=img_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=False
)

val_generator = val_datagen.flow_from_directory(
    'Vegetable Images/validation',
    target_size=img_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=False
)

# Load price data
df = pd.read_csv('vegetable_prices.csv')

# Function to extract prices from CSV using filenames and vegetable types
# Inside the extract_prices() function:
def extract_prices(generator):
    prices = []
    for filename in generator.filenames:
        parts = filename.split('/')
        
        if len(parts) < 2:
            image_name = parts[0]
            # Replace "ImagePath" with your CSV column name (e.g., "FileName")
            matched_row = df[df['FileName'] == image_name]  # <-- CHANGE HERE
        else:
            vegetable, image_name = parts[0], parts[1]
            # Replace "ImagePath" and "Commodity" with your CSV column names
            matched_row = df[
                (df['VegetableType'] == vegetable) &  # <-- CHANGE HERE (if needed)
                (df['FileName'] == image_name)        # <-- CHANGE HERE
            ]
        
        if not matched_row.empty:
            price = matched_row['Price'].values[0]
            prices.append(price)
        else:
            prices.append(np.nan)
    return np.array(prices)

# Extract prices for all sets
train_prices = extract_prices(train_generator)
test_prices = extract_prices(test_generator)
val_prices = extract_prices(val_generator)

# Handle missing prices (remove NaN entries)
train_mask = ~np.isnan(train_prices)
train_features_clean = train_features[train_mask]
train_prices_clean = train_prices[train_mask]

# CNN for feature extraction
def create_cnn(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5)
    ])
    return model

# Create and compile the CNN model
cnn_input_shape = (128, 128, 3)
cnn_model = create_cnn(cnn_input_shape)
cnn_model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')

# Extract features from images using the CNN
def extract_features(generator, model):
    features = []
    generator.reset()
    for i in range(len(generator)):
        images = generator[i]
        batch_features = model.predict(images)
        features.extend(batch_features)
    return np.array(features)

# Extract features
train_features = extract_features(train_generator, cnn_model)
test_features = extract_features(test_generator, cnn_model)
val_features = extract_features(val_generator, cnn_model)

# Train Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(train_features_clean, train_prices_clean)

# Predict prices
val_predictions = linear_reg.predict(val_features)
test_predictions = linear_reg.predict(test_features)

# Create DataFrame with filenames, actual prices, and predictions
def create_results_df(generator, actual_prices, predictions):
    return pd.DataFrame({
        'Filename': generator.filenames,
        'Actual Price': actual_prices,
        'Predicted Price': predictions
    })

# Generate results for validation and test sets
val_results_df = create_results_df(val_generator, val_prices, val_predictions)
test_results_df = create_results_df(test_generator, test_prices, test_predictions)

# Display results (excluding entries with missing prices)
print("\nValidation Set Predictions:")
print(val_results_df.dropna())

print("\nTest Set Predictions:")
print(test_results_df.dropna())

# Evaluate metrics (on non-NaN entries)
def evaluate_predictions(y_true, y_pred):
    mask = ~np.isnan(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = 100 * np.mean(np.abs((y_true - y_pred) / y_true))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "MAPE": mape}

# Validation set evaluation
val_results = evaluate_predictions(val_prices, val_predictions)
print("\nValidation Metrics:")
for metric, value in val_results.items():
    print(f"{metric}: {value:.4f}")

# Test set evaluation
test_results = evaluate_predictions(test_prices, test_predictions)
print("\nTest Metrics:")
for metric, value in test_results.items():
    print(f"{metric}: {value:.4f}")

# Plot actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(val_results_df['Actual Price'].dropna(), val_results_df['Predicted Price'].dropna(), alpha=0.5)
plt.plot([0, max(val_results_df['Actual Price'])], [0, max(val_results_df['Actual Price'])], color='red', linestyle='--')
plt.title("Actual vs Predicted Prices (Validation Set)")
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()