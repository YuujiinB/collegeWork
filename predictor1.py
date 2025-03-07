import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Preprocess the data
def preprocess_data(data, commodity, time_steps=10):
    try:
        commodity_data = data[data['Commodity'] == commodity]
        commodity_data['Date'] = pd.to_datetime(commodity_data['Date'])
        commodity_data.set_index('Date', inplace=True)
        prices = commodity_data['Average'].values

        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))

        X, y = [], []
        for i in range(len(scaled_prices) - time_steps):
            X.append(scaled_prices[i:i + time_steps])  # Using sequences of 'time_steps'
            y.append(scaled_prices[i + time_steps])  # Predict the next price
        return np.array(X), np.array(y), scaler
    except KeyError:
        print(f"Error: The commodity '{commodity}' does not exist in the dataset.")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred during preprocessing: {e}")
        return None, None, None

# Build the CNN model
def build_model(time_steps=10):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(time_steps, 1)))  # Adjusted input shape
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Main function to execute the model
def main():
    file_path = 'vegetable_prices.csv'
    data = load_data(file_path)
    
    if data is not None:
        commodities = data['Commodity'].unique()
        print("Available commodities:")
        for idx, commodity in enumerate(commodities):
            print(f"{idx + 1}: {commodity}")

        try:
            choice = int(input("Select a commodity by number: ")) - 1
            selected_commodity = commodities[choice]
        except (ValueError, IndexError):
            print("Invalid selection. Please select a valid commodity number.")
            return

        X, y, scaler = preprocess_data(data, selected_commodity)
        if X is not None and y is not None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = build_model()
            early_stopping = EarlyStopping(monitor='loss', patience=5)
            model.fit(X_train, y_train, epochs=100, callbacks=[early_stopping], verbose=0)

            predictions = model.predict(X_test)
            predictions = scaler.inverse_transform(predictions)

            # Plotting the results
            plt.figure(figsize=(12, 6))
            plt.plot(data[data['Commodity'] == selected_commodity].index[-len(predictions):], predictions, label='Predicted Prices', color='orange')
            plt.plot(data[data['Commodity'] == selected_commodity].index[-len(predictions):], scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual Prices', color='blue')
            plt.title(f'Price Prediction for {selected_commodity}')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.show()
        else:
            print("Data preprocessing failed. Please check the commodity selection.")

if __name__ == "__main__":
    main()
