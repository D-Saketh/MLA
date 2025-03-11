import numpy as np  # Numerical computations
import pandas as pd  # Data handling
import yfinance as yf  # Fetch stock data
import matplotlib.pyplot as plt  # Visualization
from sklearn.preprocessing import MinMaxScaler  # Normalization
from sklearn.metrics import mean_squared_error, r2_score  # Performance metrics
from tensorflow.keras.models import Sequential  # Neural network
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional  # LSTM layers
from tensorflow.keras.callbacks import EarlyStopping  # Stop early if no improvement

# Step 1: Fetch Stock Data
stock_symbol = "AAPL"  # Stock ticker
start_date = "2015-01-01"
end_date = "2024-01-01"
df = yf.download(stock_symbol, start=start_date, end=end_date)

# Step 2: Data Preprocessing
# Keep only closing price for training
df = df[['Close']]
scaler = MinMaxScaler(feature_range=(0,1))  # Normalize between 0 and 1
df_scaled = scaler.fit_transform(df)  # Apply scaling

# Step 3: Create Sliding Window Sequences
def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step])  # Previous 60 days as input
        y.append(data[i+time_step])  # Next day's price as output
    return np.array(X), np.array(y)

# Define sequence length
time_step = 60
X, y = create_sequences(df_scaled, time_step)

# Step 4: Split into Training and Testing Sets
train_size = int(len(X) * 0.8)  # 80% training, 20% testing
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for LSTM input (samples, time steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Step 5: Build the LSTM Model
model = Sequential([
    Bidirectional(LSTM(100, return_sequences=True, input_shape=(time_step, 1))),  # Bidirectional LSTM
    Dropout(0.2),  # Prevent overfitting
    LSTM(50, return_sequences=False),  # Standard LSTM
    Dropout(0.2),
    Dense(25, activation="relu"),  # Fully connected layer
    Dense(1)  # Output single predicted stock price
])

# Step 6: Compile and Train Model
model.compile(optimizer='adam', loss='mean_squared_error')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Stop if no improvement

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

# Step 7: Make Predictions
y_pred = model.predict(X_test)  # Predict stock prices
y_pred = scaler.inverse_transform(y_pred)  # Convert predictions back to original scale
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))  # Convert test data back

# Step 8: Evaluate Performance
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))  # RMSE
r2 = r2_score(y_test_actual, y_pred)  # R² Score
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Step 9: Plot Results
plt.figure(figsize=(10,5))
plt.plot(y_test_actual, label="Actual Prices", color="blue")
plt.plot(y_pred, label="Predicted Prices", color="red")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.title(f"LSTM Stock Price Prediction for {stock_symbol}")
plt.show()
