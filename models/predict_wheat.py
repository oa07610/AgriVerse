import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

###############################
# Step 1: Load and Preprocess Wheat Data
###############################

# Load the CSV file (adjust the file path if needed)
df = pd.read_csv('new_data/WheatMaster.csv', parse_dates=['DATE'], dayfirst=True)

# Display initial rows and information
print("First 5 rows:")
print(df.head())
print("\nDataFrame Info:")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())

# Extract date features
df['year'] = df['DATE'].dt.year
df['month'] = df['DATE'].dt.month
df['day'] = df['DATE'].dt.day
df['dayofweek'] = df['DATE'].dt.dayofweek

# Clean numeric price columns.
# Note: The CSV headers are "minimum", "maximum" (lowercase) and "AVERAGE" (uppercase)
for col in ['minimum', 'maximum', 'AVERAGE']:
    df[col] = df[col].astype(str).str.replace(',', '').astype(float)

# Optionally convert ID columns if needed
if (df['DISTRICT ID'].dropna() % 1 == 0).all():
    df['DISTRICT ID'] = df['DISTRICT ID'].apply(lambda x: int(x) if pd.notnull(x) else x)
if (df['STATION ID'].dropna() % 1 == 0).all():
    df['STATION ID'] = df['STATION ID'].apply(lambda x: int(x) if pd.notnull(x) else x)

###############################
# Step 2: Merge External Factors
###############################
# Load Crude Oil Data
crude_df = pd.read_csv('uef/crude.csv')
crude_df['Date'] = pd.to_datetime(crude_df['Date'], format='%m/%d/%Y')
crude_df.rename(columns={'Price': 'crude_oil'}, inplace=True)
crude_df = crude_df.sort_values('Date')

# Load Petrol Prices Data
petrol_df = pd.read_csv('uef/petrol.csv')
petrol_df['Date'] = pd.to_datetime(petrol_df['Date'], format='%m/%d/%Y')
petrol_df.rename(columns={'Petrol Price (PKR)': 'petrol_price'}, inplace=True)
petrol_df = petrol_df.sort_values('Date')

# Load USD-PKR Data
usd_df = pd.read_csv('uef/usd.csv')
usd_df['Date'] = pd.to_datetime(usd_df['Date'], format='%m/%d/%Y')
usd_df.rename(columns={'Price': 'usd_pkr'}, inplace=True)
usd_df = usd_df.sort_values('Date')

# Merge Crude Oil data with wheat data
df = pd.merge_asof(df.sort_values('DATE'), crude_df[['Date', 'crude_oil']],
                   left_on='DATE', right_on='Date', direction='backward')
df.drop(columns=['Date'], inplace=True)

# Merge Petrol Prices data
df = pd.merge_asof(df.sort_values('DATE'), petrol_df[['Date', 'petrol_price']],
                   left_on='DATE', right_on='Date', direction='backward')
df.drop(columns=['Date'], inplace=True)

# Merge USD-PKR data
usd_df.rename(columns={'Date': 'usd_date'}, inplace=True)
df = pd.merge_asof(df.sort_values('DATE'), usd_df[['usd_date', 'usd_pkr']],
                   left_on='DATE', right_on='usd_date', direction='backward')
df.drop(columns=['usd_date'], inplace=True)

print("\nMerged DataFrame columns:")
print(df.columns)

###############################
# Step 3: Convert Data Types
###############################
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
df['month'] = pd.to_numeric(df['month'], errors='coerce').astype('Int64')
df['day'] = pd.to_numeric(df['day'], errors='coerce').astype('Int64')
df['dayofweek'] = pd.to_numeric(df['dayofweek'], errors='coerce').astype('Int64')

df['crude_oil'] = pd.to_numeric(df['crude_oil'], errors='coerce')
df['petrol_price'] = pd.to_numeric(df['petrol_price'], errors='coerce')
df['usd_pkr'] = pd.to_numeric(df['usd_pkr'], errors='coerce')

###############################
# Step 4: Filter for Stations with > 500 Data Points
###############################
# Use the lowercase "station" column (as in the CSV)
station_counts = df.groupby('station').size()
stations_to_keep = station_counts[station_counts > 500].index
df_filtered_500 = df[df['station'].isin(stations_to_keep)].copy()

print("Number of stations after filtering:", df_filtered_500['station'].nunique())

###############################
# Step 5: Run LSTM Model and Generate Predictions (Full Forecast)
###############################

# Helper function to create sliding window dataset
def create_dataset(dataset, look_back=7, target_index=0):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back])
        Y.append(dataset[i+look_back, target_index])  # target is "AVERAGE"
    return np.array(X), np.array(Y)

look_back = 7
epochs = 20
batch_size = 16
lag_intervals = [50]  # We'll use a 50-day lag
results = {}

# Get unique stations from the filtered dataset
stations = df_filtered_500['station'].unique()
predictions_list = []

# Set forecast horizon (e.g., next 180 days). Adjust as needed.
forecast_steps = 180

for lag in lag_intervals:
    print(f"\nTesting with USD-PKR lag = {lag} day(s)")
    for station in stations:
        station_df = df_filtered_500[df_filtered_500['station'] == station].copy()
        station_df = station_df.sort_values('DATE').reset_index(drop=True)

        # Only process stations with at least 500 data points
        if len(station_df) < 500:
            continue

        # Create the lag feature for USD-PKR
        station_df[f'usd_pkr_lag'] = station_df['usd_pkr'].shift(lag)
        station_df.dropna(subset=[f'usd_pkr_lag'], inplace=True)

        # Use three features: 'AVERAGE' (target), 'usd_pkr', and the lag feature 'usd_pkr_lag'
        features = ['AVERAGE', 'usd_pkr', f'usd_pkr_lag']
        data = station_df[features].values

        # Scale each feature individually
        scalers = {}
        scaled_data = np.zeros_like(data, dtype=float)
        for i, feat in enumerate(features):
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data[:, i] = scaler.fit_transform(data[:, i].reshape(-1, 1)).flatten()
            scalers[feat] = scaler

        # Create dataset using a sliding window
        X_data, Y_target = create_dataset(scaled_data, look_back=look_back, target_index=0)
        if len(X_data) < 10:
            continue

        # Use all available data for training
        train_size = len(X_data)
        X_train = X_data[:train_size]
        Y_train = Y_target[:train_size]
        X_train = np.reshape(X_train, (X_train.shape[0], look_back, 3))

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, input_shape=(look_back, 3)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])

        # Forecast for the next forecast_steps days using a recursive approach
        last_window = scaled_data[-look_back:]  # Starting window
        forecast_scaled = []
        current_window = last_window.copy()
        for _ in range(forecast_steps):
            current_window_reshaped = current_window.reshape(1, look_back, 3)
            pred_scaled = model.predict(current_window_reshaped)
            forecast_scaled.append(pred_scaled[0, 0])
            # Update the window: shift one day and append the new prediction for the target feature
            new_row = current_window[-1].copy()
            new_row[0] = pred_scaled
            current_window = np.vstack([current_window[1:], new_row])

        forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
        forecast_inverted = scalers['AVERAGE'].inverse_transform(forecast_scaled)

        # Create forecast dates starting from the day after the last available date
        last_date = station_df['DATE'].max()
        forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')

        station_pred = pd.DataFrame({
            'station': station,
            'date': forecast_dates,
            'predicted_average': forecast_inverted.flatten()
        })
        predictions_list.append(station_pred)
        print(f"Station: {station} processed.")

if predictions_list:
    all_predictions = pd.concat(predictions_list, ignore_index=True)
    all_predictions.to_csv('data/final_wheat_predictions.csv', index=False)
    print("Predictions generated and saved to data/final_wheat_predictions.csv")
else:
    print("No predictions generated.")
