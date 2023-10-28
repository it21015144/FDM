import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

# import pmdarima as pm

st.title("Solar Radiation Prediction App")
@st.cache_data
def load_data(city):
    Data = pd.read_csv(f"{city}.csv")
    return Data


st.sidebar.header("Enter Input Data:")
efficiency = st.sidebar.number_input("Efficiency : ",min_value = 0.0,max_value = 1.0,help = 'Enter values as a percentage Ex : 0.15 = 15%')
city_options = ["Colombo","Mount-Lavinia","Kesbewa","Maharagama","Kandy","Negombo","Sri-Jayewardenepura-kotte","Kalmunai","Trincomalee","Galle","Jaffna","Athurugiriya","Weligama","Matara","Kolonnawa","Gampaha","Puttalam","Badulla","Kalutara","Bentota","Matale","Mannar","Pothuhera","Kurunegala","Mabole","Hatton","Hambantota","Oruwala"]  # Replace with your list of city names
city = st.sidebar.selectbox("City:", city_options)
covered_area = st.sidebar.number_input("Covered Area (Square meters):",min_value = 0.0)
process_button = st.sidebar.button("Process Data")


if process_button:
    user_input = {
        'efficiency': efficiency,
        'city': city,
        'covered_area': covered_area
    }
    Data = load_data(city) 

    if user_input['efficiency'] is not None and user_input['city'] is not None and user_input['covered_area'] is not None:

      st.write(city)
      
      def missing_values(df,percentage):
        columns=df.columns
        percent_missing=df.isnull().sum()*100/len(df)
        missing_value_df=pd.DataFrame({'column_name':columns,'percentage_missing':percent_missing})

        #print missing value df
        drop_column=list(missing_value_df[missing_value_df.percentage_missing > percentage].column_name)
        df=df.drop(drop_column,axis=1)
        return df

      Data=missing_values(Data,40)

      # Convert the 'time' column to datetime
      Data['time'] = pd.to_datetime(Data['time'])

      # Sort the DataFrame by the 'time' column
      # Data = Data.sort_values(by='time')

      # Extract the sorted date and solar radiation columns
      date = Data['time']
      solar_radiation = Data['shortwave_radiation_sum']

      #Preprocessing========================================================================================================

      #get the null value count in each column
      print(Data.isna().sum())

      #Replace null values with Nan
      Data=Data.replace('',np.NaN)


      #Drop missing values rows
      Data.dropna(inplace=True)

      #Number of duplicate rows
      Data.duplicated().sum()

      #drop duplicates if there's any
      Data.drop_duplicates()

      #drop columns
      Data=Data.drop(['precipitation_sum', 'rain_sum', 'snowfall_sum', 'precipitation_hours', 'windspeed_10m_max', 'windgusts_10m_max', 'winddirection_10m_dominant', 'et0_fao_evapotranspiration', 'latitude', 'longitude','country','temperature_2m_mean','apparent_temperature_max','apparent_temperature_min','apparent_temperature_mean','city','temperature_2m_max','temperature_2m_min','sunrise','sunset','elevation'],axis='columns')


      time_series = Data['time']  # Replace 'timestamp_column' with the name of your timestamp column
      values = Data['shortwave_radiation_sum']

      train_proportion = 0.8 # 80% for training, 20% for testing

      # Calculate the index to split the data
      split_index = int(len(Data) * train_proportion)

      # Split the dataset into training and testing sets
      train_data = Data.iloc[:split_index]
      test_data = Data.iloc[split_index:]

      #Find Best Order using auto_Arima

      # model = pm.auto_arima(Data['shortwave_radiation_sum'], seasonal=True, stepwise=True, suppress_warnings=True)
      # best_order = model.get_params()['order']
      # print("Optimal order (p, d, q):", best_order)


      # Fit the ARIMA model to the training data
      # Use the same ARIMA model order you've determined earlier
      model = sm.tsa.arima.ARIMA(train_data['shortwave_radiation_sum'], order=(2, 0, 1))
      model_fit = model.fit()


      # Generate forecasts for the testing set
      forecast_horizon = len(test_data)
      forecast= model_fit.forecast(steps=forecast_horizon)

      # Calculate Mean Absolute Percentage Error (MAPE)
      actual_values = test_data['shortwave_radiation_sum']

      def mean_absolute_percentage_error(y_true, y_pred):
          return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

      mape = mean_absolute_percentage_error(actual_values, forecast)

      # print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
      print(f"mean_absolute_percentage_error : {mape:.2f}%")

      model = sm.tsa.arima.ARIMA(values, order=(2, 0, 1))
      model_fit = model.fit()

      # Summary of the model
      print(model_fit.summary())
      # Generate forecasts for the next 30 days
      forecast_horizon = 30
      forecast = model_fit.forecast(steps=forecast_horizon)

      last_observation_date = time_series.iloc[-1]
      forecast_dates = pd.date_range(start=last_observation_date, periods=forecast_horizon+1)

      forecasted_values = forecast[1:]
      # # Create a DataFrame with the forecasted values and date index
      values = []
      power = []

      for i in forecast:
        values.append(i)
        power.append(i*float(efficiency)*(covered_area)/11)

      forecast_df = pd.DataFrame({'Forecasted_Values': values,'KW/H' : power}, index=forecast_dates[1:])

      plt.figure(figsize=(12, 6))
      plt.plot(forecast_df.index, forecast_df['KW/H'], label='Solar Radiation', color='red')
      plt.title('Solar power genaration Over Time')
      plt.xlabel('Date')
      plt.ylabel('Solar power genaration (kWh)')

      st.pyplot(plt)

      st.write(forecast_df)


