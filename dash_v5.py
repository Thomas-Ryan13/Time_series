# Import libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np
import pandas as pd
from datetime import date
import xgboost as xgb
import streamlit as st
import plotly.graph_objects as go
from math import sqrt
import random

def mse(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    differences = np.subract(actual, predicted)
    squared_differences = np.square(differences)
    return squared_differences.mean()

# Get today's date
date_run = str(pd.to_datetime("now").date())

# Set precision
pd.set_option("display.precision", 2)

# Create dash
st.set_page_config(layout="wide")
## Set dash title
st.title("European Energy Cost Forecast")
## Set dash description
st.subheader("The dashboard will help users plan for forecasted European energy prices.")
st.markdown("All data is sourced from (https://transparency.entsoe.eu).")
## Set title for dropdowns
st.sidebar.title("Select Country")

# Import data
df = pd.read_csv('EnergyCostAllCountries.csv', parse_dates= ['Date'], dayfirst= True)
df.sort_values(by = ['Date', 'Country'], inplace=True)

## Create country dropdown
country_list = ['Austria','Belgium','Croatia','Denmark','Estonia','Finland','Germany','Greece','Hungary','Italy','Latvia','Lithuania','Luxembourg','Netherlands','Norway','Portugal','Serbia','Slovenia','Spain','Sweden','Switzerland']
selected_status = st.sidebar.selectbox('Country',
                                       options = country_list)
country_list_range = range(0, len(country_list))
## Set country dropdown conditions
for country in country_list:
    if not selected_status == country:
        df=df
    elif selected_status == country:
        df = df[df['Country']==country]
        df2 = df[df['Country']==country]
        df2 = df2[['Date','Day-ahead Price (EUR/MWh)']]
        
# Get Country Name
country_name = df.iat[1,0]

# Display most recent Day-ahead Price (EUR/MWh)
most_recent_date = df['Date'].loc[df.index[len(df)-1]]
recent_prices_df = df[df['Date']== most_recent_date]
recent_prices_df = recent_prices_df.set_index('Country')
recent_prices_df['Date'] = recent_prices_df['Date'].astype(str).str[:10]

# Display historic and forecasted prices graph
## Smooth 2022 for training because of Russia-Ukraine war spike
df2['rolling average'] = df2['Day-ahead Price (EUR/MWh)'].transform(lambda x: x.rolling(180, 1).mean())
df2['rolling average sd'] = (df2['Day-ahead Price (EUR/MWh)'].transform(lambda x: x.rolling(180, 1).std()))*2
df2['diff from mean'] = abs(df2['Day-ahead Price (EUR/MWh)'] - df2['rolling average'])
df2['abs z-score'] = df2['diff from mean'] - df2['rolling average sd']
df2.fillna(0, inplace=True)
df2['spike_dummy'] = [1 if x > 0 else 0 for x in df2['abs z-score']]
model_df = df2[['Date','Day-ahead Price (EUR/MWh)','spike_dummy']]
model_df.sort_values(by = ['Date'], inplace=True)
## Set datetime as index
df2 = df2.set_index('Date')
model_df = model_df.set_index('Date')
## Test/Train split
np.random.seed(0)
msk = np.random.rand(len(model_df)) < 0.8
train = model_df[msk]
test = model_df[~msk]
## Define func to create features
def create_features(data):
    data = data.copy()
    data['dayofweek'] = data.index.dayofweek
    data['month'] = data.index.month
    data['year'] = data.index.year
    data['dayofyear'] = data.index.dayofyear
    return data
## Create season
def create_season(x):
    ret = 1
    if (x >= 3) and (x <= 5 ):
        ret = 2
    if (x >= 6) and (x <= 8):
        ret = 3
    if (x >= 9) and (x <= 11):
        ret = 4
    return ret
## Execute func on model_df
model_df = create_features(model_df)
model_df['season'] = model_df['month'].apply(create_season)
## Create features for train/test
train = create_features(train)
train['season'] = train['month'].apply(create_season)
test = create_features(test)
test['season'] = test['month'].apply(create_season)
## Define features/target variables
FEATURES = ['dayofweek', 'month', 'dayofyear', 'year', 'season', 'spike_dummy']
TARGET = 'Day-ahead Price (EUR/MWh)'
## Set x and y for model training
X_train = train[FEATURES]
y_train = train[TARGET]
X_test = test[FEATURES]
y_test = test[TARGET]
## Train Model
random.seed(0)
reg = xgb.XGBRegressor(n_estimators = 1000, early_stopping_rounds=50)
reg.fit(X_train, y_train,
        eval_set = [(X_train, y_train), (X_test, y_test)],
        verbose = True)

## Create forecasted predictions
df2['ninety_day_future'] = df2.index + pd.Timedelta('180D')
future = pd.DataFrame(index=df2['ninety_day_future'].iloc[-180:],)
future = create_features(future)
future['season'] = future['month'].apply(create_season)
future['spike_dummy'] = 0
future.index.names = ['Date']
X_future = future[FEATURES]
future['prediction'] = reg.predict(X_future)

# Create plot to show model performance over last 90 days
last30_df = model_df.iloc[-90:]
X_30_df = last30_df[FEATURES]
last30_df['Prediction'] = reg.predict(X_30_df)
last30_df = last30_df[['Day-ahead Price (EUR/MWh)', 'Prediction']]
fig2, ax2 = plt.subplots(figsize=(15,5))
last30_df.plot(ax=ax2, title = 'Forecasted VS Actual Prices Last 90 Days')

### RMSE
prediction = last30_df['Prediction']
actual = last30_df['Day-ahead Price (EUR/MWh)']
rmse = sqrt(mse(actual, prediction))
rmse = round(rmse, 2)

## Plot historic and forecasted prices together
historic_prices = pd.DataFrame(index = df2.index)
historic_prices['Day-ahead Price (EUR/MWh)'] = df2['Day-ahead Price (EUR/MWh)']
forecasted_prices = pd.DataFrame(index = future.index)
forecasted_prices['Day-ahead Price (EUR/MWh)'] = future['prediction']
historic_prices['lower bound'] = historic_prices['Day-ahead Price (EUR/MWh)']
historic_prices['upper bound'] = historic_prices['Day-ahead Price (EUR/MWh)']
historic_prices['forecasted'] = 0
forecasted_prices['lower bound'] = forecasted_prices['Day-ahead Price (EUR/MWh)'] - 2*rmse
forecasted_prices['upper bound'] = forecasted_prices['Day-ahead Price (EUR/MWh)'] + 2*rmse
forecasted_prices['forecasted'] = 1
df_list = [historic_prices, forecasted_prices]
merged_df = pd.concat(df_list)
fig, ax = plt.subplots(figsize=(15,5))
x = merged_df.index
ax.plot(x[-180:], merged_df['Day-ahead Price (EUR/MWh)'][-180:])
ax.plot(x[:-179], merged_df['Day-ahead Price (EUR/MWh)'][:-179])
ax.fill_between(
    x, merged_df['lower bound'], merged_df['upper bound'], color = 'lightblue', alpha=.65)
ax.set_title('Energy Prices Over Time')
fig.autofmt_xdate(rotation=45)
# Display reports in order
## Energy prices over time graph
st.header('Energy Prices Over Time')
st.markdown('Take a look at historic energy prices alongside our forecasted prices for the next 180 days.')
st.pyplot(fig)
## Most recent price table
st.header('Most Recent Price Table')
st.markdown('Stay informed with our most recent pricing data.')
st.dataframe(recent_prices_df)
## Downloadable csv
merged_df = merged_df[['Day-ahead Price (EUR/MWh)','lower bound','upper bound']]
def convert_data(data):
    return data.to_csv(index=True).encode('utf-8')
csv = convert_data(merged_df)
st.header('Download All Historic and Forecasted Price Data')
st.download_button("Download Full Dataset", csv, "Energy_Cost_Data_"+country_name+"_"+date_run+".csv","text/csv",key='download-csv')
## Model performance
### Plot
st.header('Check Model Performance last 90 days')
st.markdown("Take a look at how the model's predicted values have compared to the actual values over the last 90 days.")
st.pyplot(fig2)
### RMSE
rmse = str(rmse)
st.metric(label='Root Mean Squared Error', value = rmse)
### Downloadable Data
csv2 = convert_data(last30_df)
st.download_button('Download Model Performance Data', csv2, "Model_Performance_Data"+country_name+"_"+date_run+".csv", "text/csv", key="download-csv2")
