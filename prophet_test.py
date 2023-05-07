from prophet import Prophet

from data import read_pm2
from utils import local_names

import matplotlib.pyplot as plt

for local in local_names:
    print(local, 'doing...')
    df = read_pm2('dataset/TRAIN', local)

    df_train = df[:-24*31]
    df_test  = df[-24*31:]
    print(df_test.head())

    m = Prophet()
    m.add_country_holidays(country_name='KR')
    m.add_seasonality(name='weekly', period=7, fourier_order=3)
    m.add_seasonality(name='daily', period=1, fourier_order=3)
    m.add_seasonality(name='yearly', period=365.25, fourier_order=3)
    m.add_seasonality(name='monthly', period=365.25/12, fourier_order=3)
    m.add_seasonality(name='quarterly', period=365.25/4, fourier_order=3)

    m.fit(df_train)

    future_dates = m.make_future_dataframe(periods=24*31, freq='H')
    predictions = m.predict(future_dates).tail(24*31)['yhat']

    # plot predictions as real time series
    plt.plot(df_test['ds'], df_test['y'], label='real', color='red')
    plt.plot(df_test['ds'], predictions, label='pred', color='blue')

    plt.show()

    break


