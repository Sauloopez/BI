import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from .models import Data

# google's tensorflow imports
from tensorflow.keras.models import Sequential
from  tensorflow.keras.layers import LSTM, Dense



class Prediction:
    t2m=[]
    dates=[]

    def __init__(self) -> None:
        self.load_data()
        self.data['dates'] = self.data['dates'].apply(lambda str:datetime.strptime(str, "%Y%m%d"))
        self.df = pd.DataFrame({'Date':self.data['dates'], 't2m': self.data['t2m']})
        pass

    def load_data(self):
        data_loaded = Data.get_data()
        t2m_values = data_loaded.get('properties', {}).get('parameter', {}).get('T2M', {})
        self.data = pd.DataFrame({
            'dates':t2m_values.keys(),
            't2m':t2m_values.values()
        })
        pass

    def set_dataframe(self, df):
        self.df = df
        pass

    def add_df_column(self, column_name, column_data):
        self.df[column_name]=column_data
        pass

    def get_dataframe(self):
        return self.df

    @staticmethod
    def get_lstm_predictions():
        prediction = Prediction()
        df = prediction.get_dataframe()
        prediction.add_df_column('Ordinal_date', df['Date'].apply(lambda date: date.toordinal()))
        df = prediction.get_dataframe()

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(df[['t2m', 'Ordinal_date']])

        sequence_length = 30
        X, y = [], []
        for i in range(sequence_length, len(data_scaled)):
            X.append(data_scaled[i-sequence_length: i, :])
            y.append(data_scaled[i, 0])

        X, y = np.array(X), np.array(y)
        split = int(.8 * len(X))

        x_train, x_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = Sequential()
        model.add(
            LSTM(
                units=50,
                return_sequences=True,
                input_shape=(x_train.shape[1], x_train.shape[2])
            )
        )
        model.add(LSTM(50))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=32, validation_data=(x_test, y_test))

        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        y_train_pred = scaler.inverse_transform(np.concatenate([y_train_pred, x_train[:, -1, 1:]], axis=1))[:, 0]
        y_test_pred = scaler.inverse_transform(np.concatenate([y_test_pred, x_test[:, -1, 1:]], axis=1))[:, 0]
        y_train = scaler.inverse_transform(np.concatenate([y_train.reshape(-1, 1), x_train[:, -1, 1:]], axis=1))[:, 0]
        y_test = scaler.inverse_transform(np.concatenate([y_test.reshape(-1, 1), x_test[:, -1, 1:]], axis=1))[:, 0]

        act = datetime.strptime("20250130", "%Y%m%d")

        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)

        last_date = df['Date'].iloc[-1]
        prediction_days = (act - last_date).days

        future_predictions = []
        input_seq = x_test[-1].copy()

        for _ in range(prediction_days):
            next_pred = model.predict(input_seq[np.newaxis, :, :])[0, 0]
            future_predictions.append(next_pred)

            new_date_ordinal = input_seq[-1, 1] + 1
            input_seq = np.roll(input_seq, -1, axis=0)
            input_seq[-1] = [next_pred, new_date_ordinal]

        # Escalar las nuevas fechas
        future_dates = [last_date + timedelta(days=i + 1) for i in range(prediction_days)]
        future_dates_ordinal = [date.toordinal() for date in future_dates]
        future_predictions_scaled = scaler.inverse_transform(
            np.concatenate([np.array(future_predictions).reshape(-1, 1), np.array(future_dates_ordinal).reshape(-1, 1)], axis=1)
        )[:, 0]

        # Predecir con las nuevas fechas escaladas
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Prediction': future_predictions_scaled
        })
        """
        train_df = pd.DataFrame({
            'Ordinal date': x_train,
            'Prediction': y_train,
        })

        test_df = pd.DataFrame({
            'Ordinal date': x_test,
            'Prediction': y_test,
        })


        predict_df = pd.concat([train_df, test_df])
        """
        #res = pd.merge(df, predict_df, on='Ordinal_date', how='outer')
        res = pd.concat([df, future_df], ignore_index=True)
        return res, mse_train, mse_test

    @staticmethod
    def get_predictions():
        prediction = Prediction()
        df = prediction.get_dataframe()
        prediction.add_df_column('Ordinal_date', df['Date'].apply(lambda date : date.toordinal()))
        df = prediction.get_dataframe()
        x = df[['Ordinal_date']]
        y = df[['t2m']]

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

        poly = PolynomialFeatures(degree=9)
        x_train_poly = poly.fit_transform(X_train, y_train)
        x_test_poly = poly.fit_transform(X_test, y_test)

        svm_model_best = SVR(C=9000, epsilon=0.01)

        model = make_pipeline(StandardScaler(), svm_model_best)
        model.fit(x_train_poly, y_train)


        y_train_pred = model.predict(x_train_poly)
        y_test_pred = model.predict(x_test_poly)


        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)

        last_date = df['Date'].iloc[-1]

        act = datetime.strptime("20250130", "%Y%m%d")
        new_dates = []

        while last_date <= act:
            new_dates.append(last_date.toordinal())
            last_date += timedelta(days=1)

        new_predictions= model.predict(poly.transform(np.array(new_dates).reshape(-1, 1)))


        X_train['Prediction']=y_train_pred
        X_test['Prediction']=y_test_pred
        predict_df = pd.concat([X_train, X_test])

        new_dates_df = pd.DataFrame({'Date': [datetime.fromordinal(d) for d in new_dates],
                                    'Ordinal_date':new_dates,
                                    'Prediction': new_predictions})

        res = pd.merge(df, predict_df, on='Ordinal_date', how='outer')
        res = pd.concat([res, new_dates_df], ignore_index=True)
        return res, mse_train, mse_test
    pass
