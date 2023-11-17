import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from .models import Data


class Prediction:
    t2m=[]
    dates=[]

    def __init__(self) -> None:
        self.load_data()
        self.data['dates'] = self.data['dates'].apply(lambda str:datetime.strptime(str, "%Y%m%d"))
        self.df = pd.DataFrame({'Date':self.data['dates'], 't2m': self.data['t2m']})
        print(self.data)
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

    def get_predictions():
        prediction = Prediction()
        df = prediction.get_dataframe()
        prediction.add_df_column('Ordinal_date', df['Date'].apply(lambda date : date.toordinal()))
        df = prediction.get_dataframe()
        x = df[['Ordinal_date']]
        y = df[['t2m']]

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=42)

        poly = PolynomialFeatures(degree=9)
        x_train_poly = poly.fit_transform(X_train)
        x_test_poly = poly.fit_transform(X_test)
        
        svm_model_best = SVR(C=9000, epsilon=0.01)

        model = make_pipeline(StandardScaler(), svm_model_best)
        model.fit(x_train_poly, y_train)


        y_train_pred = model.predict(x_train_poly)
        y_test_pred = model.predict(x_test_poly)


        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)

        last_date = df['Date'].iloc[-1]

        act = datetime.strptime("20240401", "%Y%m%d")
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
        return res
    pass