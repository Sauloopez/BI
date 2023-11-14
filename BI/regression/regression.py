import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# url = 'http://127.0.0.1:8000/nasa-data'
url = 'https://bi-class.co/nasa-data'

response = requests.get(url, verify=False)

data_pages = response.json()
data=[]

for page in data_pages:
    for row in page:
        data.append(row)
#Obtains the values for each object from the request
t2m, t2max, t2min, date = zip(*[(i['t2m'], i['t2max'], i['t2min'], datetime.strptime(i['date'], "%Y%m%d")) for i in data])
#create the dataframe with the data proporcionate from the request
df = pd.DataFrame({'Date': date, 't2m': t2m, 't2max':t2max, 't2min':t2min})
#add a column with the ordinal date
df['Ordinal_date']= df['Date'].apply(lambda date : date.toordinal())



x= df[['Ordinal_date']]
y=df[['t2m']]

#rate the data in test and train
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=42)

#create and adjust the poly
poly = PolynomialFeatures(degree=9)
x_train_poly = poly.fit_transform(X_train)
x_test_poly = poly.fit_transform(X_test)


# Definir los hiperparámetros a ajustar
param_grid = {'C': [0.1, 1, 10, 100, 200, 500, 600, 800, 9000, 1500],
              'epsilon': [0.001, 0.01, 0.1, 1]}

# Crear el modelo SVR
svm_model = SVR()

# Configurar la búsqueda de cuadrícula
grid_search = GridSearchCV(svm_model, param_grid, cv=5)
grid_search.fit(X_train, y_train.values.ravel())

# Imprimir los mejores hiperparámetros
best_params = grid_search.best_params_
print("Mejores hiperparámetros:", best_params)

# Crear un nuevo modelo SVM con los mejores hiperparámetros
svm_model_best = SVR(C=best_params['C'], epsilon=best_params['epsilon'])

#model = RandomForestRegressvm_model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))sor(n_estimators=1500, max_depth=8,random_state=42)
model = make_pipeline(StandardScaler(), svm_model_best)
model.fit(x_train_poly, y_train)

#predict the data group of train and test
y_train_pred = model.predict(x_train_poly)
y_test_pred = model.predict(x_test_poly)


#define the mean square error
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)


# def predictTemperature(date):
#     date = pd.to_datetime(date)
#     ordinal_date = date.toordinal()
#     features = np.array([[ordinal_date]])
#     poly_features = poly.transform(features)
#     prediction = model.predict(poly_features)
#     plt.plot(features, prediction, '--', label=f'Prediccion: X:{features} Y: {prediction[0]}')

last_date = datetime.strptime("20230701", "%Y%m%d")

# Fecha actual
act = datetime.strptime("20241110", "%Y%m%d")

# Lista para almacenar las fechas generadas
new_dates = []

# Genera las fechas en un bucle hasta la fecha actual
while last_date <= act:
    new_dates.append(last_date.toordinal())
    last_date += timedelta(days=1)

new_predictions= model.predict(poly.transform(np.array(new_dates).reshape(-1, 1)))
plt.plot(new_dates, new_predictions, '-', label=f'Prediccion')


print(f"mse train: {mse_train} \nmse test: {mse_test}")
#print(f"Prediction: {predictTemperature(input('Input the date for predict '))}")
plt.plot(df['Ordinal_date'], df['t2m'], '--', color='Orange', label='Medium Temperature')
# plt.plot(X_test, y_test, '*', color='Blue', label='Test')
plt.legend(loc='best')


#plt.scatter(X_train, y_train, color='green', label='Datos de entrenamiento')
plt.scatter(X_train, y_train_pred, color='red', label='Predicciones de entrenamiento')
plt.title('Regresión Polinómica - Datos de Entrenamiento')
plt.xlabel('Ordinal_date')
plt.ylabel('t2m')
plt.legend()

# Graficar resultados de prueba
plt.scatter(X_test, y_test_pred,color='purple', label='Predicciones de prueba')
plt.title('Regresión Polinómica - Datos de Prueba')
plt.xlabel('Ordinal_date')
plt.ylabel('t2m')
plt.legend()
plt.show()