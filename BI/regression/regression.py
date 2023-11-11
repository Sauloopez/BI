import requests
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=42)

#create and adjust the poly
poly = PolynomialFeatures(degree=12)
x_train_poly = poly.fit_transform(X_train)
x_test_poly = poly.fit_transform(X_test)

# create and adjust the model of linear regression
model = LinearRegression()
model.fit(x_train_poly, y_train)



#predict the data group of train and test
y_train_pred = model.predict(x_train_poly)
y_test_pred = model.predict(x_test_poly)

#define the mean square error
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

#print(f"mse train: {mse_train} \nmse test: {mse_test}")
plt.plot(df['Ordinal_date'], df['t2m'], '-', color='Orange', label='Medium Temperature')
plt.plot(X_test, y_test, '*', color='Blue', label='Test')
plt.legend(loc='best')


plt.scatter(X_train, y_train, color='green', label='Datos de entrenamiento')
plt.scatter(X_train, y_train_pred, color='red', label='Predicciones de entrenamiento')
plt.title('Regresión Polinómica - Datos de Entrenamiento')
plt.xlabel('Ordinal_date')
plt.ylabel('t2m')
plt.legend()

# Graficar resultados de prueba
plt.scatter(X_test, y_test_pred, color='purple', label='Predicciones de prueba')
plt.title('Regresión Polinómica - Datos de Prueba')
plt.xlabel('Ordinal_date')
plt.ylabel('t2m')
plt.legend()
plt.show()