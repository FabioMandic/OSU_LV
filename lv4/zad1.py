import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score



cars = pd.read_csv('data_C02_emission.csv')

input_variables = ['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)']
output = 'CO2 Emissions (g/km)'

X = cars[input_variables]
y = cars[output]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=1)



plt.scatter(X_train['Engine Size (L)'], y_train, c='blue')
plt.scatter(X_test['Engine Size (L)'], y_test, c='red')
plt.xlabel('Engine Size (L)')
plt.ylabel('CO2 Emissions (g/km)')
plt.title('Emissions compared to engine size')
plt.show()

plt.hist(X_train['Engine Size (L)'])
plt.show()



sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

plt.figure()
x_train['Fuel Consumption City per (L/100km)'].plot(kind = 'hist' , bins = 20 )
plt.show()

plt.figure()
x_train['Fuel Consumption City per (L/100km)'].plot(kind = 'hist' , bins = 20 )
plt.show()

linearModel = lm.LinearRegression()
linearModel.fit(x_train_n, y_train)
print(linearModel.coef_)

y_test_p = linearModel.predict(x_test_n)
plt.scatter(y_test, y_test_p)
plt.show()



MAE = mean_absolute_error(y_test , y_test_p)
MAPE = mean_absolute_percentage_error(y_test , y_test_p)
MSE = mean_squared_error(y_test , y_test_p)
RMSE = root_mean_squared_error(y_test , y_test_p)
R2 = r2_score(y_test , y_test_p)

print(MAE, MAPE, MSE, RMSE, R2)