import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import max_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder 

data=pd.read_csv('data_C02_emission.csv')
ohe=OneHotEncoder()
ohe_df=pd.DataFrame(ohe.fit_transform(data[['Fuel Type']]).toarray())
data=data.join(ohe_df)

data.columns=['Make','Model','Vehicle Class','Engine Size (L)','Cylinders','Transmission','Fuel Type','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)','CO2 Emissions (g/km)','Fuel0', 'Fuel1', 'Fuel2', 'Fuel3']
y=data['CO2 Emissions (g/km)'].copy()
X=data.drop('CO2 Emissions (g/km)',axis=1)
X_train_all, X_test_all, y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=1)

X_train=X_train_all[['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)','Fuel0', 'Fuel1', 'Fuel2', 'Fuel3']]
X_test=X_test_all[['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)','Fuel0', 'Fuel1', 'Fuel2', 'Fuel3']]

linearModel=LinearRegression()