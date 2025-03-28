import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')

# 1.a

print(len(data))#ukupan broj redova
print(data.info())#broj redova i kolona
print(data.isnull().sum())#provjera nedostajucih vrijednosti
data.dropna(axis=0)#uklanjanje s nedostajucim podacima
categories = data.select_dtypes(include=['object']).columns #odabire sve tekstualne kolone
for category in categories: #prolazi kroz sve kolone
    data[category] = data[category].astype('category')#pretvara u kategorijske podatke
print(data.info())

# 1.b

fuel_consumption = data.sort_values(by='Fuel Consumption City (L/100km)', ascending = False)
print("Najveci potrosaci iz tablice: ")
print(fuel_consumption[["Make", "Model", "Fuel Consumption City (L/100km)"]].head(3))
print("Najmanji potrosaci iz tablice:")
print(fuel_consumption[["Make", "Model", "Fuel Consumption City (L/100km)"]].tail(3))

# 1.c

engine_filter = data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]
print(len(engine_filter))
avg = engine_filter['CO2 Emissions (g/km)'].mean()
print(avg)

# 1.d

print('\n')
audi_cars = data[data.Make == "Audi"]
print("Broj mjerenja se odnosi na vozila proizvodaca Audi:", len(audi_cars))
audi_cars_with_four_cylinders = audi_cars[audi_cars['Cylinders'] == 4]
print("Prosjecna emisija C02 plinova automobila proizvodaca Audi koji imaju 4 cilindara:", audi_cars_with_four_cylinders['CO2 Emissions (g/km)'].mean())

# 1.e

print('\n')
grouped_cars_cylinders = data.groupby('Cylinders')
print(grouped_cars_cylinders.size())
print("Prosjecna emisija C02 plinova s obzirom na broj cilindara:\n", grouped_cars_cylinders['CO2 Emissions (g/km)'].mean())

# 1.f

mean = data.groupby('Fuel Type')['CO2 Emissions (g/km)'].mean()
median = data.groupby('Fuel Type')['CO2 Emissions (g/km)'].median()
print(mean)
print(median)

# 1.g

four_cylinders_diesel = data[(data['Cylinders'] == 4) & (data['Fuel Type'] == 'D')]
max_value = four_cylinders_diesel['CO2 Emissions (g/km)'].max()
max_car = four_cylinders_diesel[four_cylinders_diesel['Fuel Consumption City (L/100km)'] == max_value]
max_car[['Make','Model','Cylinders','Fuel Type','CO2 Emissions (g/km)']]

# 1.h

manual_cars = data[data['Transmission'].str.startswith('M')]
print(len(manual_cars))

# 1.i

print(data.corr(numeric_only=True))

