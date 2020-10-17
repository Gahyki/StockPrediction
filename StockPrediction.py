import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import os
from sklearn.preprocessing import MinMaxScaler

# Read csv file (df for dataframe)
df = pd.read_csv('KO.csv', usecols=['Date', 'Open', 'High', 'Low', 'Close'])

# Displaying cvs data
# Initialize size of graph
plt.figure(figsize=(18, 9))

# Average price of day
plt.plot(range(df.shape[0]), (df['Low']+df['High'])/2.0)

# X axis of every 14 days
plt.xticks(range(0, df.shape[0], 14), df['Date'].loc[::14], rotation=45)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Mid Price', fontsize=18)
plt.show()

# Training set and test set
high = np.array(df.loc[:, 'High'])
low = np.array(df.loc[:, 'Low'])
mid = (high+low)/2

train_data = mid[:200]
test_data = mid[200:]