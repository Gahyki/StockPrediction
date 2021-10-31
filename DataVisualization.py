import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('KO.csv', usecols=['Date', 'Open', 'High', 'Low', 'Close'])
# dataframe
print("This is a dataframe: \n", df)

# shape
print("This is the shape of the dataframe: \n", df.shape)

# first 10 rows
print("This is the first 10 rows: \n", df.head(10))

# boxplot graph displayed in the same way as candlestick chart
df.transpose()[1:].plot(kind='box', sharex=False, sharey=False)
plt.show()
