import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('KO.csv', usecols=['Date', 'Open', 'High', 'Low', 'Close'])
# dataframe
print(df)

# shape
print(df.shape)

# first 10 rows
print(df.head(10))

# boxplot graph displayed in the same way as candlestick chart
df.transpose()[1:].plot(kind='box', sharex=False, sharey=False)
plt.show()
