from mt5linux import MetaTrader5
import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import pandas as pd
from fracdiff.sklearn import Fracdiff

# Initialize MetaTrader5 and other libraries
mt5 = MetaTrader5(host='localhost', port=18812)

# Generate random data for time series
np.random.seed(0)
data = np.random.randn(50)
time_index = pd.date_range(start='2023-01-01', periods=50, freq='D')
ts = pd.Series(data, index=time_index)
ts_df = ts.to_frame()

# Apply fractional differentiation
fd = Fracdiff(d=0.5)
result = fd.fit_transform(ts_df)

# Plot both original and differentiated time series on the same graph
plt.figure(figsize=(10, 6))
plt.plot(ts.index, ts.values, label='Original', color='red')  # Plot original series in red
plt.plot(ts.index, result.squeeze(), label='Differentiated', color='blue')  # Plot differentiated series in blue
plt.title('Original and Fractionally Differentiated Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()
