import warnings
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
warnings.filterwarnings("ignore")
rcParams['figure.figsize'] = 18, 8
plt.style.use('fivethirtyeight')
import statsmodels.api as sm
import itertools

df = pd.read_excel("Sample - Superstore.xls")
# We only are concerned with furniture data

furniture = df.loc[df['Category'] == 'Furniture']
cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
furniture.drop(cols, axis=1, inplace=True)
furniture = furniture.sort_values('Order Date')
furniture.isnull().sum()

furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()

# Indexing with Time Series

furniture = furniture.set_index('Order Date')
y = furniture['Sales'].resample('MS').mean()
y.plot(figsize=(15, 6)); plt.show()

# Decompose the time series into Trend, Seasonality, and Noise
decomposed_data = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposed_data.plot()
plt.show()

#ARIMA: Autoregressive Integrated Moving Average
# ARIMA(p, d, q): seasonality, trend, noise

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# for param in pdq:
#     for param_seasonal in seasonal_pdq:
#         try:
#             mod = sm.tsa.statespace.SARIMAX(y,
#                                             order=param,
#                                             seasonal_order=param_seasonal,
#                                             enforce_stationarity=False,
#                                             enforce_invertibility=False)
#             results = mod.fit()
#             print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
#         except:
#             continue

mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
import code; code.interact(local=dict(globals(), **locals()))
results.plot_diagnostics(figsize=(16, 8))
plt.show()
