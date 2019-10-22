import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from scipy import signal
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
plt.rcParams["savefig.directory"] = os.chdir(os.path.dirname('./'))

# Input file paths
flxsPath = './stockData/FLXS_2017_2018.csv'
vfcPath = './stockData/VFC_2017_2018.csv'
flxs = pd.read_csv(flxsPath, index_col = 0) # steel manu.
vfc = pd.read_csv(vfcPath, index_col = 0) # apparel company

# Plotting individual stock prices with the cross correlation value
corr = flxs['High'].corr(vfc['High'])
fig, ax = subplots()
flxs['High'].plot(legend = True,ax=ax, title=corr)
vfc['High'].plot(legend = True,ax=ax)
ax.legend(['$vfc','$flxs'])
plt.show()

# Cross correlation plot
corrSignals = signal.correlate((vfc['High'].values), (flxs['High'].values), mode='same') / 128
plt.clf()
plt.plot(corrSignals)
plt.title("Cross correlation between the two stocks")
plt.show()

# Autocorrelation_plot VFC
autocorrelation_plot(vfc['High'].values)
pyplot.title("vfc autocorrelation_plot")
pyplot.show()

# Autocorrelation_plot FLXS
autocorrelation_plot(flxs['High'].values)
pyplot.title("flxs autocorrelation_plot")
pyplot.show()

# Detrending and plotting $FLXS
X = flxs['High']
diffa = list()
for i in range(1, len(X)):
	value = X[i] - X[i - 1]
	diffa.append(value)
plt.plot(diffa)
plt.title("Detendring $flxs")
plt.show()

# Detrending and plotting $VFC
X = vfc['High']
diffb = list()
for i in range(1, len(X)):
	value = X[i] - X[i - 1]
	diffb.append(value)
plt.title("Detendring $vfc")
plt.plot(diffb)
plt.show()

# Analysing individual graphs using seasonal decomposition
result = seasonal_decompose(vfc['High'].values, model='additive', freq=1)
result.plot()
pyplot.title("$vfc characteristics")
pyplot.show()

result = seasonal_decompose(flxs['High'].values, model='additive', freq=1)
result.plot()
pyplot.title("$flxs characteristics")
pyplot.show()
plt.clf()

# ARIMA model for FLXS
model = ARIMA(flxs['High'].values, order=(5,0,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
residuals = pd.DataFrame(model_fit.resid)
print(residuals.describe())
size = int(len(flxs['High'].values) * 0.66)
train, test = flxs['High'].values[0:size], flxs['High'].values[size:len(flxs['High'].values)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.title("ARIMA predictions $FLXS")
pyplot.show()


# ARIMA model for VFC
model = ARIMA(vfc['High'].values, order=(5,0,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
print(residuals.describe())
size = int(len(vfc['High'].values) * 0.66)
train, test = vfc['High'].values[0:size], vfc['High'].values[size:len(vfc['High'].values)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.title("ARIMA predictions $VFC")
pyplot.show()
sys.exit(1)