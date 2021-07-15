import config
from binance.client import Client
from binance.enums import *
import pandas as pd
import numpy as np
import pylab as plt
from matplotlib import pyplot
from numpy import linalg
from sklearn import linear_model
import seaborn as sns

client = Client(config.API_KEY, config.API_SECRET)

# valid intervals - 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

def derivative(x):
    dx = np.zeros(len(x))
    for i in range(1,len(x)-2):
        dx[i-1] = x[i]-x[i-1]
    return dx

def print_file(pair, interval, name):

    # request historical candle (or klines) data
    bars = client.get_historical_klines(pair, interval, "14 Jun, 2018", "14 Jul, 2021")
    
    # delete unwanted data - just keep date, open, high, low, close
    for line in bars:
        del line[6:]
    
    # save as CSV file 
    df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df.set_index('date', inplace=True)
    df.to_csv(name)

print_file('BTCUSDT', '1d', "BTC_kinematics.csv")

df = pd.read_csv("BTC_kinematics.csv", parse_dates=True)
closes = df.close
volumes = df.volume
volumes = np.array(volumes)
closes = np.array(closes)
closes_v = derivative(closes)
closes_a = derivative(closes_v)
closes_j = derivative(closes_a)

dates = pd.to_datetime(df['date'], unit='ms')
timestamps = df['date']
timestamps = np.array(timestamps)

fig, axs = plt.subplots(4)
axs[0].plot(dates, closes, color = 'b')
axs[0].set_title('Bitcoin close price [$]')
axs[1].plot(dates, closes_v, color = 'r')
axs[1].set_title('Bitcoin velocity [$/d]')
axs[2].plot(dates, closes_a,color = 'g')
axs[2].set_title('Bitcoin acceleration [$/d²]')
axs[3].plot(dates, closes_j,color = 'm')
axs[3].set_title('Bitcoin jerk [$/d³]')
for ax in fig.get_axes():
    ax.label_outer()
plt.legend()
plt.show()

plt.plot(closes, closes_v,'o')
plt.xlabel("Price (x)")
plt.ylabel("Price velocity (x')")
plt.show()

plt.plot(closes, closes_a,'o')
plt.xlabel("Price (x)")
plt.ylabel("Price acceleration (x'')")
plt.show()

plt.plot(closes, closes_j,'o')
plt.xlabel("Price (x)")
plt.ylabel("Price jerk (x''')")
plt.show()

plt.plot(closes_v, closes_a,'o')
plt.xlabel("Price velocity (x')")
plt.ylabel("Price acceleration (x'')")
plt.show()

plt.plot(closes_a, closes_j,'o')
plt.xlabel("Price acceleration (x'')")
plt.ylabel("Price jerk (x''')")
plt.show()

ax = plt.axes(projection='3d')
ax.scatter(closes, closes_v, closes_a, c=closes_a, cmap='viridis', linewidth=0.5)
ax.set_xlabel("Price (x)")
ax.set_ylabel("Price velocity (x')")
ax.set_zlabel("Price acceleration (x'')")
plt.show()

ax = plt.axes(projection='3d')
ax.scatter3D(closes, closes_v, closes_a, c=closes_a, cmap='Greens')
ax.set_xlabel("Price (x)")
ax.set_ylabel("Price velocity (x')")
ax.set_zlabel("Price acceleration (x'')")
plt.show()

ax = plt.axes(projection='3d')
ax.scatter(closes_v, closes_a, closes_j, c=closes_j, cmap='viridis', linewidth=0.5)
ax.set_xlabel("Price velocity (x')")
ax.set_ylabel("Price acceleration (x'')")
ax.set_zlabel("Price jerk (x''')")
plt.show()

ax = plt.axes(projection='3d')
ax.scatter3D(closes_v, closes_a, closes_j, c=closes_j, cmap='Greens')
ax.set_xlabel("Price velocity (x')")
ax.set_ylabel("Price acceleration (x'')")
ax.set_zlabel("Price jerk (x''')")
plt.show()

sns.distplot(closes_v)
plt.xlabel('Bitcoin velocity [$/d]')
plt.ylabel("Frequency")
plt.show()

sns.distplot(closes_a)
plt.xlabel('Bitcoin acceleration [$/d²]')
plt.ylabel("Frequency")
plt.show()

sns.distplot(closes_j)
plt.xlabel('Bitcoin jerk [$/d³]')
plt.ylabel("Frequency")
plt.show()

eps = 1e-9
A = np.matrix([closes_j, closes_a, closes_v, closes, np.ones(len(closes))])
A = A.reshape(len(closes),5)
B = np.ones(len(closes))*eps
X = np.linalg.lstsq(A, B)
print(X)
