import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

data_set = Path("Housing.csv")
housing_prices = pd.read_csv(data_set)
print(housing_prices.info())
y = housing_prices[["price"]].values
x = housing_prices[["area"]].values

x = (x-x.mean())/(x.max()-x.min())
y = (y-y.mean())/(y.max()-y.min())

k = 1000

m = x.shape[0]
a = 1*10**(-1)
w = 10
b = 15
print(m)
def gradient(w,b, m):
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        dj_dw += (1/m)*((w*x[i]+b)-y[i])*x[i]
        dj_db += (1 / m)*((w * x[i] + b) - y[i])
    return dj_dw, dj_db

wl, bl = [],[]
def gradient_descent(w,b,m,a):

    dj_dw , dj_db = gradient(w,b,m)
    tm_w = w - a*dj_dw
    tm_b = b - a*dj_db
    w = tm_w
    b = tm_b
    return w,b, wl,bl

def compute(x,y,w,b,m, k,wl,bl):
    tmp_plt = np.zeros(m)
    i = 0
    while (i < k):
        w, b, wl, bl= gradient_descent(w, b, m, a)
        wl.append(w)
        bl.append(b)
        i += 1
    for i in range(m):
        tmp_plt[i] = w*x[i]+b
    return tmp_plt

prediction = compute(x,y,w,b,m,k,wl,bl)

print(w,b)

f1 = plt.figure()
plt.scatter(x,y,marker=".", c="r")
plt.plot(x,prediction, "b")
plt.show()
f2 = plt.figure()
w = np.linspace(-2000, 2000, 500)
plt.scatter(w,b, marker=".", c="g")

plt.plot(w,b, "b")
plt.show()
