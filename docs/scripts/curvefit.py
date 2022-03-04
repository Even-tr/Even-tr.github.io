import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def logit(scale, speed, offset, height):
  return lambda x:  height + scale/(1 + np.exp(-speed*(x + offset)))
  
def l2norm(f, x, y):
  yhat = f(x)
  return np.sum((y - yhat)**2)

def to_optimize(params, x, y):
  f = logit(*params)
  return l2norm(f, x, y)

n_data = 100
xmin, ymin = -10, 10
noise_scale = 0.01

def make_data(n, params):
  f = logit(*params)
  xs = np.random.uniform(xmin, ymin, n)
  ys = f(xs) + np.random.normal(0,noise_scale, n)
  return xs, ys, f

x_data, y_data, func = make_data(n_data, [100,-1,10,0.4])

def norm(x, y):
  xmean = x.mean()
  ymean = y.mean()
  xstd = x.std()
  ystd = y.std()

  return (x - xmean)/xstd, (y - ymean)/ ystd, (xmean, ymean, xstd, ystd)

x_norm, y_norm, parmamamama = norm(x_data, y_data)


xplot = np.linspace(xmin, ymin, 200)

plt.scatter(x_data, y_data)
plt.plot(xplot, func(xplot), label='True function')
plt.show()

params = np.array([1, 1, 1, 1]) #inital guess

res = minimize(to_optimize, params, args=(x_norm, y_norm), method='BFGS')
params_found = res.x

model = logit(*params_found)
plt.plot(xplot, model(xplot), label='model')
plt.scatter(x_norm, y_norm)
plt.legend()
plt.show()