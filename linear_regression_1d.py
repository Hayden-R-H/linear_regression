"""
@title: Linear Regression in 1-D
@author: hhohns
@date: Created on Wed Oct 11 10:43:35 2017
@brief: Python code for performing 1-D linear regression on a set of data
containing floats of x and y values. In addition, the r^2 value is calculated.
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. Load data
x = []
y = []
for line in open('data_1d.csv'):
    x_i, y_i = line.split(',')
    x.append(float(x_i))
    y.append(float(y_i))

# 2. Convert to numpy arrays
x = np.array(x)
y = np.array(y)

# 3. Apply linear regression equations to calculate gradient a and shift b.
denominator = x.dot(x) - x.mean() * x.sum() # a and b share denominators
a = (x.dot(y) - y.mean() * x.sum()) / denominator
b = (y.mean() * x.dot(x) - x.mean() * x.dot(y)) / denominator

# 4. Calculate predicted Y from X using a and b
yHat = a * x + b

# 5. Plot data
plt.scatter(x, y)
plt.plot(x, yHat)
plt.xlabel('x')
plt.ylabel('y = f(x) = ax + b')
plt.title('Linear Regression')
plt.show()

# 6. Goodness of Fit (R^2 value)
res = (y - yHat)
SSres = res.dot(res)
tot = y - y.mean()
SStot = tot.dot(tot)
rSquared = 1 - SSres / SStot
print("The R^2 value is ") + str(rSquared)

