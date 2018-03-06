from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1, 2, 3, 4, 5, 6], dtype = np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype = np.float64)

def best_fit_slope_and_intercept (xs, ys):
    # numerator of eqn
    m = (mean(xs) * mean(ys))
    m -= mean(xs * ys)

    # denominator of eqn
    m = m / ((mean(xs) ** 2) - mean(xs ** 2))

    b = mean(ys) - m * mean(xs)
    return m, b

# squared error function
def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) ** 2)

# calculate coefficient of determination for regression line (finds r squared value)
def determination_coefficient(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig)] * len(ys_orig)
    squared_error_regr = squared_error(ys_orig, ys_line)
    squarred_error_y_mean = squared_error(ys_orig, y_mean_line)

    return 1 - squared_error_regr / squarred_error_y_mean

m, b = best_fit_slope_and_intercept(xs, ys)

# regression line array
regression_line = [(m * x) + b for x in xs]

predict_x = 8
predict_y = m * predict_x + b

r_squared = determination_coefficient(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y)
plt.plot(xs, regression_line)
plt.show()