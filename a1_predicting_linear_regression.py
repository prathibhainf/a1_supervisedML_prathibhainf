import numpy as np
import matplotlib.pyplot as plt

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# pltot the data points
plt.scatter(x_train, y_train, marker = 'x', c = 'r')
# set the title
plt.title('Housing prices')
# set the y axis lavel
plt.ylabel('Price')
# set the x axis label
plt.xlabel('Size')
plt.show()

# the model for approximating housing prices
w = 100
b = 100

def compute_model_output(x_array, w, b):
    # m is the number of training examples
    m = len(x_array)
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x_array[i] + b
    return f_wb

# call the compute_model_output function
predicted_vals = compute_model_output(x_train, w, b)

# plot our model prediction
plt.plot(x_train, predicted_vals, c = 'b', label = 'predicted values') #following the same plot.show() command

# plot the data points
plt.scatter(x_train, y_train, c = 'g', label = 'actual values')  #following the same plot.show() command

# set the y axis label
plt.ylabel('price')
# set the x axis lavel
plt.xlabel('size')
plt.legend()
plt.show()


