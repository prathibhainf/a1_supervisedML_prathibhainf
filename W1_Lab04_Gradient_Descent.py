import math, copy
import numpy as np
import matplotlib.pyplot as plt


# load the data set
x_train = np.array([1, 2]) # features
y_train = np.array([300, 500]) # target values


# function to calculate cost
def compute_cost(x, y, w, b):
    m = len(x)
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1/ (2 * m) * cost
    return total_cost


# computing the gradient of cost function for linear regression
def compute_gradient(x, y, w, b):
    # number of training examples
    m = len(x)
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        # geting the model prediction
        f_wb = w * x[i] + b
        # intermediate step in finding the gradient 
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        # getting the gradient 
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


# gradient decent
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    J_history = [] # history of cost
    p_history = [] # history of parameters
    b = b_in
    w = w_in

    for i in range(num_iters):
        # calculate the gradient 
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # update the parameters
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        # save cost J at each iteration
        if i < 100000: #prevent resource exhaustion
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])
        # print cost every at intervals 10 time 
        if (i % math.ceil(num_iters/10) == 0):
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e}", f"w: {w: 0.3e}, b: {b:0.5e}")
    return w, b, J_history, p_history 


# initializing parameters
w_init = 0
b_init = 0
# some gradient descent setting
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient decent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f}, {b_final:8.4f})")


# here we have to decide on following things
#.....learning rate
#.....wether the cost is reducing with each iteration
#.....number of iterations sufficients