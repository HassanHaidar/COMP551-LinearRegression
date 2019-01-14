import numpy as np

#define an example set of X, of size 10 x 2
# every x has 1 fearure, but we add 1 feature of value 1 as a bias (intercept)
# without the bias, size of x would be 10 x 1
X = np.array([[0.86,0.09,-0.85,0.87,-0.44,-0.43,-1.1,0.40,-0.96,0.17],[1.0 for i in range(10)]])

print(X.shape)

#define an example set of target y, of size 10 x 1
Y = np.array([2.49,0.83,-0.25,3.10,0.87,0.02,-0.12, 1.81, -0.83, 0.43])
Y = Y.reshape(10,1)

#define initial function f of size 1 x 2
W = np.array([1, 1], dtype = float)

# find the value of (transpose of X)*(X)
xt_x = X @ np.transpose(X)
print(xt_x.shape)

# find the value of the previous value
xt_x_inverse = np.linalg.inv(xt_x)
print(xt_x_inverse.shape)

# weights = (x^T * x)^(-1) * x^T * Y
W = (xt_x_inverse @ X) @ Y

print(W.shape)

print("Best fit line for the input data is y = {}x + {}".format(W[0][0],W[1][0]))



