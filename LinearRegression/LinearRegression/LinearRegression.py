import numpy as np

# obtain input data x from user
# every x has 1 feature, but we add 1 feature of value 1 as a bias (intercept)
x_data =  input("Enter the x data as x1 x2 ... xn: ")
x_data = x_data.split();
x_data = list(map(float, x_data))

X = np.array([x_data,[1.0 for i in range(len(x_data))]])
X = np.transpose(X);


# obtain data y
y_data = input("Enter the corresponding y data. Make sure you enter the same number of data for y as for x: ") #request input from user
y_data = y_data.split()
Y = list(map(float, y_data))
if (len(Y) != len(X)):
    print("Different number of data!")
else:
    Y = np.array(y_data, float)

    Y = Y.reshape(Y.size,1)

    # find the value of (transpose of X)*(X)
    xt_x = np.transpose(X) @ X

    # find the value of the previous value
    xt_x_inverse = np.linalg.inv(xt_x)

    # weights = (x^T * x)^(-1) * x^T * Y
    W = (xt_x_inverse @ np.transpose(X)) @ Y

    print("Best fit line for the input data is y = {}x + {}".format(W[0][0],W[1][0]))

 




