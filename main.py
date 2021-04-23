import math

from iminuit import Minuit
from probfit import Chi2Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings


def findErrorForChi2overNDOFis1(function, start, end, dx, err_size):
    errors = np.arange(start, end, dx)
    errors_array = []
    for error in errors:
        for i in range(0, err_size):
            errors_array.append(error)
    y_err = np.array(errors_array[0:err_size])
    reg = Chi2Regression(linear_fun, x, y, y_err, weights=None)
    i = 0
    opt = Minuit(reg, a=0, b=0)
    opt.migrad()
    # print(reg.__call__(0.1,0.1))
    while abs(opt.fval / reg.ndof - 1) > 0.01 and i < len(errors_array) - 12:
        y_err = np.array(errors_array[i:i + err_size])
        reg = Chi2Regression(linear_fun, x, y, y_err, weights=None)
        # print(reg.__call__(0,0)/reg.ndof)
        opt = Minuit(reg, a=0, b=0)
        opt.migrad()
        i += err_size
    print(y_err)
    return y_err


df = pd.read_excel('Data.xlsx')
y = df["y"].values  # Voltage
x= df["x"].values  # diameter
y_err = df["y_err"].values
x_err = df["x_err"].values


# i = np.full_like(T,dtype=float, fill_value=i[0])
x_err = np.full_like(x,dtype=float, fill_value=x_err[0])
y_err = np.full_like(y,dtype=float, fill_value=y_err[0])
# B_err = np.full_like(T ,dtype=float, fill_value=B_err[0])

# calculations
i = y
i_squared = i * i
r = x / 2  # radius
# r_squared = r ** 2  # r^2

# y = 1 / (i ** 2)
# x = r**2
x=r
y=i
# x_err = x_err * r
# y_err = (2 / (i ** 3)) * y_err
# y=(((2*np.pi)/T)**2)*i
# x=B
# y_err = ((2*(2*np.pi/T)*(1/(T**2))*T_err)**2+((((2*np.pi)/T)**2)*i_err)**2)**0.5
# print(y_err)
# x_err = B_err
# print("y: ", y)
# print("y error: ", y_err)
# print("x: ", x)
# print("x error: ", x_err)

# function
linear_fun = lambda x, a, b: a * x + b
pol_2_fun = lambda x, a, b, c: a * x ** 2 + b * x + c
pol_3_fun = lambda x, a, b, c, d: a * x ** 3 + b * x ** 2 + c * x + d
sin_fun = lambda x, a, b: a * np.sin(b * x)
exp_fun = lambda x, a, b: a * np.exp(b * x)
gaussian = lambda x, a, e, s: a * np.exp(-((x - e) ** 2) / (2 * s ** 2))
hiperbol_fun = lambda x, a, b, c: (a / (b * x)) + c
# y_err=findErrorForChi2overNDOFis1(linear_fun, 4.13348623, 4.13348627, 0.0000000001, len(x))

# define the regression,i.e. the function to be minimized, in our usual case we use Chi2Regression this is an already
# built method to pass chi2 to Minuit as the function to be minimized
reg = Chi2Regression(linear_fun, x, y, y_err, weights=None)
# Chi2Regression takes the arguments (function_to_fit, x_values,y_values,y_uncertainties, weights)
# if you are dealing with data that has no measured point by point uncertainty (i.e. you have no "y_err" in our example)
# you need an unweighted least squares fit equivalent to weights=1 (or y_err =1 b/c 1/1^2 also equals 1)
# Chi2Regression will do this by default if you don't give it errors
# reg = Chi2Regression(pol_2_fun,X,y)
# which is identical to
# reg = Chi2Regression(pol_2_fun,X,y,error=None)
# (you may find it interesting to compare the different fit output for a given dataset if the errors are ignored)


# run the optimization
opt = Minuit(reg)
# here we are setting up minuit (the minimization tool);
# it takes the arguments (function_to_minimize,parameter_0=initial_value_0,parameter_i=init_value_i,...,parameter_n=init_value_n) for n parameters
# typically the function to minimize will be chi2 as defined above (reg=Chi2Regression(...)), but it could be something else e.g. Minuit(least_squares, a=0,b=0...)
# this is also the place to set limits on the parameters if desired; e.g.
# opt = Minuit(reg, a=0, b=0, c=0, limit_a=(0,1000),limit_b=(-200,None),limit_c=(-500,500))
# where limit_a was really limit_"name" where "name" was the name of the parameter you're setting a limit on

opt.migrad()  # actually do the fitting, i.e. run the optimization, i.e. minimize chi2 as defined in reg

# The output into np arrays
# v = opt.np_values() #the np_ methods return a np style (i.e. format) array
# u = opt.np_errors()
# the np_ methods no longer work in Minuit, opt.values & opt.errors return a "dictionary" which has both values and a key (in this case the parameter names)
# asking for .values() gets you just the values without the keys
# v = np.array(opt.values.values())
# u = np.array(opt.errors.values())
# print(v)
# print(u)

# covariance
# just like the values and uncertainties above we grab just the values from the covariance matrix dictionary
# a = np.array(list(opt.covariance.values()))

# print(a)
# you can see that our previous line got the values but unfortunately turned it into an array with length NxN rather than an NxN matrix

# N = int(len(a) ** 0.5)
# so to turn it back into an array, let's first grab N (as the sqrt of the length of a)

# covm = np.reshape(a, (-1, N))
# and then "reshape" the array into an NxN matrix
# We could have done it all in one line as:
# covm = np.reshape(np.array(list(opt.covariance.values())), (-1, int(len(np.array(list(opt.covariance.values())))**0.5)))
# which is identically the same steps, but just a little harder to follow without seeing them as separate steps

# print("covariance matrix:\n", covm)
# print("And just to see the roots of the diagonals",
#       round(np.sqrt(covm[0][0]), 8),
#       round(np.sqrt(covm[1][1]), 8),
#       round(np.sqrt(covm[2][2]), 8))
# print("compared to u:                           ", u)

# plot

plt.rc("font", size=16, family="Times New Roman")
fig = plt.figure(figsize=(10, 6))
ax = fig.add_axes([0.1, 0.15, .85, .72])
# L B R T
ax.set_xlabel(r'$ I[A] $', fontdict={"size": 20, "weight": "bold"})
ax.set_ylabel(r'$ B[\mu T] $', fontdict={"size": 20, "weight": "bold"})
ax.set_title("Magnetic Field as a Function of Current", fontdict={"size": 24, "weight": "bold"})
ax.errorbar(x=x, y=y, yerr=y_err, xerr=x_err, capsize=4, elinewidth=3, fmt='none', ecolor="blue")
ax.scatter(x, y, c='blue', s=10)
right_parm_loc = (0.7, 0.95)
reg.show(minuit=None, errors=opt.errors)
