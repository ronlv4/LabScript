import matplotlib.pyplot as plt
from iminuit import Minuit, describe
from iminuit.util import make_func_code
from matplotlib.offsetbox import AnchoredText
from pylab import *
from numpy import *
import numpy as np
import pandas as pd
from matplotlib import pylab, mlab, pyplot
import warnings

warnings.simplefilter("ignore")


class EffVarChi2Reg:  # This class is like Chi2Regression but takes into account dx
    # this part defines the variables the class will use
    def __init__(self, model, x, y, dx, dy):
        self.model = model  # model predicts y value for given x value
        self.x = array(x)  # the x values
        self.y = array(y)  # the y values
        self.dx = array(dx)  # the x-axis uncertainties
        self.dy = array(dy)  # the y-axis uncertainties
        self.func_code = make_func_code(describe(self.model)[1:])
        self.h = (x[-1] - x[
            0]) / 10000  # this is the step size for the numerical calculation of the df/dx = last value in x (x[-1]) - first value in x (x[0])/10000

    # this part defines the calculations when the function is called
    def __call__(self, *par):  # par are a variable number of model parameters
        self.ym = self.model(self.x, *par)
        df = (self.model(self.x + self.h,
                         *par) - self.ym) / self.h  # the derivative df/dx at point x is taken as [f(x+h)-f(x)]/h
        chi2 = sum(((self.y - self.ym) ** 2) / (self.dy ** 2 + (
                df * self.dx) ** 2))  # chi2 is now Sum of: (f(x)-y)^2/(uncert_y^2+(df/dx*uncert_x)^2)
        return chi2

    # this part defines a function called "show" which will make a nice plot when invoked
    def show(self, optimizer, x_title="X", y_title="Y", goodness_loc='upper right'):
        self.par = optimizer.parameters
        self.chi2 = optimizer.fval
        self.ndof = len(self.x) - len(self.par)
        self.chi_ndof = self.chi2 / self.ndof
        self.par_values = optimizer.values
        self.par_errors = optimizer.errors
        text = ""
        for _ in (self.par):
            text += "%s = %0.4f \u00B1 %0.4f \n" % (_, self.par_values[_], self.par_errors[_])

        text = text + "\u03C7\u00B2 /ndof = %0.4f(%0.4f/%d)" % (self.chi_ndof, self.chi2, self.ndof)
        self.func_x = np.linspace(self.x[0], self.x[-1], 10000)  # 10000 linearly spaced numbers
        self.y_fit = self.model(self.func_x, self.par_values[0], self.par_values[1])
        plt.rc("font", size=16, family="Times New Roman")
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_axes([0.12, 0.15, .85, .8])
        ax.plot(self.func_x, self.y_fit)  # plot the function over 10k points covering the x axis
        ax.scatter(self.x, self.y, c="red")
        ax.errorbar(self.x, self.y, self.dy, self.dx, fmt='none', ecolor='red', capsize=3)
        ax.set_xlabel(x_title, fontdict={"size": 21})
        ax.set_ylabel(y_title, fontdict={"size": 21})
        ax.set_title('Stopping Voltage as a Function of the Inverse of Minimal Wavelength', fontdict={"size":18})
        anchored_text = AnchoredText(text, loc=goodness_loc)
        ax.add_artist(anchored_text)
        plt.grid(True)
        plt.show()


# an example usage
df = pd.read_excel(r'Data.xlsx')
measured_x = df['x'].values
measured_y = df['y'].values
measured_x = 1 /measured_x
print(measured_x)
x_uncertainties = np.full_like(measured_x, df['x_err'].values[0], float)
# y_uncertainties = np.full_like(measured_y, df['y_err'].values[0], float)
y_uncertainties = df['y_err'].values

# Let's do a fit to a linear function (but EffVarChi2Reg will work for any function of x)
linear_fun = lambda x, a, b: a * x + b
# just note that in practice it will be used for the same thing but in "coding" terms this is
# a different variable x than x in the class def above
exp_fun = lambda x, a, b: a * np.exp(b * x)
# Now we create a regression object (from the class defined above) with our example data and function
efvtest = EffVarChi2Reg(linear_fun, measured_x, measured_y, x_uncertainties,
                        y_uncertainties)  # the syntax for EffVarChi2Reg is (function, x,y,dx,dy)

# Feed the regression object we just made to Minuit
efopt = Minuit(efvtest, a=1, b=-1)
# the syntax is similar the same as when using Chi2Regression or any other function that Minuit works with, i.e. you could put limits etc here also

# Run the minimization
efopt.migrad()

# and look at the results
v = np.array(efopt.values)
u = np.array(efopt.errors)
number_param = len(list(efopt.values))
ndof = len(measured_x) - number_param
chi2_ndof = efopt.fval / ndof
# print("intercept is %0.4f \u00B1 %0.4f" %(v[0],u[0]))
# print(10 * "---")
# print("slope is %0.4f \u00B1 %0.4f"%( v[1],u[1]))
# print(10 * "---")
# print("Chi2/ndof is %0.4f(%0.4f/%d)"%(chi2_ndof,efopt.fval,ndof))

efvtest.show(efopt, goodness_loc='upper right', x_title=r'$\frac{1}{\lambda} [\frac{1}{nm}]$', y_title=r'$ V_0 [V]$')
