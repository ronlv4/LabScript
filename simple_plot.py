import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_excel(r'Data.xlsx')
measured_x = df['x'].values
measured_y = df['y'].values
# x_uncertainties = np.full_like(measured_x, df['x_err'].values[0], float)
# y_uncertainties = np.full_like(measured_y, df['y_err'].values[0], float)
x_uncertainties = df['x_err'].values
y_uncertainties = df['y_err'].values

# plot1 = ax.plot(measured_x, measured_y, label='check')
# ax.legend()
# plt.figure(1, figsize=(10, 6))
marker_color = 'blue'
line_color = 'brown'
plt.figure(figsize=(10,6))
plt.plot(measured_x, measured_y,color= line_color,marker='.', label='540 [nm]', markeredgecolor=marker_color, markerfacecolor=marker_color)
plt.grid()
# plt.legend()
plt.xlabel('Voltage [V]')
plt.ylabel('Current [mA]')
plt.title('I-V Characteristic - filter 602 [nm]')
plt.show()
