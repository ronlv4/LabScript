import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_excel(r'Data.xlsx')
x = df['x'].values
y = df['y'].values
x2 = df['x2'].values
y2 = df['y2'].values
x3 = df['x3'].values
y3 = df['y3'].values
# x_uncertainties = np.full_like(measured_x, df['x_err'].values[0], float)
# y_uncertainties = np.full_like(measured_y, df['y_err'].values[0], float)
x_uncertainties = df['x_err'].values
y_uncertainties = df['y_err'].values

# plot1 = ax.plot(measured_x, measured_y, label='check')
# ax.legend()
# plt.figure(1, figsize=(10, 6))
marker_color = 'blue'
line_color = 'red'
marker_color2 = 'blue'
line_color2 = 'gray'
marker_color3 = 'blue'
line_color3 = 'brown'
plt.figure(figsize=(10,6))
plt.plot(x2,y2,color= line_color2,marker='.', label='450 [nm]', markeredgecolor=marker_color2, markerfacecolor=marker_color2)
plt.plot(x,y,color= line_color,marker='.', label='540 [nm]', markeredgecolor=marker_color, markerfacecolor=marker_color)
plt.plot(x3,y3,color= line_color3,marker='.', label='602 [nm]', markeredgecolor=marker_color3, markerfacecolor=marker_color3)
plt.grid()
plt.legend()
plt.xlabel('Voltage [V]')
plt.ylabel('Current [mA]')
plt.title('I-V Characteristic - all filters')
plt.show()
