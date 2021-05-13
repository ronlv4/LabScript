import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uuid


df = pd.read_excel('Data.xlsx')
x = df['voltage'].values

s_with = df['s_with'].values
s_without = df['s_without'].values
a_with = df['a_with'].values
a_without = df['a_without'].values
voltage = df['voltage'].values

y = 1 - (a_with * s_without) / (a_without * s_with)
plt.style.use('ggplot')
plt.figure(figsize=(10,6))
plt.plot(x, y, label="probability")
plt.xlabel(r'Voltage [V]')
plt.ylabel('Probability')
plt.title('Scattering probability of Electrons as a Function of Voltage ')
plt.legend()
plt.show()

print(np.min(y[4:]))
print(np.argmin(y))
print(x[3])