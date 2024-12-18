# to plot expontential signal

# import libraries

import numpy as np
import matplotlib.pyplot as plt

# define independent and dependent variables

n = np.arange(0,10,1)
x_n = np.exp(0.1*n)

# plot

plt.stem(n,x_n)
plt.title('Exponential Signal')
plt.xlabel('---->n')
plt.ylabel('x(n)')
plt.show()
