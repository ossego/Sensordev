# To plot discrete (sampled) sin and cos signal

# import libraries
import numpy as np
import matplotlib.pyplot as plt

# to define independent and dependent variable
n = np.arange(0,1,0.01)
xn = np.sin(2*(np.pi)*3*n)
print(n)

# to plot signal (np.stem means discrete while np.plot is for continous.)

plt.stem(n,xn)
plt.title('Discrete Sin Wave')
plt.xlabel('----->n')
plt.ylabel("Amplitude")
plt.show()
