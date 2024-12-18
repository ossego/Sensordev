
#import libraries
import numpy as np
import matplotlib.pyplot as plt

#define independent (t) and dependent (xt1,xt2,xt3) variables

t = np.arange(0,1,0.01)
xt1 = np.cos(2*(np.pi)*3*t)
xt2 = np.cos(2*np.pi*5*t)
xt3 = np.cos(2*np.pi*7*t)
xt4 = np.cos(2*np.pi*9*t)

# plot signals using subplots

plt.subplot(4,1,1)
plt.plot(t,xt1)
plt.subplot(4,1,2)
plt.plot(t,xt2)
plt.subplot(4,1,3)
plt.plot(t,xt3)
plt.subplot(4,1,4)
plt.plot(t,xt3)

plt.show()code 