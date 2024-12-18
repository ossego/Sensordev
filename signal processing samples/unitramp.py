#to plot unit ramp fuction

#import libraries

import matplotlib.pyplot as plt

#define variables

n = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
r_n = [0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5]

#plot discrete (stem)

plt.stem(n, r_n)
plt.title('Unit Ramp Function')
plt.xlabel('--->n')
plt.ylabel('r(n)')
plt.show()
