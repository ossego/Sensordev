# to Plot unit impulse function

# import libaries
import matplotlib.pyplot as plt

#define independent and dependent variables

n = [-5, -4, -3, -2, -1, 0 , 1, 2, 3, 4, 5]
delta = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

# plot discrete = stem

plt.stem(n,delta)
plt.title('Unit Impulse Fuction')
plt.xlabel('---> n')
plt.ylabel('Delta Function')
plt.show()
