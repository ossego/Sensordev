# to plot unit step function

#import libraries
import matplotlib.pyplot as plt

#define independent and dependent variables

n = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
u_n = [0 ,0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

#plot discrete (use np.stem) for continues use np.plot

plt.stem(n,u_n)
plt.title('Unit Step Function')
plt.xlabel('--->n')
plt.ylabel('u[n]')
plt.show()
