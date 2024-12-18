# To plot discrete time sequence

# import libraries
import matplotlib.pyplot as plt

# define independent and dependentvariable as index (list)
n = [0, 1, 2, 3, 4, 5, 6]
xn = [-1, 4, -3, 0, 9, 8, 9]

# plot discrete time sequence

plt.stem(n,xn)
plt.title('Discrete Time Sequence')
plt.xlabel('---> n')
plt.ylabel('Amplitude')
plt.show()
