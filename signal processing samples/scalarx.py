# Scalar multiplication and vector addition

#import libraries
import numpy as np

#define a column vector
x = np.array([[1],[2],[3],[4]])
print('x= ', x)

#multipy column vector with scalar 3
a = 3

print('column vector times scalar =', a*x)

#addition of column vectors
y = -3*x
print('column vector= ',y)

z = x + y
print('z is vector addition= ',z)
