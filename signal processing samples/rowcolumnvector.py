# to define row and column vector

# import libraries
import numpy as np

# define row vector
e = np.array([[1,2,-4,5]])
print('e= ', e)

#define type of vector (1 row 4 column)
print('shape of e= ',np.shape(e))

#define column vector
d = np.array([[1],[2],[3],[4]])
print('d= ',d)

#define type of vector (4 row 1 colum)
print('shape of d= ', np.shape(d))

#define row and column vector for dot product

x = np.array([[1,2,3]])
y = np.array([[1],[2],[3]])
print('x = ',x)
print('y = ',y)

# perform dot product
z = np.dot(x,y)
print('dot product answer using np.dot = ', z)

# perform cross product two methods (outer or multiplty)
w = np.outer(x,y)
print('cross product answer using np.outer= ', w)

k = np.multiply(x,y)
print('cross product answer using np.multiply= ',k)

# perform element by element multiplication (use x,x for row answer, y,y for column answer)
j = np.multiply(x,x)
print('Element by Element answer = ', j)
