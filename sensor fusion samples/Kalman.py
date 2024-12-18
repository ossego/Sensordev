import numpy as np
import matplotlib.pyplot as plt

# Define sensor models
def measure_position(x, noise_std):
    return x + np.random.normal(0, noise_std)

# Simulate sensor data
dt = 0.1
time = np.arange(0, 10, dt)
print(time)
true_position = np.sin(time)
true_position2= np.cos(time)
sensor1_data = measure_position(true_position, 0.5)
sensor2_data = measure_position(true_position2, 0.8)

# Implement Kalman Filter
x = np.array([[0], [0]])  # Initial state (position, velocity)
P = np.eye(2)  # Initial covariance matrix
Q = np.array([[0.01, 0], [0, 0.01]])  # Process noise covariance
R = np.array([[0.25, 0], [0, 0.64]])  # Measurement noise covariance
F = np.array([[1, dt], [0, 1]])  # State transition matrix
H = np.array([[1, 0], [1, 0]])  # Measurement matrix

fused_position = []
for i in range(len(time)):
    # Prediction  (Projection into K+1  - fig 11.1 https://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf)
    x = F @ x
    P = F @ P @ F.T + Q

    # Update
    z = np.array([[sensor1_data[i]], [sensor2_data[i]]])
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)  #Kalman Gain
    x = x + K @ (z - H @ x)  #Update Estimate
    P = (np.eye(2) - K @ H) @ P  #Update Covariance

    fused_position.append(x[0, 0])

# Evaluate performance
plt.plot(time, true_position, label='True Position')
plt.plot(time, sensor1_data, label='Sensor 1')
plt.plot(time, sensor2_data, label='Sensor 2')
plt.plot(time, fused_position, label='Fused Position')
plt.legend()
plt.show()
