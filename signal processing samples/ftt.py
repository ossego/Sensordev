import numpy as np
n = np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))
print(n)