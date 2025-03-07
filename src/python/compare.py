from functions import *
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 1024  # Number of points

# Load CSV file
loaded_data = np.loadtxt("signal_data.csv", delimiter=",", skiprows=1)

# Extract time and signal separately
t = loaded_data[:N, 0]
signal = loaded_data[:N, 1]

# Convert to float array
real = np.array(signal, dtype=np.float32)
imag = np.zeros_like(signal)

# Compute FFT
npFFt = np.fft.fft(real)# NumPy FFT
fft_result = vFFT(real, imag, N)[0] # Your FFT implementation
# Compute magnitude spectrum
np_magnitude = np.abs(npFFt)/ np.sqrt(N)
my_magnitude = np.abs(fft_result)/ np.sqrt(N)
frequencies = np.fft.fftfreq(N)  # Frequency axis

# Plot FFT magnitude spectrum (side by side)
plt.figure(figsize=(12, 5))

# Your FFT plot
plt.subplot(1, 2, 1)
plt.plot(frequencies[:N//2], my_magnitude[:N//2])
plt.title("My FFT Magnitude Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()

# NumPy FFT plot
plt.subplot(1, 2, 2)
plt.plot(frequencies[:N//2], np_magnitude[:N//2])
plt.title("NumPy FFT Magnitude Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()

plt.tight_layout()
plt.show()
