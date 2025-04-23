from functions import *
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 2**23  # 33,554,432 points
Fs = 1_000_000  # Sampling frequency in Hz
T = 1 / Fs      # Sampling interval

# Time vector
t = np.arange(N) * T

# Generate synthetic signal (sum of sine waves)
signal = (
    0.6 * np.sin(2 * np.pi * 1000 * t) +  # 1 kHz
    0.3 * np.sin(2 * np.pi * 5000 * t) +  # 5 kHz
    0.1 * np.sin(2 * np.pi * 20000 * t)   # 20 kHz
).astype(np.float32)

# Prepare real and imaginary parts
real = signal
imag = np.zeros_like(signal)

# Compute FFT
npFFt = np.fft.fft(real)
fft_result = vFFT(real, imag, N)[0]  # Your FFT implementation

# Compute magnitude spectrum
np_magnitude = np.abs(npFFt) / np.sqrt(N)
my_magnitude = np.abs(fft_result) / np.sqrt(N)
frequencies = np.fft.fftfreq(N, d=T)

# Plot FFT magnitude spectrum (first 100 kHz for clarity)
max_plot_freq = 100_000
max_index = np.argmax(frequencies > max_plot_freq)

plt.figure(figsize=(12, 5))

# Your FFT plot
plt.subplot(1, 2, 1)
plt.plot(frequencies[:max_index], my_magnitude[:max_index])
plt.title("My FFT Magnitude Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()

# NumPy FFT plot
plt.subplot(1, 2, 2)
plt.plot(frequencies[:max_index], np_magnitude[:max_index])
plt.title("NumPy FFT Magnitude Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()

plt.tight_layout()
plt.show()
