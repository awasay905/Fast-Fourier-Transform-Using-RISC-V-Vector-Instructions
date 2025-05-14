import numpy as np
import matplotlib.pyplot as plt
from functions import vFFT # This will import from functions.py

# --- Test Configuration ---
MIN_POWER = 1
MAX_POWER = 13    # Test N from 2^1 (2) up to 2^12 (4096)
Fs = 1_000_000  # Sampling frequency in Hz
T = 1 / Fs      # Sampling interval

# Configure which N values to plot for. Empty list [] means no plots.
# Plotting for all N values can generate many plots.
PLOT_FOR_N_VALUES = [] # e.g., only plot for the largest N
#PLOT_FOR_N_VALUES = [2**MAX_POWER] # e.g., only plot for the largest N
# PLOT_FOR_N_VALUES = [2**p for p in range(MIN_POWER, MAX_POWER + 1) if p % 2 == 0] # Plot for some Ns

# Signal parameters (same as your example)
signal_frequencies_hz = [1000, 5000, 20000]
signal_amplitudes     = [0.6, 0.3, 0.1]

# --- Helper Function for Plotting ---
def plot_fft_comparison(N_val, T_val, my_fft_complex, numpy_fft_complex, Fs_val):
    # Normalize magnitudes for plotting (as in your example)
    # Note: Numerical comparison should be done on raw, unnormalized FFT outputs.
    my_magnitude = np.abs(my_fft_complex) / np.sqrt(N_val)
    numpy_magnitude = np.abs(numpy_fft_complex) / np.sqrt(N_val)
    
    frequencies = np.fft.fftfreq(N_val, d=T_val)

    # Determine a reasonable upper frequency limit for plotting
    # Show up to 100 kHz or Nyquist frequency, whichever is smaller
    max_plot_freq = min(Fs_val / 2, 100_000) 
    
    # Get indices for the positive frequency spectrum up to max_plot_freq
    # For real input signals, the negative frequency spectrum is conjugate symmetric.
    plot_indices = np.where((frequencies >= 0) & (frequencies <= max_plot_freq))[0]
    
    # Fallback for very small N or if max_plot_freq is too restrictive
    if len(plot_indices) < 2 and N_val > 1 : # Ensure at least two points if possible
        plot_indices = np.arange(N_val // 2 + 1) # Show up to Nyquist
    elif N_val == 1:
        plot_indices = np.arange(1)


    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.plot(frequencies[plot_indices], my_magnitude[plot_indices])
    plt.title(f"My vFFT Output (N={N_val})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude / sqrt(N)")
    plt.grid(True)
    plt.ylim(bottom=0) # Magnitudes are non-negative

    plt.subplot(1, 2, 2)
    plt.plot(frequencies[plot_indices], numpy_magnitude[plot_indices])
    plt.title(f"NumPy FFT Output (N={N_val})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude / sqrt(N)")
    plt.grid(True)
    plt.ylim(bottom=0)

    plt.suptitle(f"FFT Magnitude Spectrum Comparison (N={N_val})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
    plt.show()

# --- Main Test Loop ---
overall_status = True
test_results_summary = []

print("Starting FFT Implementation Test...")
print("===================================")

for power in range(MIN_POWER, MAX_POWER + 1):
    N = 2**power
    print(f"\n--- Testing for N = {N} (2^{power}) ---")

    # 1. Generate Time Vector and Synthetic Signal
    # Ensure time vector 't' and 'signal' are float32, as in your C code context
    t = (np.arange(N) * T).astype(np.float32)
    signal = np.zeros(N, dtype=np.float32)
    for freq, amp in zip(signal_frequencies_hz, signal_amplitudes):
        signal += amp * np.sin(2 * np.pi * freq * t)
    
    # Prepare inputs for your vFFT: real and imaginary parts
    # Pass copies because vFFT (or its C backend) might modify inputs in-place
    real_input_for_my_fft = signal.copy()
    imag_input_for_my_fft = np.zeros_like(signal, dtype=np.float32)

    # 2. Compute FFT using your vFFT implementation
    my_fft_output_complex = None
    try:
        # Your code calls vFFT(real, imag, N)[0]
        # This implies vFFT returns a tuple, and the first element is the complex result array
        returned_tuple = vFFT(real_input_for_my_fft, imag_input_for_my_fft, N)
        my_fft_output_complex = returned_tuple[0]
        my_fft_output_complex = my_fft_output_complex[:N]

        
        # Basic validation of your FFT's output
        if not isinstance(my_fft_output_complex, np.ndarray):
            raise TypeError(f"vFFT output is not a NumPy array (type: {type(my_fft_output_complex)})")
        if my_fft_output_complex.shape != (N,):
            raise ValueError(f"vFFT output shape is {my_fft_output_complex.shape}, expected ({N},)")
        if my_fft_output_complex.dtype not in [np.complex64, np.complex128]:
            print(f"  Note: vFFT output dtype is {my_fft_output_complex.dtype}. Comparing with complex64 from NumPy.")
        print(f"  My vFFT executed successfully. Output dtype: {my_fft_output_complex.dtype}")

    except Exception as e:
        print(f"  ERROR executing My vFFT for N={N}: {e}")
        test_results_summary.append({"N": N, "status": "ERROR_MY_FFT", "details": str(e)})
        overall_status = False
        if N in PLOT_FOR_N_VALUES: # If plotting was intended, show error plot
            plt.figure()
            plt.title(f"My vFFT FAILED for N={N}")
            plt.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', wrap=True)
            plt.show()
        continue # Skip to next N

    # 3. Compute FFT using NumPy as a reference
    # np.fft.fft on a real float32 array produces a complex64 array
    numpy_fft_output_complex = np.fft.fft(signal)
    print(f"  NumPy FFT executed. Output dtype: {numpy_fft_output_complex.dtype}")

    # 4. Compare the results
    # Ensure my_fft_output_complex is compatible for comparison (e.g. complex64)
    try:
        my_fft_to_compare = my_fft_output_complex.astype(np.complex64, copy=False)
    except: # If cast fails for some reason
        my_fft_to_compare = my_fft_output_complex


    # Define tolerances for np.allclose. These might need adjustment based on your vFFT's precision.
    # For float32 operations, atol=1e-5 or 1e-6 is often reasonable. rtol is relative tolerance.
    rtol_val = 1e-4 
    atol_val = 1e-4 
    
    are_results_close = np.allclose(my_fft_to_compare, numpy_fft_output_complex, rtol=rtol_val, atol=atol_val)
    
    max_abs_difference = np.max(np.abs(my_fft_to_compare - numpy_fft_output_complex))
    mean_squared_error = np.mean(np.abs(my_fft_to_compare - numpy_fft_output_complex)**2)

    if are_results_close:
        status = "PASSED"
        print(f"  {status}: Results are close for N={N}.")
    else:
        status = "FAILED"
        print(f"  {status}: Results significantly differ for N={N}.")
        overall_status = False
    
    print(f"    Max Absolute Difference: {max_abs_difference:.3e}")
    print(f"    Mean Squared Error: {mean_squared_error:.3e}")
    test_results_summary.append({
        "N": N, "status": status, 
        "max_abs_diff": max_abs_difference, "mse": mean_squared_error
    })

    # 5. Plot if N is in the list for plotting
    if N in PLOT_FOR_N_VALUES:
        print(f"  Generating plot for N={N}...")
        plot_fft_comparison(N, T, my_fft_output_complex, numpy_fft_output_complex, Fs)

# --- Final Summary ---
print("\n===================================")
print("Test Run Summary:")
print("===================================")
for res in test_results_summary:
    if res['status'] not in ["PASSED", "ERROR_MY_FFT"]:
        print(f"N={res['N']:<5}: {res['status']:<15} MaxAbsDiff={res['max_abs_diff']:.2e}, MSE={res['mse']:.2e}")
    elif res['status'] == "ERROR_MY_FFT":
         print(f"N={res['N']:<5}: {res['status']:<15} Details: {res['details']}")
    else: # PASSED
        print(f"N={res['N']:<5}: {res['status']:<15} MaxAbsDiff={res['max_abs_diff']:.2e}, MSE={res['mse']:.2e}")


if overall_status:
    print("\nSUCCESS: All tests where vFFT executed passed or results were close within tolerance.")
else:
    print("\nWARNING: Some tests failed or vFFT encountered errors. Please review the output.")