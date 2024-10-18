# FFT/IFFT Python Wrapper Functions Documentation

## List of Functions

1. [format_array_as_data_string](#format_array_as_data_string)
2. [write_fft_type_to_assembly_file](#write_fft_type_to_assembly_file)
3. [write_array_to_assembly_file](#write_array_to_assembly_file)
4. [find_log_pattern](#find_log_pattern)
5. [simulate_on_veer](#simulate_on_veer)
6. [hex_to_float](#hex_to_float)
7. [find_log_index](#find_log_index)
8. [find_log_pattern_index](#find_log_pattern_index)
9. [process_file](#process_file)
10. [run](#run)
11. [npFFT](#npfft)
12. [npIFFT](#npifft)
13. [nFFT](#nfft)
14. [nIFFT](#nifft)
15. [vFFT](#vfft)
16. [vIFFT](#vifft)
17. [nFFT2](#nfft2)
18. [nIFFT2](#nifft2)
19. [vFFT2](#vfft2)
20. [vIFFT2](#vifft2)
21. [performFFT_IFFT](#performfft_ifft)
22. [compute_FFT_IFFT_with_benchmarks](#compute_fft_ifft_with_benchmarks)
23. [changeVectorSize](#changevectorsize)
24. [saveResultsToCSV](#saveresultstocsv)
25. [loadResultsFromCSV](#loadresultsfromcsv)
26. [benchmark_different_sizes](#benchmark_different_sizes)
27. [performTestsAndSaveResults](#performtestsandsaveresults)

## Function Descriptions

### format_array_as_data_string

Formats a given array into a string for a readable format in an assembly file.

**Parameters:**
- `data`: The array to be formatted
- `num_group_size` (optional): Number of values per group (default: 4)
- `num_per_line` (optional): Number of values per line (default: 32)

**Returns:**
A list of formatted strings.

### write_fft_type_to_assembly_file

Writes the FFT type (FFT or IFFT) to perform in the assembly file text section.

**Parameters:**
- `input_file`: Path to the input assembly file
- `output_file`: Path to the output assembly file
- `fft_type`: Type of FFT to perform ('FFT' or 'IFFT')

**Returns:**
None

### write_array_to_assembly_file

Writes array values to the assembly file data section.

**Parameters:**
- `input_file`: Path to the input assembly file
- `output_file`: Path to the output assembly file
- `real`: Array of real values
- `imag`: Array of imaginary values
- `array_size`: Size of the arrays

**Returns:**
None

### find_log_pattern

Checks for a specific pattern in the buffer to start reading the log.

**Parameters:**
- `lines`: List of lines to search for the pattern

**Returns:**
Boolean indicating if the pattern was found

### simulate_on_veer

Runs assembly code on Veer, saves the log to a file, and returns the CPU cycle count and time taken.

**Parameters:**
- `assembly_file`: Path to the assembly file
- `log_file`: Path to save the log file
- `delete_files` (optional): Whether to delete temporary files (default: True)
- `save_full_log` (optional): Whether to save the full log (default: False)

**Returns:**
Tuple containing (retired_instructions, time_taken)

### hex_to_float

Converts an array of hexadecimal values to floating-point numbers.

**Parameters:**
- `hex_array`: Array of hexadecimal strings

**Returns:**
Array of floating-point numbers

### find_log_index

Finds the index for the markers placed in the assembly code.

**Parameters:**
- `file_path`: Path to the file to search

**Returns:**
Tuple containing (start_index, end_index)

### find_log_pattern_index

Finds the pattern twice in the file and returns line indices.

**Parameters:**
- `lines`: List of lines to search

**Returns:**
List of indices where the pattern was found

### process_file

Reads the log file and extracts real and imaginary float values.

**Parameters:**
- `file_name`: Path to the log file
- `delete_log_files` (optional): Whether to delete log files after processing (default: True)

**Returns:**
Complex numpy array containing the extracted values

### run

Runs the specified FFT/IFFT type and returns the result, cycles, and time taken.

**Parameters:**
- `type`: Type of FFT/IFFT to run ('vFFT', 'vIFFT', 'FFT', 'IFFT', 'vFFT2', 'vIFFT2', 'FFT2', 'IFFT2')
- `real`: Array of real values
- `imag`: Array of imaginary values
- `array_size`: Size of the arrays
- `delete_temp_files` (optional): Whether to delete temporary files (default: True)
- `delete_log_files` (optional): Whether to delete log files (default: True)

**Returns:**
Tuple containing (result, cycles, time)

### npFFT

Calculates FFT using numpy.

**Parameters:**
- `real`: Array of real values
- `imag`: Array of imaginary values
- `_`: Unused parameter (for consistency with other FFT functions)

**Returns:**
Tuple containing (fft_result, cycles, time)

### npIFFT

Calculates IFFT using numpy.

**Parameters:**
- `real`: Array of real values
- `imag`: Array of imaginary values
- `_`: Unused parameter (for consistency with other IFFT functions)

**Returns:**
Tuple containing (ifft_result, cycles, time)

### nFFT

Calculates FFT using RISC-V assembly code simulated on Veer.

**Parameters:**
- `real`: Array of real values
- `imag`: Array of imaginary values
- `array_size`: Size of the arrays
- `deleteFiles` (optional): Whether to delete temporary files (default: True)

**Returns:**
Tuple containing (fft_result, cycles, time)

### nIFFT

Calculates IFFT using RISC-V assembly code simulated on Veer.

**Parameters:**
- `real`: Array of real values
- `imag`: Array of imaginary values
- `array_size`: Size of the arrays
- `deleteFiles` (optional): Whether to delete temporary files (default: True)

**Returns:**
Tuple containing (ifft_result, cycles, time)

### vFFT

Calculates FFT using vectorized RISC-V assembly code simulated on Veer.

**Parameters:**
- `real`: Array of real values
- `imag`: Array of imaginary values
- `array_size`: Size of the arrays
- `deleteFiles` (optional): Whether to delete temporary files (default: True)

**Returns:**
Tuple containing (fft_result, cycles, time)

### vIFFT

Calculates IFFT using vectorized RISC-V assembly code simulated on Veer.

**Parameters:**
- `real`: Array of real values
- `imag`: Array of imaginary values
- `array_size`: Size of the arrays
- `deleteFiles` (optional): Whether to delete temporary files (default: True)

**Returns:**
Tuple containing (ifft_result, cycles, time)

### nFFT2

Calculates FFT using an alternative RISC-V assembly code simulated on Veer.

**Parameters:**
- `real`: Array of real values
- `imag`: Array of imaginary values
- `array_size`: Size of the arrays
- `deleteFiles` (optional): Whether to delete temporary files (default: True)

**Returns:**
Tuple containing (fft_result, cycles, time)

### nIFFT2

Calculates IFFT using an alternative RISC-V assembly code simulated on Veer.

**Parameters:**
- `real`: Array of real values
- `imag`: Array of imaginary values
- `array_size`: Size of the arrays
- `deleteFiles` (optional): Whether to delete temporary files (default: True)

**Returns:**
Tuple containing (ifft_result, cycles, time)

### vFFT2

Calculates FFT using an alternative vectorized RISC-V assembly code simulated on Veer.

**Parameters:**
- `real`: Array of real values
- `imag`: Array of imaginary values
- `array_size`: Size of the arrays
- `deleteFiles` (optional): Whether to delete temporary files (default: True)

**Returns:**
Tuple containing (fft_result, cycles, time)

### vIFFT2

Calculates IFFT using an alternative vectorized RISC-V assembly code simulated on Veer.

**Parameters:**
- `real`: Array of real values
- `imag`: Array of imaginary values
- `array_size`: Size of the arrays
- `deleteFiles` (optional): Whether to delete temporary files (default: True)

**Returns:**
Tuple containing (ifft_result, cycles, time)

### performFFT_IFFT

Helper function to perform FFT and IFFT with given functions and labels.

**Parameters:**
- `fft_func`: Function to perform FFT
- `ifft_func`: Function to perform IFFT
- `label`: Label for the FFT/IFFT type
- `real`: Array of real values
- `imag`: Array of imaginary values
- `array_size`: Size of the arrays

**Returns:**
Dictionary containing FFT and IFFT results, cycles, and times

### compute_FFT_IFFT_with_benchmarks

Performs FFT and IFFT on an array of given size, using either provided or random values.

**Parameters:**
- `array_size`: Size of the array to process
- `real` (optional): Array of real values
- `imag` (optional): Array of imaginary values
- `hardcoded` (optional): Whether to use provided values or generate random ones (default: False)

**Returns:**
Dictionary containing benchmark results for different FFT/IFFT implementations

### changeVectorSize

Changes the Veer vector size to the specified number of bytes.

**Parameters:**
- `size`: New vector size in bytes

**Returns:**
None

### saveResultsToCSV

Saves FFT/IFFT benchmark results to a CSV file.

**Parameters:**
- `results`: List of dictionaries containing size and result data
- `filename`: Name of the CSV file to save to

**Returns:**
None

### loadResultsFromCSV

Loads FFT/IFFT benchmark results from a CSV file.

**Parameters:**
- `filename`: Name of the CSV file to load from

**Returns:**
List of dictionaries containing size and result data

### benchmark_different_sizes

Runs FFT/IFFT on arrays of different sizes.

**Parameters:**
- `sizes`: List of array sizes to benchmark
- `real` (optional): Array of real values
- `imag` (optional): Array of imaginary values
- `hardcoded` (optional): Whether to use provided values or generate random ones (default: False)

**Returns:**
List of dictionaries containing benchmark results for each size

### performTestsAndSaveResults

Performs FFT/IFFT tests on given sizes, checks for existing results, and saves new results to the CSV file.

**Parameters:**
- `sizes`: List of sizes to test
- `filename` (optional): Name of the CSV file to save results to (default: 'fft_ifft_results.csv')
- `real` (optional): Array of real values
- `imag` (optional): Array of imaginary values
- `hardcoded` (optional): Whether to use provided values or generate random ones (default: False)

**Returns:**
List of all results (existing and new)