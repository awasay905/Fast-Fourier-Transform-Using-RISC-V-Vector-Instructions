import pickle
import numpy as np
VEER_TEMP_FOLDER_PATH = './veer/tempFiles'
VEER_FOLDER_PATH = './veer'
TEST_TEMP_FOLDER_PATH = './src/assemblyForPython/tempFiles'
TEST_CODE_FOLDER_PATH = './src/assemblyForPython'
RESULT_FOLDER_PATH = './results/data'

# Formats given array to string for a readable format for assembly file


def format_array_as_data_string(data, num_group_size=4, num_per_line=32):
    formatted_lines = []
    current_line = ".float "
    for i, value in enumerate(data):
        current_line += f"{value:.12f}, "
        if (i + 1) % num_group_size == 0:  # Add space after every (num_group_size)th number
            current_line += " "
        if (i + 1) % num_per_line == 0:  # New line after every (num_per_line) numbers
            formatted_lines.append(current_line.strip(", "))
            current_line = ".float "
    # Add remaining line if not exactly multiple of (num_per_line)
    if current_line.strip(", "):
        formatted_lines.append(current_line.strip(", "))
    return formatted_lines

# Write which type (FFT or IFFT) to perform to assembly file text section


def write_fft_type_to_assembly_file(input_file, output_file, fft_type):
    # Read the input file
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Insert the fft/ifft after finding call XXXX in assembly file
    for i, line in enumerate(lines):
        if "call" in line and "XXXX" in line:
            lines[lines.index(line)] = f"    call {fft_type}"
            break

    # Write the modified content to a new file
    with open(output_file, 'w') as file:
        file.writelines(lines)

    return

# Write array values to the assembly file data section.


def write_array_to_assembly_file(input_file, output_file, real, imag, array_size):
    # Read the input file and find the .data section
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Insert the array values after .data
    for i, line in enumerate(lines):
        if ".data" in line:
            # Insert real array
            lines.insert(i + 1, "real:\n")
            real_lines = format_array_as_data_string(real)
            for j, real_line in enumerate(real_lines):
                lines.insert(i + 2 + j, real_line + "\n")

            # Insert a blank line
            lines.insert(i + 2 + len(real_lines), "\n")

            # Insert imag array
            lines.insert(i + 3 + len(real_lines), "imag:\n")
            imag_lines = format_array_as_data_string(imag)
            for j, imag_line in enumerate(imag_lines):
                lines.insert(i + 4 + len(real_lines) + j, imag_line + "\n")

            # Insert a blank line and data size declaration
            lines.insert(i + 4 + len(real_lines) + len(imag_lines), "\n")
            lines.insert(i + 5 + len(real_lines) + len(imag_lines),
                         f".set dataSize, {array_size}\n")
            break

    # Write the modified content to a new file
    with open(output_file, 'w') as file:
        file.writelines(lines)

    return

# Function to check for the pattern in the buffer to start reading log


def find_log_pattern(lines):
    # Split each line and check if the 7th column has the desired values
    required_values = ["00123000", "00123456",
                       "00234000", "00234567", "00345000", "00345678"]
    found_values = []

    for line in lines:
        columns = line.split()
        if len(columns) > 6:  # Check if there are enough columns
            value = columns[6]  # Get the 7th column (index 6)
            if value in required_values:
                found_values.append(value)

    # Check if we found all required values
    return all(value in found_values for value in required_values)

# Runs assembly code on Veer, saves the log to logFile if deletefiles false,
# and returns the cpu cycle count and time taken
# Requires file to have eevrything set in it


def simulate_on_veer(assembly_file, log_file, delete_files=True, save_full_log=False):
    import subprocess as sp
    import re
    import time
    GCC_PREFIX = "riscv32-unknown-elf"
    ABI = "-march=rv32gcv_zbb_zbs -mabi=ilp32f"
    LINK = f"{VEER_FOLDER_PATH}/link.ld"
    # removes .s  and gets file name from the file path
    fileName = assembly_file[assembly_file.rfind('/') + 1:-2]
    tempPath = f"{TEST_TEMP_FOLDER_PATH}/{fileName}"
    timetaken = 0

    # Commands to run
    commands = [
        f"{GCC_PREFIX}-gcc {ABI} -lgcc -T{LINK} -o {tempPath}.exe {assembly_file} -nostartfiles -lm",
        # Delete assembly code after translation if deleteFiles is True
        f"rm -f {assembly_file}" if delete_files else "",
        f"{GCC_PREFIX}-objcopy -O verilog {tempPath}.exe {tempPath}.hex",
        # Remove executable if deleteFiles is True
        f"rm -f {tempPath}.exe" if delete_files else "",
        # f"{GCC_PREFIX}-objdump -S {tempPath}.exe > {tempPath}.dis" if not deleteFiles else f"rm -f {tempPath}.dis",  # Optional disassembly
        f"whisper -x {tempPath}.hex -s 0x80000000 --tohost 0xd0580000 -f /dev/stdout --configfile {VEER_FOLDER_PATH}/whisper.json" if not save_full_log
        else f"whisper -x {tempPath}.hex -s 0x80000000 --tohost 0xd0580000 -f {log_file} --configfile {VEER_FOLDER_PATH}/whisper.json",
        # Delete hex file after translation if deleteFiles is True
        f"rm -f {tempPath}.hex" if delete_files else "",
    ]

    # Remove any empty strings from the commands
    commands = [cmd for cmd in commands if cmd]  # Filter out empty strings

    retired_instructions = None  # Variable to store the number of retired instructions

    # Regular expression to find "Retired X instructions"
    instruction_regex = r"Retired\s+(\d+)\s+instructions"

    # Execute the commands one by one
    start_time = -1
    end_time = -1
    measure_stop = False
    for command in commands:
        try:
            if not measure_stop:
                start_time = time.time()
            if "whisper" in command and not save_full_log:
                # redirect output. once for capturing time , twice for reading logs
                process = sp.Popen(command, shell=True,
                                   stdout=sp.PIPE, stderr=sp.PIPE, text=True)

                buffer_size = 10
                lines_buffer = []
                recorded_log = []
                pattern_found = False

                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        if not measure_stop:
                            end_time = time.time()
                            measure_stop = True
                        break  # Exit if the process is done

                    if output:  # Process only if there's output
                        # Add the output to the buffer
                        lines_buffer.append(output)
                        if len(lines_buffer) > buffer_size:
                            # Maintain only the last 10 lines
                            lines_buffer.pop(0)

                        # Check for the pattern in the 7th column
                        if not pattern_found:
                            if find_log_pattern(lines_buffer):
                                pattern_found = True
                                recorded_log.extend(lines_buffer)

                        # If pattern found, record all lines after it
                        if pattern_found:
                            recorded_log.append(output)

                with open(log_file, 'w') as file:
                    file.writelines(recorded_log)

                stderr_output = process.stderr.read()
                if stderr_output:
                    # Search for the "Retired X instructions" pattern in the stderr
                    match = re.search(instruction_regex, stderr_output)
                    if match:
                        retired_instructions = match.group(
                            1)  # Extract the number

            else:
                result = sp.run(command, capture_output=True,
                                shell=True, text=True)
                if not measure_stop:
                    end_time = time.time()
                    if 'whisper' in command:
                        measure_stop = True
                if result.stderr:
                    # Search for the "Retired X instructions" pattern in the stderr
                    match = re.search(instruction_regex, result.stderr)
                    if match:
                        retired_instructions = match.group(
                            1)  # Extract the number

        except sp.CalledProcessError as e:
            print(f"An error {e}occurred while executing: {command}")
            print(f"Error: {e}")
            exit(-1)

    timetaken = end_time - start_time

    return int(retired_instructions), timetaken

# Convert hex values array to float array. HEX should be IEEE format


def hex_to_float(hex_array):
    import struct
    float_array = []

    for hex_str in hex_array:
        # Ensure the hex string is exactly 8 characters long
        if len(hex_str) != 8:
            raise ValueError(
                f"Hex string '{hex_str}' is not 8 characters long")

        # Convert the hex string to a 32-bit integer
        int_val = int(hex_str, 16)

        # Pack the integer as a 32-bit unsigned integer
        packed_val = struct.pack('>I', int_val)

        # Unpack as a float (IEEE 754)
        float_val = struct.unpack('>f', packed_val)[0]

        float_array.append(float_val)

    return float_array

# Function to find the pattern twice in the file and return line indices


def find_log_pattern_index(file_name):

    with open(file_name, 'r') as file:
        lines = file.readlines()

    # List of required values in the 7th column
    required_values = ["00123000", "00123456",
                       "00234000", "00234567", "00345000", "00345678"]
    found_values = []  # List to track when we find the required values
    pattern_indices = []  # To store the line index when the full pattern is found
    current_pattern_start = None  # To store where a potential pattern starts

    for i, line in enumerate(lines):
        columns = line.split()
        if len(columns) > 6:  # Check if there are enough columns
            value = columns[6]  # Get the 7th column (index 6)
            if value in required_values:
                if current_pattern_start is None and value == required_values[0]:
                    current_pattern_start = i  # Start tracking pattern from this line
                found_values.append(value)

            # If we found all the required values, save the index and reset
            if all(val in found_values for val in required_values):
                pattern_indices.append(current_pattern_start)
                found_values = []  # Reset for the next pattern
                current_pattern_start = None  # Reset pattern start

            # If we've found the pattern twice, we can stop
            if len(pattern_indices) == 2:
                break

    # Return the line indices where the pattern was found twice
    return pattern_indices

# Reads log file and extract real and imag float values


def process_file(file_name, delete_log_files=False):
    import numpy as np
    start_index, end_index = find_log_pattern_index(file_name)
    real = []
    imag = []

    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()

        if delete_log_files:
            import os
            os.remove(file_name)

        # Ensure start and end indexes are within the valid range
        start_index = max(0, start_index)
        end_index = min(len(lines), end_index)

        # Initialize a flag to alternate between real and imag
        save_to_real = True
        is_vectorized = False

        # Process lines within the specified range
        for i in range(start_index, end_index):
            if not is_vectorized:
                if "vsetvli" in lines[i]:
                    is_vectorized = True
                    continue

                if "c.flw" in lines[i] or "flw" in lines[i]:
                    words = lines[i].split()
                    if len(words) > 1:
                        if "c.flw" in lines[i]:
                            index_of_cflw = words.index("c.flw")
                        else:
                            index_of_cflw = words.index("flw")
                        if index_of_cflw > 0:
                            if save_to_real:
                                real.append(words[index_of_cflw - 1])
                                save_to_real = False
                            else:
                                imag.append(words[index_of_cflw - 1])
                                save_to_real = True

            else:
                if "vle32.v" in lines[i]:
                    words = lines[i].split()
                    index_of_cflw = words.index("vle32.v")
                    if save_to_real:
                        real.append(words[index_of_cflw - 1])
                        save_to_real = False
                    else:
                        imag.append(words[index_of_cflw - 1])
                        save_to_real = True

        # return hex_to_float(real), hex_to_float(imag)
        if (is_vectorized):
            realVal = []
            imagVal = []

            for i in range(len(real)):
                realVector = real[i]
                imagVector = imag[i]

                # split the strings into 8 bit chunks
                realVector = [realVector[i:i+8]
                              for i in range(0, len(realVector), 8)]
                imagVector = [imagVector[i:i+8]
                              for i in range(0, len(imagVector), 8)]

                # reverse the order of the chunks
                realVector = realVector[::-1]
                imagVector = imagVector[::-1]

                realVal.extend(realVector)
                imagVal.extend(imagVector)

            real = realVal
            imag = imagVal
        return np.array(hex_to_float(real)) + 1j * np.array(hex_to_float(imag))

    except FileNotFoundError:
        print(f"The file {file_name} does not exist.")
        return real, imag

# runs "type", returning the result, cycles and time taken used in a tuple


def run(type, real, imag, array_size, delete_temp_files=True, delete_log_files=False):
    import os
    import numpy as np

    if not os.path.exists(TEST_TEMP_FOLDER_PATH):
        os.mkdir(TEST_TEMP_FOLDER_PATH)

    assemblyFile = f"{TEST_TEMP_FOLDER_PATH}/temp{type}.s"
    logFile = f"{TEST_TEMP_FOLDER_PATH}/temp{type}log.txt"
    if type == 'vFFT' or type == 'vIFFT':
        write_array_to_assembly_file(
            f"{TEST_CODE_FOLDER_PATH}/vFFTforPython.s", assemblyFile, real, imag, array_size)
        write_fft_type_to_assembly_file(assemblyFile, assemblyFile, type)
    elif type == 'FFT' or type == 'IFFT':
        write_array_to_assembly_file(
            f"{TEST_CODE_FOLDER_PATH}/FFTforPython.s", assemblyFile, real, imag, array_size)
        write_fft_type_to_assembly_file(assemblyFile, assemblyFile, type)
    else:
        print("ERROR")
        exit(-1)

    cycles, time = simulate_on_veer(assemblyFile, logFile, delete_temp_files)
    result = process_file(logFile, delete_log_files)

    return (result, cycles, time)

# Calculate FFT using numpy, returns the FFT, cycles and the time taken to calculate


def npFFT(real, imag, _):
    import time
    import numpy as np
    complex_numbers = np.array(real) + 1j * np.array(imag)
    start_time = time.time()
    fft = np.fft.fft(complex_numbers)
    end_time = time.time()
    elapsed_time = end_time - start_time
    npFFTcycles = -1  # implement later
    return fft, npFFTcycles, elapsed_time

# Calculate IFFT using numpy, returns the IFFT, cycles and the time taken to calculate


def npIFFT(real, imag, _):
    import time
    import numpy as np
    complex_numbers = np.array(real) + 1j * np.array(imag)
    start_time = time.time()
    ifft = np.fft.ifft(complex_numbers)
    end_time = time.time()
    elapsed_time = end_time - start_time
    npIFFTcycles = -1  # implement later
    return ifft, npIFFTcycles, elapsed_time

# Calculaye FFT using risc v assembly code simulated on veer


def nFFT(real, imag, array_size, deleteFiles=True):
    return run('FFT', real, imag, array_size, deleteFiles, False)

# Calculaye IFFT using risc v assembly code simulated on veer


def nIFFT(real, imag, array_size, deleteFiles=True):
    return run('IFFT', real, imag, array_size, deleteFiles, False)

# Calculaye FFT using vectorized risc v assembly code simulated on veer


def vFFT(real, imag, array_size, deleteFiles=True):
    return run('vFFT', real, imag, array_size, deleteFiles, False)

# Calculaye IFFT using vecctorized risc v assembly code simulated on veer


def vIFFT(real, imag, array_size, deleteFiles=True):
    return run('vIFFT', real, imag, array_size, deleteFiles, False)


# Performs FFT and IFFT on array of n size, of real and imag. if hardcoded if flase then random floats
# from -1000 to 1000 will be used
# Returns FFT, IFFT and time taken in performing them on numpy, riscv, and nevctorized risc v
def compute_FFT_IFFT_with_benchmarks(array_size, real=[], imag=[], hardcoded=False):
    if not hardcoded:
        import random
        import numpy as np
        real = [random.uniform(-1000, 1000) for _ in range(array_size)]
        imag = [random.uniform(-1000, 1000) for _ in range(array_size)]
        
        # Combine into a complex array
        complex_array = np.array([complex(r, i) for r, i in zip(real, imag)], dtype=np.complex64)

        # Extract real and imaginary parts as 32-bit floats
        real = complex_array.real.astype(np.float32)
        imag = complex_array.imag.astype(np.float32)

    print(f"\n\nStarting benchmark for {array_size}")

    # Dictionary to store results
    benchmark_results = {}

    print(f"Performing npFFT for array of size {array_size}")
    npFFTresult, npFFTcycles, npFFTtime = npFFT(real, imag, array_size)
    print(f"Done. Took {npFFTtime} milliseconds")

    print(f"Performing npIFFT for array of size {array_size}")
    npIFFTresult, npIFFTcycles, npIFFTtime = npIFFT(
        npFFTresult.real, npFFTresult.imag, array_size)
    print(f"Done. Took {npIFFTtime} milliseconds")

    print(f"Performing nFFT for array of size {array_size}")
    nFFTresult, nFFTcycles, nFFTtime = nFFT(real, imag, array_size)
    print(f"Done. Took {nFFTtime} milliseconds")

    print(f"Performing nIFFT for array of size {array_size}")
    nIFFTresult, nIFFTcycles, nIFFTtime = nIFFT(
        nFFTresult.real, nFFTresult.imag, array_size)
    print(f"Done. Took {nIFFTtime} milliseconds")

    print(f"Performing vFFT for array of size {array_size}")
    vFFTresult, vFFTcycles, vFFTtime = vFFT(real, imag, array_size)
    print(f"Done. Took {vFFTtime} milliseconds")

    print(f"Performing vIFFT for array of size {array_size}")
    vIFFTresult, vIFFTcycles, vIFFTtime = vIFFT(
        vFFTresult.real, vFFTresult.imag, array_size)
    print(f"Done. Took {vIFFTtime} milliseconds")

    import numpy as np
    benchmark_results['size'] = array_size
    benchmark_results['input'] = np.array(real) + 1j * np.array(imag)

    benchmark_results['npFFT'] = {
        'result': npFFTresult.astype(np.complex64),
        'cycles': npFFTcycles,
        'time': npFFTtime
    }

    benchmark_results['npIFFT'] = {
        'result': npIFFTresult.astype(np.complex64),
        'cycles': npIFFTcycles,
        'time': npIFFTtime
    }

    benchmark_results['nFFT'] = {
        'result': nFFTresult.astype(np.complex64),
        'cycles': nFFTcycles,
        'time': nFFTtime
    }

    benchmark_results['nIFFT'] = {
        'result': nIFFTresult.astype(np.complex64),
        'cycles': nIFFTcycles,
        'time': nIFFTtime
    }

    benchmark_results['vFFT'] = {
        'result': vFFTresult[:array_size].astype(np.complex64),
        'cycles': vFFTcycles,
        'time': vFFTtime
    }

    benchmark_results['vIFFT'] = {
        'result': vIFFTresult[:array_size].astype(np.complex64),
        'cycles': vIFFTcycles,
        'time': vIFFTtime
    }

    print(f"\n\nAll benchmarks done for {array_size}\n\n")
    return benchmark_results

# Changes Veer vector size to number of bytes


def changeVectorSize(size):
    import json
    file_path = VEER_FOLDER_PATH+'/whisper.json'

    with open(file_path, 'r') as file:
        data = json.load(file)

    data["vector"]["bytes_per_vec"] = size

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

    return


def saveResults(results, filename):
    with open(filename+'.pickle', 'wb') as f:
        pickle.dump(results, f)


def loadResults(filename):
    with open(filename+'.pickle', 'rb') as f:
        return pickle.load(f)

# RUNS FFT/IFFT on arrays of different sizes on dirrent real/imag array (pass array counraninf array) (if hardcodedgiven)).
# TODO custom array values are not implemented yet
# Returns an array conatinign output of each test


def benchmark_different_sizes(sizes, real=[], imag=[], hardcoded=False):
    results = []
    for size in sizes:
        result = compute_FFT_IFFT_with_benchmarks(size, real, imag, hardcoded)
        results.append(result)

    return results

# Perform FFT/IFFT tests on given sizes, check for existing results,
#    and save new results to the CSV file.
#    :param sizes: List of sizes to test.
#    :param filename: Name of the CSV file to save results to.
#    :param real: Real part of the input array.
#    :param imag: Imaginary part of the input array.
#    :param hardcoded: Whether to use hardcoded values or random values.


def performTestsAndSaveResults(sizes, filename=f"{RESULT_FOLDER_PATH}/fft_ifft_results", real=[], imag=[], hardcoded=False):
    import os
    # Load previous results if the CSV file exists
    existing_results = []
    if os.path.exists(filename+'.pickle'):
        existing_results = loadResults(filename)
        print(f"Loaded existing results from {filename}")

    # Gather tested sizes
    tested_sizes = {result['size'] for result in existing_results}

    # Identify sizes that need testing
    sizes_to_test = [size for size in sizes if size not in tested_sizes]

    if not sizes_to_test:
        print("All sizes have already been tested. No new tests will be performed.")
        return existing_results  # Return existing results if no new tests are needed

    # Perform tests on remaining sizes
    new_results = benchmark_different_sizes(
        sizes_to_test, real, imag, hardcoded)

    # Combine existing results with new results
    all_results = existing_results + new_results

    # Save all results to CSV
    saveResults(all_results, filename)
    print(f"Results saved to {filename}")

    return all_results


def flatten_results(results):
    import numpy as np
    data = {
        'size': [],
        'input': [],
        'npFFT_result': [],
        'npFFT_cycles': [],
        'npFFT_time': [],
        'npIFFT_result': [],
        'npIFFT_cycles': [],
        'npIFFT_time': [],
        'nFFT_result': [],
        'nFFT_cycles': [],
        'nFFT_time': [],
        'nIFFT_result': [],
        'nIFFT_cycles': [],
        'nIFFT_time': [],
        'vFFT_result': [],
        'vFFT_cycles': [],
        'vFFT_time': [],
        'vIFFT_result': [],
        'vIFFT_cycles': [],
        'vIFFT_time': [],
    }

# Flatten the data
    for result in results:
        data['size'].append(result['size'])
        data['input'].append(result['input'])

        # npFFT
        data['npFFT_result'].append((result['npFFT']['result']))
        data['npFFT_cycles'].append(result['npFFT']['cycles'])
        data['npFFT_time'].append(result['npFFT']['time'])

        # npIFFT
        data['npIFFT_result'].append((result['npIFFT']['result']))
        data['npIFFT_cycles'].append(result['npIFFT']['cycles'])
        data['npIFFT_time'].append(result['npIFFT']['time'])

        # nFFT
        data['nFFT_result'].append((result['nFFT']['result']))
        data['nFFT_cycles'].append(result['nFFT']['cycles'])
        data['nFFT_time'].append(result['nFFT']['time'])

        # nIFFT
        data['nIFFT_result'].append((result['nIFFT']['result']))
        data['nIFFT_cycles'].append(result['nIFFT']['cycles'])
        data['nIFFT_time'].append(result['nIFFT']['time'])

        # vFFT
        data['vFFT_result'].append((result['vFFT']['result']))
        data['vFFT_cycles'].append(result['vFFT']['cycles'])
        data['vFFT_time'].append(result['vFFT']['time'])

        # vIFFT
        data['vIFFT_result'].append((result['vIFFT']['result']))
        data['vIFFT_cycles'].append(result['vIFFT']['cycles'])
        data['vIFFT_time'].append(result['vIFFT']['time'])


    return (data)
