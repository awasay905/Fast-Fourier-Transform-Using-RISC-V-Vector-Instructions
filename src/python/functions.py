from numpy.typing import NDArray
from typing import TypedDict, Dict
import pickle
import numpy as np
VEER_TEMP_FOLDER_PATH = './veer/tempFiles'
VEER_FOLDER_PATH = './veer'
TEST_TEMP_FOLDER_PATH = './src/assemblyForPython/tempFiles'
TEST_CODE_FOLDER_PATH = './src/assemblyForPython'
RESULT_FOLDER_PATH = './results/data'


def format_array_as_data_string(data: list[float], num_group_size: int = 4, num_per_line: int = 32) -> list[str]:
    """
    Format a list of floats into assembly directives for a `.float` statement.

    Each float is formatted to 12 decimal places and grouped with extra spacing.

    Parameters
    ----------
    data : list of float
        The list of floating-point numbers to format.
    num_group_size : int, optional
        Number of floats to group together (default is 4).
    num_per_line : int, optional
        Number of floats per line before inserting a newline (default is 32).

    Returns
    -------
    list of str
        A list of strings, each representing a line of assembly `.float` directives.
    """
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


def write_fft_type_to_assembly_file(input_file: str, output_file: str, fft_type: str) -> None:
    """
    Modify an assembly file to specify the FFT operation type.

    This function searches for a line containing both "call" and "XXXX" in the input file
    and replaces it with a call to the specified FFT type (e.g., "fft" or "ifft").

    Parameters
    ----------
    input_file : str
        Path to the input assembly file.
    output_file : str
        Path where the modified assembly file will be saved.
    fft_type : str
        The FFT operation type to insert (e.g., "fft" or "ifft").

    Returns
    -------
    None
    """
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


def write_array_to_assembly_file(input_file: str, output_file: str, real: list[int], imag: list[int], array_size: int) -> None:
    """
    Insert real and imaginary data arrays into the `.data` section of an assembly file.

    The function locates the `.data` section in the input file and inserts the provided real and
    imaginary arrays (formatted as assembly `.float` directives) along with a data size declaration.

    Parameters
    ----------
    input_file : str
        Path to the input assembly file.
    output_file : str
        Path where the modified assembly file will be saved.
    real : list of int
        List of real-number values.
    imag : list of int
        List of imaginary-number values.
    array_size : int
        Size of the data array.

    Returns
    -------
    None
    """
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


def find_log_pattern(lines: list[str]) -> bool:
    """
    Check for the presence of required hexadecimal log patterns in the 7th column of the provided lines.

    The required values are:
    "00123000", "00123456", "00234000", "00234567", "00345000", "00345678".

    Parameters
    ----------
    lines : list of str
        List of log file lines to search.

    Returns
    -------
    bool
        True if all required values are found in the 7th column, False otherwise.
    """
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


def simulate_on_veer(assembly_file: str, log_file: str, delete_files: bool = False, save_full_log: bool = False) -> tuple[int, float, tuple[dict[str, int], dict[str, int]]]:
    """
    Compile and run an assembly file on the Veer RISC-V simulator, capturing execution logs and performance metrics.

    This function builds and executes a series of commands to compile the assembly file, convert it,
    and run a simulation using the 'whisper' tool. It then parses the output to determine the number
    of retired instructions and the execution time.

    Parameters
    ----------
    assembly_file : str
        Path to the assembly source file.
    log_file : str
        Path to save the execution log.
    delete_files : bool, optional
        If True, intermediate files are deleted after execution (default is True).
    save_full_log : bool, optional
        If True, saves the full log output; otherwise, extracts relevant log portions (default is False).

    Returns
    -------
    tuple
        A tuple containing:
        - An integer representing the number of retired instructions.
        - A float representing the execution time in seconds.

    Raises
    ------
    subprocess.CalledProcessError
        If any of the subprocess commands fail.
    """
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

    # To count the amount of time each instruction was executed
    vector_instructions = {}
    non_vector_instructions = {}

    # Execute the commands one by one
    start_time = -1
    end_time = -1
    measure_stop = False
    for command in commands:
        print(command)
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
                        # Save instruction count
                        parts = output.split()
                        if len(parts) >= 8:
                            # Extracting the 8th column as the instruction
                            instr = parts[7]
                            if instr.startswith('v'):
                                vector_instructions[instr] = vector_instructions.get(
                                    instr, 0) + 1
                            else:
                                non_vector_instructions[instr] = non_vector_instructions.get(
                                    instr, 0) + 1
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
                    print(stderr_output)

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
    print(retired_instructions)
    return int(retired_instructions), timetaken, (vector_instructions, non_vector_instructions)


def hex_to_float(hex_array: list[str]) -> list[float]:
    """
    Convert an array of IEEE 754 single-precision hexadecimal strings to floating-point numbers.

    Parameters
    ----------
    hex_array : list of str
        A list of 8-character hexadecimal strings representing IEEE 754 floats.

    Returns
    -------
    list of float
        A list of floating-point numbers corresponding to the input hex values.

    Raises
    ------
    ValueError
        If any hex string is not exactly 8 characters long.
    """
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


def parse_log_lines(lines):
    merged_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i].rstrip()
        parts = line.split()
        
        # Check if this is a continuation line (ends with +)
        if line.endswith("+") or "+\n" in lines[i]:
            # Get the prefix (first 6 columns)
            if len(parts) >= 7:
                prefix = " ".join(parts[:6])
                hex_value = parts[6].rstrip("+")
                
                # Process continuation lines
                j = i + 1
                while j < len(lines):
                    cont_parts = lines[j].rstrip().split()
                    if len(cont_parts) >= 7:
                        # Prepend the new hex value to the current hex value
                        new_hex = cont_parts[6].rstrip("+")
                        hex_value = new_hex + hex_value
                    
                    # If this line doesn't end with +, break after processing it
                    if not lines[j].endswith("+") and "+\n" not in lines[j]:
                        break
                    
                    j += 1
                
                # Include any remaining content from original line (after the 7th column)
                remaining_content = ""
                if len(parts) > 7:
                    remaining_content = " " + " ".join(parts[7:])
                
                # Create merged line and add to results
                merged_line = f"{prefix} {hex_value}{remaining_content}"
                merged_lines.append(merged_line)
                
                # Move index to after the last processed line
                i = j + 1
            else:
                # Handle malformed lines
                merged_lines.append(line)
                i += 1
        else:
            # For normal lines, just add them as is
            merged_lines.append(line)
            i += 1
    
    return merged_lines



def find_log_pattern_index(file_name: str) -> list[int]:
    """
    Locate the starting line indices where a specific hexadecimal log pattern occurs twice in a file.

    The function searches for the pattern defined by a set of required hexadecimal values.

    Parameters
    ----------
    file_name : str
        Path to the log file.

    Returns
    -------
    list of int
        A list containing up to two line indices where the pattern starts.
    """
    with open(file_name, 'r') as file:
        lines = file.readlines()
        lines = parse_log_lines(lines)

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


def process_file(file_name: str, delete_log_files: bool = False) -> np.ndarray:
    """
    Process a log file to extract complex numbers represented by separate real and imaginary hex strings.

    The function determines the start and end indices based on log patterns, extracts the real and imaginary
    components from the log, and converts them into floating-point numbers. For vectorized data, the strings
    are split into 8-character chunks (in reverse order) before conversion.

    Parameters
    ----------
    file_name : str
        Path to the log file.
    delete_log_files : bool, optional
        If True, deletes the log file after processing (default is False).

    Returns
    -------
    np.ndarray
        A NumPy array of complex numbers constructed from the extracted real and imaginary parts.
    """
    import numpy as np
    start_index, end_index = find_log_pattern_index(file_name)
    print(start_index, end_index)
    real = []
    imag = []

    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()
            lines = parse_log_lines(lines)

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


def run(type: str, real: list[float], imag: list[float], array_size: int, delete_temp_files: bool = False, delete_log_files: bool = False, save_full_log: bool = False) -> tuple[np.ndarray, int, float, tuple[dict[str, int], dict[str, int]]]:
    """
    Execute a specified FFT/IFFT operation using RISC-V assembly simulation on Veer.

    Depending on the 'type' parameter ('FFT', 'IFFT', 'vFFT', or 'vIFFT'), this function prepares the
    corresponding assembly file, performs necessary modifications, runs the simulation, and processes the
    output log to extract the result.

    Parameters
    ----------
    type : str
        Type of operation ('FFT', 'IFFT', 'vFFT', or 'vIFFT').
    real : list of float
        Real parts of the input array.
    imag : list of float
        Imaginary parts of the input array.
    array_size : int
        Size of the input data array.
    delete_temp_files : bool, optional
        If True, temporary files are deleted after execution (default is True).
    delete_log_files : bool, optional
        If True, log files are deleted after processing (default is False).

    Returns
    -------
    tuple
        A tuple containing:
        - A NumPy array of complex numbers representing the computed FFT/IFFT result.
        - An integer representing the number of cycles (from simulation).
        - A float indicating the execution time in seconds.
    """
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

    cycles, time, ins_count = simulate_on_veer(
        assemblyFile, logFile, delete_temp_files, save_full_log)
    result = process_file(logFile, delete_log_files)

    return (result, cycles, time, ins_count)


def npFFT(real: list[float], imag: list[float], _: int) -> tuple[np.ndarray, int, float, tuple[dict[str, int], dict[str, int]]]:
    """
    Compute the Fast Fourier Transform (FFT) using NumPy.

    Parameters
    ----------
    real : list of float
        List of real components.
    imag : list of float
        List of imaginary components.
    _ : int
        Unused parameter (for compatibility).

    Returns
    -------
    tuple
        A tuple containing:
        - A NumPy array of 32-bit complex numbers representing the FFT result.
        - An integer for the number of cycles (placeholder: -1).
        - A float for the execution time in seconds.
    """
    import time
    complex_numbers = np.array(real, dtype=np.float32) + \
        1j * np.array(imag, dtype=np.float32)
    start_time = time.time()
    fft = np.fft.fft(complex_numbers).astype(np.complex64)
    end_time = time.time()
    elapsed_time = end_time - start_time
    npFFTcycles = -1  # Not implemented
    return fft, npFFTcycles, elapsed_time, ({}, {})


def npIFFT(real: list[float], imag: list[float], _: int) -> tuple[np.ndarray, int, float, tuple[dict[str, int], dict[str, int]]]:
    """
    Compute the Inverse Fast Fourier Transform (IFFT) using NumPy.

    Parameters
    ----------
    real : list of float
        List of real components.
    imag : list of float
        List of imaginary components.
    _ : int
        Unused parameter (for compatibility).

    Returns
    -------
    tuple
        A tuple containing:
        - A NumPy array of complex numbers representing the IFFT result.
        - An integer for the number of cycles (placeholder: -1).
        - A float for the execution time in seconds.
    """
    import time
    complex_numbers = np.array(real) + 1j * np.array(imag)
    start_time = time.time()
    ifft = np.fft.ifft(complex_numbers)
    end_time = time.time()
    elapsed_time = end_time - start_time
    npIFFTcycles = -1  # Not implemented
    return ifft, npIFFTcycles, elapsed_time, ({}, {})


def nFFT(real: list[float], imag: list[float], array_size: int, deleteFiles: bool = False) -> tuple[np.ndarray, int, float, tuple[dict[str, int], dict[str, int]]]:
    """
    Compute the FFT using RISC-V assembly code simulation on Veer.

    Parameters
    ----------
    real : list of float
        List of real components.
    imag : list of float
        List of imaginary components.
    array_size : int
        Size of the input data array.
    deleteFiles : bool, optional
        If True, temporary files are deleted after execution (default is True).

    Returns
    -------
    tuple
        A tuple containing:
        - A NumPy array of complex numbers representing the FFT result.
        - An integer for the number of cycles (from simulation).
        - A float for the execution time in seconds.
    """
    return run('FFT', real, imag, array_size, deleteFiles, False)


def nIFFT(real: list[float], imag: list[float], array_size: int, deleteFiles: bool = False) -> tuple[np.ndarray, int, float, tuple[dict[str, int], dict[str, int]]]:
    """
    Compute the IFFT using RISC-V assembly code simulation on Veer.

    Parameters
    ----------
    real : list of float
        List of real components.
    imag : list of float
        List of imaginary components.
    array_size : int
        Size of the input data array.
    deleteFiles : bool, optional
        If True, temporary files are deleted after execution (default is True).

    Returns
    -------
    tuple
        A tuple containing:
        - A NumPy array of complex numbers representing the IFFT result.
        - An integer for the number of cycles (from simulation).
        - A float for the execution time in seconds.
    """
    return run('IFFT', real, imag, array_size, deleteFiles, False)


def vFFT(real: list[float], imag: list[float], array_size: int, deleteFiles: bool = False, save_full_log: bool = False) -> tuple[np.ndarray, int, float, tuple[dict[str, int], dict[str, int]]]:
    """
    Compute the FFT using vectorized RISC-V assembly code simulation on Veer.

    Parameters
    ----------
    real : list of float
        List of real components.
    imag : list of float
        List of imaginary components.
    array_size : int
        Size of the input data array.
    deleteFiles : bool, optional
        If True, temporary files are deleted after execution (default is True).

    Returns
    -------
    tuple
        A tuple containing:
        - A NumPy array of complex numbers representing the FFT result (truncated to array_size elements).
        - An integer for the number of cycles (from simulation).
        - A float for the execution time in seconds.
    """
    return run('vFFT', real, imag, array_size, deleteFiles, False, save_full_log=save_full_log)


def vIFFT(real: list[float], imag: list[float], array_size: int, deleteFiles: bool = False, save_full_log: bool = False) -> tuple[np.ndarray, int, float, tuple[dict[str, int], dict[str, int]]]:
    """
    Compute the IFFT using vectorized RISC-V assembly code simulation on Veer.

    Parameters
    ----------
    real : list of float
        List of real components.
    imag : list of float
        List of imaginary components.
    array_size : int
        Size of the input data array.
    deleteFiles : bool, optional
        If True, temporary files are deleted after execution (default is True).

    Returns
    -------
    tuple
        A tuple containing:
        - A NumPy array of complex numbers representing the IFFT result (truncated to array_size elements).
        - An integer for the number of cycles (from simulation).
        - A float for the execution time in seconds.
    """
    return run('vIFFT', real, imag, array_size, deleteFiles, False, save_full_log=save_full_log)


class FFTResult(TypedDict):
    result: NDArray[np.complex64]
    cycles: int
    time: float
    VectorIns: dict[str, int]
    nonVectorIns: dict[str, int]


class BenchmarkResults(TypedDict):
    size: int
    input: NDArray[np.complex64]
    npFFT: FFTResult
    npIFFT: FFTResult
    nFFT: FFTResult
    nIFFT: FFTResult
    vFFT: FFTResult
    vIFFT: FFTResult


def compute_FFT_IFFT_with_benchmarks(array_size: int, real: list[float] = [], imag: list[float] = [], hardcoded: bool = False) -> BenchmarkResults:
    """
    Computes Fast Fourier Transform (FFT) and Inverse FFT (IFFT) benchmarks using different implementations.

    This function compares the performance of NumPy's FFT/IFFT with standard RISC-V and vectorized RISC-V
    implementations. If hardcoded is False, it generates random complex values within the range [-1000, 1000]
    for both real and imaginary components.

    Parameters
    ----------
    array_size : int
        Size of the array to perform FFT/IFFT operations on.
    real : list[float], optional
        Real components of the input array. If empty and hardcoded=False, 
        random values will be generated.
    imag : list[float], optional
        Imaginary components of the input array. If empty and hardcoded=False, 
        random values will be generated.
    hardcoded : bool, optional
        If True, uses the provided real and imag arrays.
        If False, generates random arrays regardless of provided inputs.

    Returns
    -------
    dict
        A dictionary containing benchmark results with the following structure:
        {
            'size': int,              # Size of the input array
            'input': ndarray[complex], # Complex input array
            'npFFT': {                # NumPy FFT results
                'result': ndarray[complex],  # FFT output
                'cycles': int,               # CPU cycles used (-1 if not implemented)
                'time': float                # Execution time in seconds
                'vectorIns' : dict[str, int] # list of vector instructions executed and their count
                'nonVectorIns' : dict[str, int] # list of non-vector instructions executed and their count
            },
            'npIFFT': { ... },        # NumPy IFFT results (same structure)
            'nFFT': { ... },          # RISC-V FFT results (same structure)
            'nIFFT': { ... },         # RISC-V IFFT results (same structure)
            'vFFT': { ... },          # Vectorized RISC-V FFT results (same structure)
            'vIFFT': { ... }          # Vectorized RISC-V IFFT results (same structure)
        }

    Notes
    -----
    - The function prints progress information during benchmark execution.
    - All complex results are converted to 32-bit floating point precision.
    - For vectorized results, only the first array_size elements are included.

    Examples
    --------
    >>> results = compute_FFT_IFFT_with_benchmarks(64)
    >>> print(f"NumPy FFT took {results['npFFT']['time']} seconds")
    >>> print(f"Vectorized FFT took {results['vFFT']['time']} seconds")

    >>> # Using custom input arrays
    >>> real_data = [1.0, 2.0, 3.0, 4.0]
    >>> imag_data = [0.0, 0.0, 0.0, 0.0]
    >>> results = compute_FFT_IFFT_with_benchmarks(4, real_data, imag_data, True)
    """
    if not hardcoded:
        import random
        import numpy as np
        real = [random.uniform(-1000, 1000) for _ in range(array_size)]
        imag = [random.uniform(-1000, 1000) for _ in range(array_size)]

        # Combine into a complex array
        complex_array = np.array([complex(r, i)
                                 for r, i in zip(real, imag)], dtype=np.complex64)

        # Extract real and imaginary parts as 32-bit floats
        real = complex_array.real.astype(np.float32)
        imag = complex_array.imag.astype(np.float32)

    print(f"\n\nStarting benchmark for {array_size}")

    # Dictionary to store results
    benchmark_results = {}

    print(f"Performing npFFT for array of size {array_size}")
    npFFTresult, npFFTcycles, npFFTtime, npFFTIns = npFFT(
        real, imag, array_size)
    print(f"Done. Took {npFFTtime} milliseconds")

    print(f"Performing npIFFT for array of size {array_size}")
    npIFFTresult, npIFFTcycles, npIFFTtime, npIFFTIns = npIFFT(
        npFFTresult.real, npFFTresult.imag, array_size)
    print(f"Done. Took {npIFFTtime} milliseconds")

    print(f"Performing nFFT for array of size {array_size}")
    nFFTresult, nFFTcycles, nFFTtime, nFFTIns = nFFT(real, imag, array_size)
    print(f"Done. Took {nFFTtime} milliseconds")

    print(f"Performing nIFFT for array of size {array_size}")
    nIFFTresult, nIFFTcycles, nIFFTtime, nIFFTIns = nIFFT(
        nFFTresult.real, nFFTresult.imag, array_size)
    print(f"Done. Took {nIFFTtime} milliseconds")

    print(f"Performing vFFT for array of size {array_size}")
    vFFTresult, vFFTcycles, vFFTtime, vFFTIns = vFFT(real, imag, array_size)
    print(f"Done. Took {vFFTtime} milliseconds")

    print(f"Performing vIFFT for array of size {array_size}")
    vIFFTresult, vIFFTcycles, vIFFTtime, vIFFTIns = vIFFT(
        vFFTresult.real, vFFTresult.imag, array_size)
    print(f"Done. Took {vIFFTtime} milliseconds")

    import numpy as np
    benchmark_results['size'] = array_size
    benchmark_results['input'] = np.array(real) + 1j * np.array(imag)

    benchmark_results['npFFT'] = {
        'result': npFFTresult.astype(np.complex64),
        'cycles': npFFTcycles,
        'time': npFFTtime,
        'vectorIns': npFFTIns[0],
        'nonVectorIns': npFFTIns[1],
    }

    benchmark_results['npIFFT'] = {
        'result': npIFFTresult.astype(np.complex64),
        'cycles': npIFFTcycles,
        'time': npIFFTtime,
        'vectorIns': npFFTIns[0],
        'nonVectorIns': npFFTIns[1],
    }

    benchmark_results['nFFT'] = {
        'result': nFFTresult.astype(np.complex64),
        'cycles': nFFTcycles,
        'time': nFFTtime,
        'vectorIns': nFFTIns[0],
        'nonVectorIns': nFFTIns[1],
    }

    benchmark_results['nIFFT'] = {
        'result': nIFFTresult.astype(np.complex64),
        'cycles': nIFFTcycles,
        'time': nIFFTtime,
        'vectorIns': nIFFTIns[0],
        'nonVectorIns': nIFFTIns[1],
    }

    benchmark_results['vFFT'] = {
        'result': vFFTresult[:array_size].astype(np.complex64),
        'cycles': vFFTcycles,
        'time': vFFTtime,
        'vectorIns': vFFTIns[0],
        'nonVectorIns': vFFTIns[1],
    }

    benchmark_results['vIFFT'] = {
        'result': vIFFTresult[:array_size].astype(np.complex64),
        'cycles': vIFFTcycles,
        'time': vIFFTtime,
        'vectorIns': vIFFTIns[0],
        'nonVectorIns': vIFFTIns[1],
    }

    print(f"\n\nAll benchmarks done for {array_size}\n\n")
    return benchmark_results


def changeVectorSize(size: int) -> None:
    """
    Updates the vector size in the Veer configuration file.

    Args:
        size (int): The number of bytes to set for the vector size.

    Modifies:
        Updates the "bytes_per_vec" field in the 'whisper.json' configuration file
        located in the VEER_FOLDER_PATH directory.

    Returns:
        None
    """
    import json
    file_path = VEER_FOLDER_PATH+'/whisper.json'

    with open(file_path, 'r') as file:
        data = json.load(file)

    data["vector"]["bytes_per_vec"] = size

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

    return


def saveResults(results: BenchmarkResults, filename: str) -> None:
    """
    Saves the given results dictionary to a pickle file.

    Args:
        results (BenchmarkResults): The data to be saved.
        filename (str): The base filename (without extension) for the pickle file.

    Saves:
        A file named '<filename>.pickle' containing the serialized results.
    """
    with open(filename+'.pickle', 'wb') as f:
        pickle.dump(results, f)


def loadResults(filename: str) -> BenchmarkResults:
    """
    Loads results from a pickle file.

    Args:
        filename (str): The base filename (without extension) for the pickle file.

    Returns:
        dict: The deserialized data stored in the pickle file.
    """
    with open(filename+'.pickle', 'rb') as f:
        return pickle.load(f)


def benchmark_different_sizes(sizes: list[int], real: list[float] = [], imag: list[float] = [], hardcoded: bool = False) -> list[BenchmarkResults]:
    """
    Runs FFT/IFFT benchmarks on arrays of different sizes.

    This function computes FFT and IFFT for the given list of sizes.
    If `hardcoded` is True, predefined input values will be used; otherwise, 
    the function will use the provided `real` and `imag` arrays.

    Args:
        sizes (list[int]): List of array sizes to benchmark.
        real (list[float], optional): Real part of input arrays. Defaults to an empty list.
        imag (list[float], optional): Imaginary part of input arrays. Defaults to an empty list.
        hardcoded (bool, optional): Whether to use predefined input values. Defaults to False.

    Returns:
        list[BenchmarkResults]: A list containing benchmark results for each tested size.
    """
    results = []
    for size in sizes:
        result = compute_FFT_IFFT_with_benchmarks(size, real, imag, hardcoded)
        results.append(result)

    return results


def performTestsAndSaveResults(sizes: list[int], filename: str = f"{RESULT_FOLDER_PATH}/fft_ifft_results", real: list[float] = [], imag: list[float] = [], hardcoded: bool = False) -> list[BenchmarkResults]:
    """
    Runs FFT/IFFT benchmarks, checks for existing results, and saves new results.

    This function first loads previously saved results (if available) and determines
    which sizes still need testing. It then benchmarks those sizes, updates the results, 
    and saves them to a file.

    Args:
        sizes (list[int]): List of FFT/IFFT sizes to test.
        filename (str, optional): Path to the file where results are saved. Defaults to 'fft_ifft_results'.
        real (list[float], optional): Real part of the input array. Defaults to an empty list.
        imag (list[float], optional): Imaginary part of the input array. Defaults to an empty list.
        hardcoded (bool, optional): Whether to use hardcoded values instead of random inputs. Defaults to False.

    Returns:
        list[dict]: A list of dictionaries containing FFT/IFFT benchmark results.
    """
    import os
    # Load previous results if the CSV file exists
    existing_results = []
    if os.path.exists(filename+'.pickle'):
        existing_results = loadResults(filename)
        # print(f"Loaded existing results from {filename}")

    # Gather tested sizes
    tested_sizes = {result['size'] for result in existing_results}

    # Identify sizes that need testing
    sizes_to_test = [size for size in sizes if size not in tested_sizes]

    if not sizes_to_test:
        # print("All sizes have already been tested. No new tests will be performed.")
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


def flatten_results(results: list[BenchmarkResults]):
    """
    Converts a list of benchmark results into a structured dictionary format.

    This function extracts relevant performance metrics from each result and 
    organizes them into a dictionary for easier analysis and storage.

    Args:
        results (list[BenchmarkResults]): List of benchmark results to flatten.

    Returns:
        dict: A dictionary containing extracted FFT/IFFT benchmark data, with keys representing 
              different metrics such as size, input values, FFT results, execution cycles, and execution times.
    """
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
