# Formats given array to string for a readable format for assembly file
def format_array(array):
    formatted_lines = []
    current_line = ".float "
    for i, value in enumerate(array):
        current_line += f"{value:.6f}, "
        if (i + 1) % 4 == 0:  # Add space after every 4th number
            current_line += " "
        if (i + 1) % 32 == 0:  # New line after every 32 numbers
            formatted_lines.append(current_line.strip(", "))
            current_line = ".float "
    if current_line.strip(", "):  # Add remaining line if not exactly multiple of 32
        formatted_lines.append(current_line.strip(", "))
    return formatted_lines

# Write array values to the assembly file data section
def writeArrayToAssemblyFile(input_file, output_file, real, imag, n, type):
    # Read the input file and find the .data section
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Insert the array values after .data
    for i, line in enumerate(lines):    
        if "call" in line and "XXXX" in line:
            lines[lines.index(line)] = f"    call {type}"
        if ".data" in line:
            # Insert real array
            lines.insert(i + 1, "real:\n")
            real_lines = format_array(real)
            for j, real_line in enumerate(real_lines):
                lines.insert(i + 2 + j, real_line + "\n")
            
            # Insert a blank line
            lines.insert(i + 2 + len(real_lines), "\n")
            
            # Insert imag array
            lines.insert(i + 3 + len(real_lines), "imag:\n")
            imag_lines = format_array(imag)
            for j, imag_line in enumerate(imag_lines):
                lines.insert(i + 4 + len(real_lines) + j, imag_line + "\n")
            
            # Insert a blank line and data size declaration
            lines.insert(i + 4 + len(real_lines) + len(imag_lines), "\n")
            lines.insert(i + 5 + len(real_lines) + len(imag_lines), f".set dataSize, {n}\n")
            break

    # Write the modified content to a new file
    with open(output_file, 'w') as file:
        file.writelines(lines)

    return

# Function to check for the pattern in the buffer to start reading log
def check_pattern(lines):
    # Split each line and check if the 7th column has the desired values
    required_values = ["00123000", "00123456", "00234000", "00234567", "00345000", "00345678"]
    found_values = []

    for line in lines:
        columns = line.split()
        if len(columns) > 6:  # Check if there are enough columns
            value = columns[6]  # Get the 7th column (index 6)
            if value in required_values:
                found_values.append(value)

    # Check if we found all required values
    return all(value in found_values for value in required_values)

# Runs assembly code on Veer, saves the log to logFile if deletefiles false, and returns the cpu cycle count and time taken
def runOnVeer(assemblyFile, logFile, deleteFiles = True, smallLog = True):
    import subprocess as sp
    import re
    import time
    GCC_PREFIX = "riscv32-unknown-elf"
    ABI = "-march=rv32gcv -mabi=ilp32f"
    LINK = "./VeerFiles/link.ld"
    fileName = assemblyFile[assemblyFile.rfind('/') + 1:-2]  # removes .s  and gets file name from the file path
    tempPath = f"./PythonFiles/tempFiles/{fileName}"
    timetaken = 0

    # Commands to run
    commands = [
        f"{GCC_PREFIX}-gcc {ABI} -lgcc -T{LINK} -o {tempPath}.exe {assemblyFile} -nostartfiles -lm",
        f"rm -f {assemblyFile}" if deleteFiles else "",  # Delete assembly code after translation if deleteFiles is True
        f"{GCC_PREFIX}-objcopy -O verilog {tempPath}.exe {tempPath}.hex",
        f"rm -f {tempPath}.exe" if deleteFiles else "",  # Remove executable if deleteFiles is True
        # f"{GCC_PREFIX}-objdump -S {tempPath}.exe > {tempPath}.dis" if not deleteFiles else f"rm -f {tempPath}.dis",  # Optional disassembly
        f"whisper -x {tempPath}.hex -s 0x80000000 --tohost 0xd0580000 -f /dev/stdout --configfile ./VeerFiles/whisper.json" if smallLog 
        else f"whisper -x {tempPath}.hex -s 0x80000000 --tohost 0xd0580000 -f {logFile} --configfile ./VeerFiles/whisper.json" ,
        f"rm -f {tempPath}.hex" if deleteFiles else "",  # Delete hex file after translation if deleteFiles is True
    ]

    # Remove any empty strings from the commands
    commands = [cmd for cmd in commands if cmd]  # Filter out empty strings

    retired_instructions = None  # Variable to store the number of retired instructions

    # Regular expression to find "Retired X instructions"
    instruction_regex = r"Retired\s+(\d+)\s+instructions"  
    
    start_time = time.time()
    # Execute the commands one by one
    for command in commands:
        try:
            if "whisper" in command and smallLog:
                process = sp.Popen(command, shell=True, stdout = sp.PIPE, stderr= sp.PIPE, text = True) # run this twice. once for capturing time , twice for reading logs
                
                buffer_size = 10
                lines_buffer = []
                recorded_log = []
                pattern_found = False
                
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break  # Exit if the process is done
                    
                    if output:  # Process only if there's output
                        # Add the output to the buffer
                        lines_buffer.append(output)
                        if len(lines_buffer) > buffer_size:
                            lines_buffer.pop(0)  # Maintain only the last 10 lines

                        # Check for the pattern in the 7th column
                        if not pattern_found:
                            if check_pattern(lines_buffer):
                                pattern_found = True
                                recorded_log.extend(lines_buffer)
                                print(f"[PATTERN DETECTED]: Starting to record output from here on.")

                        # If pattern found, record all lines after it
                        if pattern_found:
                            recorded_log.append(output)
                            
               
                           
                with open(logFile, 'w') as file:
                    file.writelines(recorded_log)
                
                stderr_output = process.stderr.read()
                if stderr_output:
                    # Search for the "Retired X instructions" pattern in the stderr
                    match = re.search(instruction_regex, stderr_output)
                    if match:
                        retired_instructions = match.group(1)  # Extract the number
        
            else:
                result = sp.run(command,capture_output=True, shell=True, text=True)
                if result.stderr:
                    # Search for the "Retired X instructions" pattern in the stderr
                    match = re.search(instruction_regex, result.stderr)
                    if match:
                        retired_instructions = match.group(1)  # Extract the number
        
        except sp.CalledProcessError as e:
            print(f"An error {e}occurred while executing: {command}")
            print(f"Error: {e}")
            exit(-1)
            
    end_time = time.time()
    timetaken = end_time - start_time
            
    return int(retired_instructions), timetaken

# Convert hex values array to float array. HEX should be IEEE format
def hex_to_float(hex_array):
    import struct
    float_array = []
    
    for hex_str in hex_array:
        # Ensure the hex string is exactly 8 characters long
        if len(hex_str) != 8:
            raise ValueError(f"Hex string '{hex_str}' is not 8 characters long")
        
        # Convert the hex string to a 32-bit integer
        int_val = int(hex_str, 16)
        
        # Pack the integer as a 32-bit unsigned integer
        packed_val = struct.pack('>I', int_val)
        
        # Unpack as a float (IEEE 754)
        float_val = struct.unpack('>f', packed_val)[0]
        
        float_array.append(float_val)
    
    return float_array

# Find index for the markers places in assembly code
def find_log_index(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the line with the specific pattern in reverse order
    start_index = -1
    end_index = -1
    for i in range(len(lines) - 1, -1, -1):
        if "lui" in lines[i]:
            words = lines[i].split()
            if "lui" in words and len(words) > 1 and words[words.index("lui") - 1] == "00123000":
                if "addi" in lines[i+1]:
                    words2 = lines[i+1].split()
                    if "addi" in words2 and len(words2) > 1 and words2[words2.index("addi") - 1] == "00123456":
                        if "lui" in lines[i+2]:
                            words3 = lines[i+2].split()
                            if "lui" in words3 and len(words3) > 1 and words3[words3.index("lui") - 1] == "00234000":
                                if "addi" in lines[i+3]:
                                    words4 = lines[i+3].split()
                                    if "addi" in words4 and len(words4) > 1 and words4[words4.index("addi") - 1] == "00234567":
                                        if "lui" in lines[i+4]:
                                            words5 = lines[i+4].split()
                                            if "lui" in words5 and len(words5) > 1 and words5[words5.index("lui") - 1] == "00345000":
                                                if "addi" in lines[i+5]:
                                                    words6 = lines[i+5].split()
                                                    if "addi" in words6 and len(words6) > 1 and words6[words6.index("addi") - 1] == "00345678":
                                                        if end_index == -1: 
                                                            end_index = i
                                                        elif start_index == -1:
                                                            start_index = i
                                                        else:
                                                            break

    # If the pattern line is found, call helper and process the file normally
    return start_index, end_index

# Reads log file and extract real and imag float values
def process_file(file_name, deleteTempFiles = True):
    start_index, end_index = find_log_index(file_name)
    real = []
    imag = []
    
    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()
        
        if deleteTempFiles:
            import os
            os.remove(file_name)
            
        # Ensure start and end indexes are within the valid range
        start_index = max(0, start_index)
        end_index = min(len(lines), end_index)

        # Initialize a flag to alternate between real and imag
        save_to_real = True
        
        # Process lines within the specified range
        for i in range(start_index, end_index):
            if "c.flw" in lines[i]:
                words = lines[i].split()
                if len(words) > 1:
                    index_of_cflw = words.index("c.flw")
                    if index_of_cflw > 0:
                        if save_to_real:
                            real.append(words[index_of_cflw - 1])
                            save_to_real = False
                        else:
                            imag.append(words[index_of_cflw - 1])
                            save_to_real = True

        
        
        return hex_to_float(real), hex_to_float(imag)
    
    except FileNotFoundError:
        print(f"The file {file_name} does not exist.")
        return real, imag
    
# runs "type", returning the result, cycles and time taken used in a tuple
def run(type, real, imag, array_size, deleteTempFiles = True, deleteLogFiles = True):
    print(f"RUnning {type} on array size {array_size}")
    import numpy as np
    assemblyFile = f"./PythonFiles/tempFiles/temp{type}.s"
    logFile = f"./PythonFiles/tempFiles/temp{type}log.txt"
    if type == 'vFFT' or type == 'vIFFT':
        writeArrayToAssemblyFile('./PythonFiles/vFFTforPython.s', assemblyFile, real, imag, array_size, type)
    elif type == 'FFT' or type == 'IFFT':
        writeArrayToAssemblyFile('./PythonFiles/FFTforPython.s', assemblyFile, real, imag, array_size, type)
    else:
        print("ERROR")
        exit(-1)
    
    cycles, time = runOnVeer(assemblyFile, logFile, deleteTempFiles)
    realOutput, imagOutput = process_file(logFile, deleteLogFiles)

    result =  np.array(realOutput) + 1j * np.array(imagOutput)  

    return (result, cycles, time)

# Calculate FFT using numpy, returns the FFT, cycles and the time taken to calculate
def npFFT(real, imag):
    import time
    import numpy as np
    start_time = time.time()
    complex_numbers = np.array(real) + 1j * np.array(imag)
    fft = np.fft.fft(complex_numbers)
    end_time = time.time()
    elapsed_time = end_time - start_time  
    npFFTcycles = -1  #implement later
    return fft, npFFTcycles,elapsed_time

# Calculate IFFT using numpy, returns the IFFT, cycles and the time taken to calculate
def npIFFT(real, imag):
    import time
    start_time = time.time()
    complex_numbers = np.array(real) + 1j * np.array(imag)
    ifft = np.fft.ifft(complex_numbers)
    end_time = time.time()
    elapsed_time = end_time - start_time  
    npIFFTcycles = -1  #implement later
    return ifft, npIFFTcycles,elapsed_time

# Calculaye FFT using risc v assembly code simulated on veer
def FFT(real, imag, array_size, deleteFiles = True):
    return run('FFT', real, imag, array_size, deleteFiles)

# Calculaye IFFT using risc v assembly code simulated on veer
def IFFT(real, imag, array_size, deleteFiles = True):
    return run('IFFT', real, imag, array_size, deleteFiles)

# Calculaye FFT using vectorized risc v assembly code simulated on veer
def vFFT(real, imag, array_size, deleteFiles = True):
    return run('vFFT', real, imag, array_size, deleteFiles)

# Calculaye IFFT using vecctorized risc v assembly code simulated on veer
def vIFFT(real, imag, array_size, deleteFiles = True):
    return run('vIFFT', real, imag, array_size, deleteFiles)

# Performs FFT and IFFT on array of n size, of real and imag. if hardcoded if flase then simple floats will be used
# Returns FFT, IFFT and time taken in performing them on numpy, riscv, and nevctorized risc v
def computeFFT_IFFTWithBenchmarks(array_size, real = [], imag = [], hardcoded = False):
    if not hardcoded:
        real = [i * 2 for i in range(array_size)] 
        imag = [i * 2 for i in range(array_size)]  

    npFFTresult, npFFTcycles, npFFTtime = npFFT(real, imag)
    npIFFTresult, npIFFTcycles,npIFFTtime =  npIFFT(npFFTresult.real, npFFTresult.imag)
    FFTresult, FFTcycles, FFTtime= FFT(real, imag, array_size)
    IFFTresults, IFFTcycles, IFFtime =  IFFT(FFTresult.real, FFTresult.imag, array_size)
    vFFTresult, vFFTcycles, vFFTtime = vFFT(real, imag, array_size)
    vIFFTresult, vIFFTcycles, vIFFTtime =  vIFFT(vFFTresult.real, vFFTresult.imag, array_size)

    return [npFFTresult,npFFTcycles,npFFTtime, npIFFTresult,npIFFTcycles, npIFFTtime, FFTresult, FFTcycles,FFTtime, IFFTresults, IFFTcycles, IFFtime,vFFTresult, vFFTcycles, vFFTtime,vIFFTresult, vIFFTcycles,vIFFTtime]

# Changes Veer vector size to number of bytes
def changeVectorSize(size):
    import json
    import os
    file_path = os.path.join("VeerFiles", "whisper.json")
    
    with open(file_path, 'r') as file:
        data = json.load(file)

    data["vector"]["bytes_per_vec"] = size

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
            
    return
  
# Function to save the results
def save_results_to_csv(results_csv,results, append=False):
    import csv
    import os
    # Helper functions to serialize/deserialize complex numbers and NumPy arrays. Convert a NumPy array or list to a string format for CSV.
    def serialize_array(array):
        return ','.join(map(str, array))
    
    mode = 'a' if append else 'w'  # 'a' for append, 'w' for overwrite
    header = ['size', 'npFFTresult', 'npFFTcycles', 'npFFTtime', 'npIFFTresult', 'npIFFTcycles', 'npIFFTtime', 
              'FFTresult', 'FFTcycles', 'FFTtime', 'IFFTresults', 'IFFTcycles', 'IFFtime', 
              'vFFTresult', 'vFFTcycles', 'vFFTtime', 'vIFFTresult', 'vIFFTcycles', 'vIFFTtime']
    
    with open(results_csv, mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header only if the file is new or in write mode
        if not append or os.stat(results_csv).st_size == 0:
            writer.writerow(header)
        
        for result in results:
            writer.writerow([
                result['size'],
                serialize_array(result['npFFTresult']),
                result['npFFTcycles'],
                result['npFFTtime'],
                serialize_array(result['npIFFTresult']),
                result['npIFFTcycles'],
                result['npIFFTtime'],
                serialize_array(result['FFTresult']),
                result['FFTcycles'],
                result['FFTtime'],
                serialize_array(result['IFFTresults']),
                result['IFFTcycles'],
                result['IFFtime'],
                serialize_array(result['vFFTresult']),
                result['vFFTcycles'],
                result['vFFTtime'],
                serialize_array(result['vIFFTresult']),
                result['vIFFTcycles'],
                result['vIFFTtime']
            ])

# Function to load results from the CSV
def load_results_from_csv(results_csv, sizes):
    import os
    import csv
    csv.field_size_limit(10**10)  # Set limit to 1,000,000 bytes (or adjust as necessary)
    
    #Convert a serialized string back to a NumPy array, handling empty strings
    def deserialize_array(array_str):
        if not array_str:
            return np.array([])  # Return an empty array if the string is empty
        return np.array([complex(x) if 'j' in x else float(x) for x in array_str.split(',') if x])
    
    results = []
    
    if os.path.exists(results_csv):
        with open(results_csv, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                results.append({
                    'size': int(row['size']),
                    'npFFTresult': deserialize_array(row['npFFTresult']),
                    'npFFTcycles': int(row['npFFTcycles']),
                    'npFFTtime': float(row['npFFTtime']),
                    'npIFFTresult': deserialize_array(row['npIFFTresult']),
                    'npIFFTcycles': int(row['npIFFTcycles']),
                    'npIFFTtime': float(row['npIFFTtime']),
                    'FFTresult': deserialize_array(row['FFTresult']),
                    'FFTcycles': int(row['FFTcycles']),
                    'FFTtime': float(row['FFTtime']),
                    'IFFTresults': deserialize_array(row['IFFTresults']),
                    'IFFTcycles': int(row['IFFTcycles']),
                    'IFFtime': float(row['IFFtime']),
                    'vFFTresult': deserialize_array(row['vFFTresult']),
                    'vFFTcycles': int(row['vFFTcycles']),
                    'vFFTtime': float(row['vFFTtime']),
                    'vIFFTresult': deserialize_array(row['vIFFTresult']),
                    'vIFFTcycles': int(row['vIFFTcycles']),
                    'vIFFTtime': float(row['vIFFTtime']),
                })
        
        print("Loaded results from CSV.")
        loaded_sizes = [result['size'] for result in results]
        sizes_to_load = [size for size in sizes if size not in loaded_sizes]
        if len(sizes_to_load) > 0: 
            results.extend(runTests(sizes_to_load)) 
            save_results_to_csv(results_csv, results)
            
    else:
      # Run tests and save results to CSV
      results = runTests(sizes)
      save_results_to_csv(results_csv, results)
      print("Test results saved to CSV.")
        
    return results

# RUNS FFT/IFFT on arrays of different sizes on dirrent real/imag array (pass array counraninf array) (if hardcodedgiven)). 
# TODO custom array values are not implemented yet
# Returns an array conatinign output of each test
def runTests(sizes,real = [], imag = [], hardcoded = False): 
    results = []
    for size in sizes:
        result = computeFFT_IFFTWithBenchmarks(size, real, imag, hardcoded)
        results.append({
            'size': size,
            'npFFTresult': result[0],
            'npFFTcycles': result[1],
            'npFFTtime': result[2],
            'npIFFTresult': result[3],
            'npIFFTcycles': result[4],
            'npIFFTtime': result[5],
            'FFTresult': result[6],
            'FFTcycles': result[7],
            'FFTtime': result[8],
            'IFFTresults': result[9],
            'IFFTcycles': result[10],
            'IFFtime': result[11],
            'vFFTresult': result[12],
            'vFFTcycles': result[13],
            'vFFTtime': result[14],
            'vIFFTresult': result[15],
            'vIFFTcycles': result[16],
            'vIFFTtime': result[17]
        })
        
    return results



# TESTING AND MAKING GRAPHS
import numpy as np

results_csv = 'test_results.csv' # FIle which will have the results
sizes = [2 ** i for i in range(2, 20)]  # Define the sizes for testing. must be power of 2
results = load_results_from_csv(results_csv, sizes)
# Run tests only if the file does not exist or append new sizes


import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

df = pd.DataFrame(results)
print(df['FFTtime']/df['vFFTtime'])
with PdfPages('FFT_IFFT_Analysis_Report.pdf') as pdf:   
    
    
    # Veer Instruction Cycle Count Differnce Between FFT and vFFT of Different input sizes
    plt.figure(figsize=(12, 6))
    plt.plot(df['size'], df['FFTcycles'], label='FFT Cycles', marker='D')
    plt.plot(df['size'], df['vFFTcycles'], label='vFFT Cycles', marker='x')
    plt.ylabel('VeeR Instruction Count')
    plt.xlabel('Input Size (log)')
    plt.xscale('log')
    plt.legend()
    plt.title("Instructions Count Difference Between FFT and vFFT")
    pdf.savefig()
    plt.close() 
    
    # Veer Instruction Cycle Count Differnce Between FFT and vFFT of Different input sizes
    plt.figure(figsize=(12, 6))
    plt.plot(df['size'], df['FFTcycles'], label='FFT Cycles', marker='D')
    plt.plot(df['size'], df['vFFTcycles'], label='vFFT Cycles', marker='x')
    plt.plot(df['size'], pd.DataFrame([(i*i) for i in df['size']]), label='O(n*2)', marker='o')
    plt.plot(df['size'], pd.DataFrame([3*i*np.log(i) for i in df['size']]), label='O(n*logn)', marker='o')
    plt.ylabel('VeeR Instruction Count')
    plt.xlabel('Input Size (log)')
    plt.xscale('log')
    plt.legend()
    plt.title("Instructions Count Difference Between FFT and vFFT With Big-O")
    pdf.savefig()
    plt.close()
    
    # Runtime Differnce Between npFFT, FFT and vFFT of Different input sizes
    plt.figure(figsize=(12, 6))
    plt.plot(df['size'], df['npFFTtime'], label='npFFT time', marker='.')
    plt.plot(df['size'], df['FFTtime'], label='FFT time', marker='D')
    plt.plot(df['size'], df['vFFTtime'], label='vFFT time', marker='x')
    plt.ylabel('Run time in millisecond')
    plt.xlabel('Input Size (log)')
    plt.xscale('log')
    plt.legend()
    plt.title("Runtime Difference Between npFFT, FFT and vFFT")
    pdf.savefig()
    plt.close()
    
    exit()

    for idx, row in df.iterrows():
        size = row['size']
        np_fft_result = np.array(row['npFFTresult'])
        fft_result = np.array(row['FFTresult'])
        v_fft_result = np.array(row['vFFTresult'])

        # Create a figure for each size
        plt.figure(figsize=(10, 6))

        # Plot the real part of the results
        plt.plot(np.real(np_fft_result), label='npFFT Real', color='blue', linestyle='-')
        plt.plot(np.real(fft_result), label='FFT Real', color='green', linestyle='--')
        plt.plot(np.real(v_fft_result), label='vFFT Real', color='red', linestyle='-.')

        # Optionally plot the imaginary part (uncomment the following lines if needed)
        plt.plot(np.imag(np_fft_result), label='npFFT Imaginary', color='blue', linestyle=':')
        plt.plot(np.imag(fft_result), label='FFT Imaginary', color='green', linestyle=':')
        plt.plot(np.imag(v_fft_result), label='vFFT Imaginary', color='red', linestyle=':')

        # Set the title and labels
        plt.title(f'FFT Results Comparison for Size {size}')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()

        # Show the plot
        pdf.savefig()
        plt.close()
    
    for idx, row in df.iterrows():
        size = row['size']
        np_fft_result = np.array(row['npFFTresult'])
        fft_result = np.array(row['FFTresult'])
        v_fft_result = np.array(row['vFFTresult'])

        # Calculate errors (Mean Squared Error)
        fft_real_error = np.mean((np.real(fft_result) - np.real(np_fft_result))**2)
        fft_imag_error = np.mean((np.imag(fft_result) - np.imag(np_fft_result))**2)
        v_fft_real_error = np.mean((np.real(v_fft_result) - np.real(np_fft_result))**2)
        v_fft_imag_error = np.mean((np.imag(v_fft_result) - np.imag(np_fft_result))**2)

        # Plot the error for real and imaginary parts
        plt.figure(figsize=(10, 6))

        plt.bar(['FFT Real Error', 'vFFT Real Error'], [fft_real_error, v_fft_real_error], color=['blue', 'red'], alpha=0.7)
        plt.bar(['FFT Imag Error', 'vFFT Imag Error'], [fft_imag_error, v_fft_imag_error], color=['blue', 'red'], alpha=0.7, hatch='/')

        # Set the title and labels
        plt.title(f'FFT Error Comparison for Size {size}')
        plt.ylabel('Mean Squared Error')
        plt.xlabel('Error Type')
        
        # Show the plot
        pdf.savefig()
        plt.close()
        # Run time difference between different input sizes for np, normal, vectorized
    def calculate_errors(df):
        sizes = []
        fft_mse, fft_ae, fft_max_error = [], [], []
        vfft_mse, vfft_ae, vfft_max_error = [], [], []
        
        for idx, row in df.iterrows():
            size = row['size']
            sizes.append(size)
            
            # Convert npFFT, FFT, and vFFT results to numpy arrays
            np_fft_result = np.array(row['npFFTresult'])
            fft_result = np.array(row['FFTresult'])
            v_fft_result = np.array(row['vFFTresult'])
            
            # Calculate errors for FFT
            fft_diff = np_fft_result - fft_result
            fft_mse.append(np.mean(np.abs(fft_diff)**2))  # MSE
            fft_ae.append(np.mean(np.abs(fft_diff)))      # Absolute Error
            fft_max_error.append(np.max(np.abs(fft_diff)))  # Max Error
            
            # Calculate errors for vFFT
            vfft_diff = np_fft_result - v_fft_result
            vfft_mse.append(np.mean(np.abs(vfft_diff)**2))  # MSE
            vfft_ae.append(np.mean(np.abs(vfft_diff)))      # Absolute Error
            vfft_max_error.append(np.max(np.abs(vfft_diff)))  # Max Error

        return sizes, fft_mse, fft_ae, fft_max_error, vfft_mse, vfft_ae, vfft_max_error

    # Function to plot the error metrics
    def plot_error_metrics(sizes, fft_mse, fft_ae, fft_max_error, vfft_mse, vfft_ae, vfft_max_error):
        plt.figure(figsize=(10, 6))

        # Plot MSE for FFT and vFFT
        plt.plot(sizes, fft_mse, label='FFT MSE', color='blue', linestyle='-')
        plt.plot(sizes, vfft_mse, label='vFFT MSE', color='red', linestyle='--')

        # Plot Absolute Error for FFT and vFFT
        plt.plot(sizes, fft_ae, label='FFT Absolute Error', color='green', linestyle='-')
        plt.plot(sizes, vfft_ae, label='vFFT Absolute Error', color='orange', linestyle='--')

        # Plot Max Error for FFT and vFFT
        plt.plot(sizes, fft_max_error, label='FFT Max Error', color='purple', linestyle='-')
        plt.plot(sizes, vfft_max_error, label='vFFT Max Error', color='brown', linestyle='--')

        # Set labels and title
        plt.xlabel('Size')
        plt.ylabel('Error')
        plt.title('Error Comparison for FFT and vFFT vs npFFT')
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()

    # Call functions to calculate errors and plot them
    sizes, fft_mse, fft_ae, fft_max_error, vfft_mse, vfft_ae, vfft_max_error = calculate_errors(df)
    plot_error_metrics(sizes, fft_mse, fft_ae, fft_max_error, vfft_mse, vfft_ae, vfft_max_error)
    
    
    
    # TRYING NEW ERROR SHOWING
    plt.figure(figsize=(12, 8))

    sizes = []
    fft_mse, fft_ae, fft_max_error, fft_sd = [], [], [], []
    vfft_mse, vfft_ae, vfft_max_error, vfft_sd = [], [], [], []

    # Loop over each row in the dataframe to calculate errors for each size
    for idx, row in df.iterrows():
        size = row['size']
        sizes.append(size)
        
        # Convert npFFT, FFT, and vFFT results to numpy arrays
        np_fft_result = np.array(row['npFFTresult'])
        fft_result = np.array(row['FFTresult'])
        v_fft_result = np.array(row['vFFTresult'])
        
        # Calculate errors for FFT
        fft_diff = np_fft_result - fft_result
        fft_mse.append(np.mean(np.abs(fft_diff)**2))  # Mean Squared Error (MSE)
        fft_ae.append(np.mean(np.abs(fft_diff)))      # Absolute Error (AE)
        fft_max_error.append(np.max(np.abs(fft_diff)))  # Max Error (Max)
        fft_sd.append(np.std(np.abs(fft_diff)))       # Standard Deviation (SD)
        
        # Calculate errors for vFFT
        vfft_diff = np_fft_result - v_fft_result
        vfft_mse.append(np.mean(np.abs(vfft_diff)**2))  # MSE
        vfft_ae.append(np.mean(np.abs(vfft_diff)))      # Absolute Error
        vfft_max_error.append(np.max(np.abs(vfft_diff)))  # Max Error
        vfft_sd.append(np.std(np.abs(vfft_diff)))       # Standard Deviation (SD)

    # Plotting the errors for FFT
    plt.plot(sizes, fft_mse, label='FFT MSE', color='blue', linestyle='-')
    plt.plot(sizes, fft_ae, label='FFT Absolute Error', color='green', linestyle='-')
    plt.plot(sizes, fft_max_error, label='FFT Max Error', color='purple', linestyle='-')
    plt.plot(sizes, fft_sd, label='FFT Std Dev', color='cyan', linestyle='-')

    # Plotting the errors for vFFT
    plt.plot(sizes, vfft_mse, label='vFFT MSE', color='red', linestyle='--')
    plt.plot(sizes, vfft_ae, label='vFFT Absolute Error', color='orange', linestyle='--')
    plt.plot(sizes, vfft_max_error, label='vFFT Max Error', color='brown', linestyle='--')
    plt.plot(sizes, vfft_sd, label='vFFT Std Dev', color='magenta', linestyle='--')

    # Set labels and title
    plt.xlabel('Size')
    plt.ylabel('Error')
    plt.title('Error and Standard Deviation Comparison for FFT and vFFT vs npFFT')

    # Show legend
    plt.legend()

    # Show grid
    plt.grid(True)

    # Save the figure to a file
    pdf.savefig()
    plt.close()



    # avg_errors = [np.mean(error) for error in errors]
    # sizes = [result['size'] for result in results]

    # # Average error
    # plt.figure(figsize=(10, 5))
    # plt.plot(sizes, avg_errors, marker='o', label='Average Error')
    # plt.xscale('log')
    # plt.xlabel('Input Size (log scale)')
    # plt.ylabel('Average Error')
    # plt.title('Average Error of vFFT vs npFFT')
    # plt.legend()
    # plt.grid()
    # pdf.savefig()
    # plt.close()
    
    # # Maximum error
    # max_errors = [np.max(error) for error in errors]

    # plt.figure(figsize=(10, 5))
    # plt.plot(sizes, max_errors, marker='o', color='r', label='Max Error')
    # plt.xscale('log')
    # plt.xlabel('Input Size (log scale)')
    # plt.ylabel('Max Error')
    # plt.title('Maximum Error of vFFT vs npFFT')
    # plt.legend()
    # plt.grid()
    # pdf.savefig()
    # plt.close()
    
    
    # # Standard deviation of errors
    # std_errors = [np.std(error) for error in errors]

    # plt.figure(figsize=(10, 5))
    # plt.plot(sizes, std_errors, marker='o', color='g', label='Std Error')
    # plt.xscale('log')
    # plt.xlabel('Input Size (log scale)')
    # plt.ylabel('Error Standard Deviation')
    # plt.title('Error Standard Deviation of vFFT vs npFFT')
    # plt.legend()
    # plt.grid()
    # pdf.savefig()
    # plt.close()
    
    # # Histogram of errors for the largest size
    # largest_size_index = np.argmax(sizes)
    # largest_errors = errors[largest_size_index]

    # plt.figure(figsize=(10, 5))
    # plt.hist(largest_errors, bins=30, alpha=0.7, color='purple')
    # plt.xlabel('Error Magnitude')
    # plt.ylabel('Frequency')
    # plt.title(f'Error Histogram for Input Size {sizes[largest_size_index]}')
    # plt.grid()
    # pdf.savefig()
    # plt.close()



    # Loop through each data type for reports
    # for data_type, results in results_by_type.items():
    #     # Report: Difference between NumPy FFT and vFFT
        
    #     for result in results:
    #         plt.figure()
    #         plt.title(f'Difference between NumPy FFT and vFFT for {data_type} for size {result['size']}')
    #         plt.plot(result['npFFT'].real - result['vFFT'].real, label='Real Part Diff', color='blue')
    #         plt.plot(result['npFFT'].imag - result['vFFT'].imag, label='Imag Part Diff', color='red')
    #         plt.xlabel('Index')
    #         plt.ylabel('Value')
    #         plt.legend()
    #         pdf.savefig()  # saves the current figure into a pdf page
    #         plt.close()

    # for data_type, results in results_by_type.items():
    #     # Report: Comparison of cycles used
    #     plt.figure()
    #     plt.title(f'CPU Cycles Comparison: FFT and vFFT for {data_type}')
    #     sizes = [result['size'] for result in results]  # Extract sizes for plotting
    #     fft_cycles = [result['FFT_cycles'] for result in results]
    #     vfft_cycles = [result['vFFT_cycles'] for result in results]
    #     plt.plot(sizes, fft_cycles, label='FFT Cycles', marker='o')
    #     plt.plot(sizes, vfft_cycles, label='vFFT Cycles', marker='o')
    #     plt.xlabel('Array Size')
    #     plt.ylabel('CPU Cycles')
    #     plt.xscale('log')
    #     plt.legend()
    #     pdf.savefig()  # saves the current figure into a pdf page
    #     plt.close()


print("Reports and graphs have been saved to 'FFT_IFFT_Analysis_Report.pdf'.")
exit(0)





