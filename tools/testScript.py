# Formats given array to string for a readable format for assembly file
def format_array(array):
    formatted_lines = []
    current_line = ".float "
    for i, value in enumerate(array):
        current_line += f"{value:.12f}, "
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
    import os    
    import numpy as np
    
    if not os.path.exists("./PythonFiles/tempFiles"):
        os.mkdir("./PythonFiles/tempFiles")
        
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
        import random
        real = [random.uniform(-1000, 1000) for _ in range(array_size)] 
        imag = [random.uniform(-1000, 1000)  for _ in range(array_size)]  

    print(f"\n\nStarting benchmark for {array_size}")
    print(f"Performing npFFT for array of size {array_size}")
    npFFTresult, npFFTcycles, npFFTtime = npFFT(real, imag)
    print("Done.")
    print(f"Performing npIFFT for array of size {array_size}")
    npIFFTresult, npIFFTcycles,npIFFTtime =  npIFFT(npFFTresult.real, npFFTresult.imag)
    print("Done.")
    print(f"Performing FFT for array of size {array_size}")
    FFTresult, FFTcycles, FFTtime= FFT(real, imag, array_size)
    print("Done.")
    print(f"Performing IFFT for array of size {array_size}")
    IFFTresults, IFFTcycles, IFFtime =  IFFT(FFTresult.real, FFTresult.imag, array_size)
    print("Done.")
    print(f"Performing vFFT for array of size {array_size}")
    vFFTresult, vFFTcycles, vFFTtime = vFFT(real, imag, array_size)
    print("Done.")
    print(f"Performing vIFFT for array of size {array_size}")
    vIFFTresult, vIFFTcycles, vIFFTtime =  vIFFT(vFFTresult.real, vFFTresult.imag, array_size)
    print("Done")
    
    print(f"\n\nAll benchmark done for {array_size}\n\n")

    return [npFFTresult,npFFTcycles,npFFTtime, npIFFTresult,npIFFTcycles, npIFFTtime, FFTresult, FFTcycles,FFTtime, IFFTresults, IFFTcycles, IFFtime,vFFTresult, vFFTcycles, vFFTtime,vIFFTresult, vIFFTcycles,vIFFTtime, real, imag]

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
    
    print(f"Trying to load results from {results_csv}")
    if os.path.exists(results_csv):
        print(f"File {results_csv} found. Loading Results...")
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
        
        print("Results loaded from CSV.")
        loaded_sizes = [result['size'] for result in results]
        sizes_to_load = [size for size in sizes if size not in loaded_sizes]
        if len(sizes_to_load) > 0: 
            print("Looks like some new array sizes have been added")
            print(f"Performing tests on {sizes_to_load}")
            results.extend(runTests(sizes_to_load)) 
            print("Appended new results to CSV")
            save_results_to_csv(results_csv, results)
            print("Modified CSV saved")
            
    else:
        print(f"File {results_csv} not found.")
        print("Rerunnning Tests")
        # Run tests and save results to CSV
        results = runTests(sizes)
        print(f"Saving results to {results_csv}.")
        save_results_to_csv(results_csv, results)
        print(f"Results saved to {results_csv}.")
        
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
            'real': result[-2],
            'imag' : result[-1],
            'npFFTresult': result[0],
            'npFFTcycles': result[1],
            'npFFTtime': result[2],
            'npIFFTresult': result[3],
            'npIFFTcycles': result[4],
            'npIFFTtime': result[5],
            'FFTresult': result[6],
            'FFTcycles': result[7],
            'FFTtime': result[8],
            'IFFTresult': result[9],
            'IFFTcycles': result[10],
            'IFFTtime': result[11],
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
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# RUN FFT/IFFT ON DIFFERENT SIZES AND SAVE THE RESULTS
results_csv = 'test_results random.csv' # File to save the results to
sizes = [2 ** i for i in range(4, 15)]  # Define the sizes for testing. must be power of 2
#results = load_results_from_csv(results_csv, sizes)    # RN I do not save to file as i think it is causing precision issues
results = runTests(sizes)


df = pd.DataFrame(results)
# Make plots and save it to pdf
with PdfPages('FFT_and_IFFT_Analysis_Report.pdf') as pdf:   
    
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
    
    # Veer Instruction Cycle Count Differnce Between FFT and vFFT of Different input sizes with big O
    plt.figure(figsize=(12, 6))
    plt.plot(df['size'], df['FFTcycles'], label='FFT Cycles', marker='D')
    plt.plot(df['size'], df['vFFTcycles'], label='vFFT Cycles', marker='x')
    plt.plot(df['size'], pd.DataFrame([(i*i) for i in df['size']]), label='O(n*2)', marker='o')
    plt.plot(df['size'], pd.DataFrame([i*np.log(i) for i in df['size']]), label='O(n*logn)', marker='o')
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
    
    #   Improvement of vFFT over FFT of Different input sizes i.e ratio
    plt.figure(figsize=(12, 6))
    plt.plot(df['size'], df['FFTtime']/df['vFFTtime'], label='vFFT Improvement', marker='x')
    plt.plot(df['size'], df['IFFTtime']/df['vIFFTtime'], label='vIFFT Improvement', marker='o')
    plt.ylabel('Ratio')
    plt.xlabel('Input Size (log)')
    plt.xscale('log')
    plt.legend()
    plt.title("vFFT run time improvement over FFTs")
    pdf.savefig()
    plt.close()
    
    #   Improvement of vFFT over FFT of Different input sizes i.e ratio
    plt.figure(figsize=(12, 6))
    plt.plot(df['size'], df['FFTtime']/df['vFFTtime'], label='vFFT Improvement', marker='x')
    plt.plot(df['size'], df['IFFTtime']/df['vIFFTtime'], label='vIFFT Improvement', marker='o')
    plt.ylabel('Ratio')
    plt.xlabel('Input Size (log)')
    plt.xscale('log')
    plt.legend()
    plt.title("vFFT cpu instructions improvement over FFTs")
    pdf.savefig()
    plt.close()
    
    # Show value difference per size
    # for idx, row in df.iterrows():
    #     size = row['size']
    #     np_fft_result = np.array(row['npFFTresult'])
    #     fft_result = np.array(row['FFTresult'])
    #     v_fft_result = np.array(row['vFFTresult'])

    #     # Create a figure for each size
    #     plt.figure(figsize=(12, 6))

    #     # Plot the abs diff of real part of the results
    #     # plt.plot(np.abs(np_fft_result - fft_result), label='FFT Diff', color='blue', linestyle='-')
    #     # plt.plot(np.abs(np_fft_result - v_fft_result), label='vFFT  Diff', color='green', linestyle='--')
     

    #     plt.plot(np.abs(np.real(np_fft_result) - np.real(fft_result)), label='FFT Real Diff', color='blue', linestyle='-')
    #     plt.plot(np.abs(np.real(np_fft_result) - np.real(v_fft_result)), label='vFFT Real Diff', color='green', linestyle='--')
     
    # # Set the title and labels
    #     plt.title(f'vFFT/FFT  REAL diff from npFFT for Size {size}')
    #     plt.xlabel('Index')
    #     plt.ylabel('Diff')
    #     plt.legend()

    #     # Show the plot
    #     pdf.savefig()
    #     plt.close()
    #     # Create a figure for each size
    #     plt.figure(figsize=(12, 6))
        
    #     # # Optionally plot the dif of imaginary part (uncomment the following lines if needed)
    #     plt.plot(np.abs(np.imag(np_fft_result) - np.imag(fft_result)), label='FFT Imag Diff', color='red', linestyle='-.')
    #     plt.plot(np.abs(np.imag(np_fft_result) - np.imag(v_fft_result)), label='vFFT Imag Diff', color='orange', linestyle=':')

    #     # Set the title and labels
    #     plt.title(f'vFFT/FFT IMAG diff from npFFT for Size {size}')
    #     plt.xlabel('Index')
    #     plt.ylabel('Diff')
    #     plt.legend()

    #     # Show the plot
    #     pdf.savefig()
    #     plt.close()


def create_spreadsheet(results, file_name):
    import pandas as pd
    import numpy as np
    
    # Example difference function
    def diff_func(a, b):
        return abs(a - b)
    # Create an Excel writer object
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        # Loop through each entry in results
        for entry in results:
            size = entry['size']
            
            data = {
                'real' : [result for result in entry['real']],
                'imag' : [result for result in entry['imag']],
                'npFFT_real': [result.real for result in entry['npFFTresult']],
                'npFFT_imag': [result.imag for result in entry['npFFTresult']],
                #'FFT_real': [result.real for result in entry['FFTresult']],
               # 'FFT_imag': [result.imag for result in entry['FFTresult']],
                'vFFT_real': [result.real for result in entry['vFFTresult']],
                'vFFT_imag': [result.imag for result in entry['vFFTresult']],
               # 'npIFFT_real': [result.real for result in entry['npIFFTresult']],
               # 'npIFFT_imag': [result.imag for result in entry['npIFFTresult']],
               # 'IFFT_real': [result.real for result in entry['IFFTresult']],
                #'IFFT_imag': [result.imag for result in entry['IFFTresult']],
                'vIFFT_real': [result.real for result in entry['vIFFTresult']],
                'vIFFT_imag': [result.imag for result in entry['vIFFTresult']],
            }
            
        
             # Calculate abs differences  between ffts
          #  data['diff_npFFT_FFT_abs'] = [diff for diff in np.abs(entry['npFFTresult'] - entry['FFTresult'])]
            data['diff_npFFT_vFFT_abs'] = [diff for diff in np.abs(entry['npFFTresult'] - entry['vFFTresult'])]
            #data['diff_FFT_vFFT_abs'] = [diff for diff in np.abs(entry['FFTresult'] - entry['vFFTresult'])]
            data['diff_npIFFT_IFFT_abs'] = [diff for diff in np.abs(entry['npIFFTresult'] - entry['IFFTresult'])]
           # data['diff_npIFFT_vIFFT_abs'] = [diff for diff in np.abs(entry['npIFFTresult'] - entry['vIFFTresult'])]
           # data['diff_IFFT_vIFFT_abs'] = [diff for diff in np.abs(entry['IFFTresult'] - entry['vIFFTresult'])]
            
       
            # Add runtime and cycles
           # data['npFFT_cycles'] = [entry['npFFTcycles']] * len(entry['npFFTresult'])
          # data['npFFT_time'] = [entry['npFFTtime']] * len(entry['npFFTresult'])
          #  data['FFT_cycles'] = [entry['FFTcycles']] * len(entry['FFTresult'])
          #  data['FFT_time'] = [entry['FFTtime']] * len(entry['FFTresult'])
          #  data['vFFT_cycles'] = [entry['vFFTcycles']] * len(entry['vFFTresult'])
           # data['vFFT_time'] = [entry['vFFTtime']] * len(entry['vFFTresult'])

            #data['npIFFT_cycles'] = [entry['npIFFTcycles']] * len(entry['npIFFTresult'])
           # data['npIFFT_time'] = [entry['npIFFTtime']] * len(entry['npIFFTresult'])
           # data['IFFT_cycles'] = [entry['IFFTcycles']] * len(entry['IFFTresult'])
           # data['IFFT_time'] = [entry['IFFTtime']] * len(entry['IFFTresult'])
          #  data['vIFFT_cycles'] = [entry['vIFFTcycles']] * len(entry['vIFFTresult'])
           # data['vIFFT_time'] = [entry['vIFFTtime']] * len(entry['vIFFTresult'])

             # in the end Calculate differences  between real and imag coutnerpart. optional
            # data['diff_npFFT_FFT_real'] = [diff_func(entry['npFFTresult'][i].real, entry['FFTresult'][i].real) for i in range(len(entry['npFFTresult']))]
            # data['diff_npFFT_FFT_imag'] = [diff_func(entry['npFFTresult'][i].imag, entry['FFTresult'][i].imag) for i in range(len(entry['npFFTresult']))]
            # data['diff_npFFT_vFFT_real'] = [diff_func(entry['npFFTresult'][i].real, entry['vFFTresult'][i].real) for i in range(len(entry['npFFTresult']))]
            # data['diff_npFFT_vFFT_imag'] = [diff_func(entry['npFFTresult'][i].imag, entry['vFFTresult'][i].imag) for i in range(len(entry['npFFTresult']))]
            # data['diff_FFT_vFFT_real'] = [diff_func(entry['FFTresult'][i].real, entry['vFFTresult'][i].real) for i in range(len(entry['FFTresult']))]
            # data['diff_FFT_vFFT_imag'] = [diff_func(entry['FFTresult'][i].imag, entry['vFFTresult'][i].imag) for i in range(len(entry['FFTresult']))]
            # data['diff_npIFFT_IFFT_real'] = [diff_func(entry['npIFFTresult'][i].real, entry['IFFTresult'][i].real) for i in range(len(entry['npFFTresult']))]
            # data['diff_npIFFT_IFFT_imag'] = [diff_func(entry['npIFFTresult'][i].imag, entry['IFFTresult'][i].imag) for i in range(len(entry['npFFTresult']))]
            # data['diff_npIFFT_vIFFT_real'] = [diff_func(entry['npIFFTresult'][i].real, entry['vIFFTresult'][i].real) for i in range(len(entry['npFFTresult']))]
            # data['diff_npIFFT_vIFFT_imag'] = [diff_func(entry['npIFFTresult'][i].imag, entry['vIFFTresult'][i].imag) for i in range(len(entry['npFFTresult']))]
            # data['diff_IFFT_vIFFT_real'] = [diff_func(entry['IFFTresult'][i].real, entry['vIFFTresult'][i].real) for i in range(len(entry['FFTresult']))]
            # data['diff_IFFT_vIFFT_imag'] = [diff_func(entry['IFFTresult'][i].imag, entry['vIFFTresult'][i].imag) for i in range(len(entry['FFTresult']))]

            
            # Create a DataFrame for the current size
            df = pd.DataFrame(data)
            
            # Write to a new sheet named after the size
            df.to_excel(writer, sheet_name=f'Size_{size}', index=False)

# Usage
create_spreadsheet(results, 'onlyvfft_results.xlsx')



print("Reports and graphs have been saved to 'FFT_IFFT_Analysis_Report.pdf'.")
exit(0)





