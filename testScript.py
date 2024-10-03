import numpy as np

# Calculate FFT using numpy, returns the FFT, cycles and the time taken to calculate
def npFFT(real, imag):
    import time
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

# Runs assembly code on Veer, saving the log to logFile and returning cycle count and time taken
def runOnVeer(assemblyFile, logFile, deleteFiles = True):
    print(deleteFiles)
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
    if deleteFiles:
        commands = [
            f"{GCC_PREFIX}-gcc {ABI} -lgcc -T{LINK} -o {tempPath}.exe {assemblyFile} -nostartfiles -lm",
            f"rm -f {assemblyFile}", # delete the assembly code after its done being translated
            f"{GCC_PREFIX}-objcopy -O verilog {tempPath}.exe {tempPath}.hex",
            # f"{GCC_PREFIX}-objdump -S {tempPath}.exe > {tempPath}.dis",
            # f"rm -f {tempPath}.dis" # im not even sure why we are disassemblign it
            f"rm -f {tempPath}.exe",
            f"whisper -x {tempPath}.hex -s 0x80000000 --tohost 0xd0580000 -f {logFile} --configfile ./VeerFiles/whisper.json",
            f"rm -f {tempPath}.hex", # delete the  hex file after its done being translated
        ]
    else:
        commands = [
            f"{GCC_PREFIX}-gcc {ABI} -lgcc -T{LINK} -o {tempPath}.exe {assemblyFile} -nostartfiles -lm",
            f"{GCC_PREFIX}-objcopy -O verilog {tempPath}.exe {tempPath}.hex",
            f"{GCC_PREFIX}-objdump -S {tempPath}.exe > {tempPath}.dis",
            f"whisper -x {tempPath}.hex -s 0x80000000 --tohost 0xd0580000 -f {logFile} --configfile ./VeerFiles/whisper.json"
        ]

    retired_instructions = None  # Variable to store the number of retired instructions

    # Regular expression to find "Retired X instructions"
    instruction_regex = r"Retired\s+(\d+)\s+instructions"  
    
    # Execute the commands one by one
    for command in commands:
        try:
            start_time = time.time()
            result = sp.run(command,capture_output=True, shell=True, text=True)
            end_time = time.time()
            timetaken = end_time - start_time # And save the time
            if result.stderr:
                # Search for the "Retired X instructions" pattern in the stderr
                match = re.search(instruction_regex, result.stderr)
                if match:
                    retired_instructions = match.group(1)  # Extract the number
        
        except sp.CalledProcessError as e:
            print(f"An error {e}occurred while executing: {command}")
            print(f"Error: {e}")
            exit(-1)
            
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
def process_file(file_name, deleteFiles = True):
    start_index, end_index = find_log_index(file_name)
    real = []
    imag = []
    
    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()
        
        if deleteFiles:
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
    
# runs FFT, IFFT, vFFT, vIFFT (type), returning the result, cycles and time taken used in a tuple
def run(type, real, imag, array_size, deleteFiles = True):
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
    
    cycles, time = runOnVeer(assemblyFile, logFile, deleteFiles)
    realOutput, imagOutput = process_file(logFile, deleteFiles)

    result =  np.array(realOutput) + 1j * np.array(imagOutput)  

    return (result, cycles, time)

# Performs FFT and IFFT on array of n size, of real and imag. if hardcoded if flase then simple floats will be used
# Returns FFT, IFFT and time taken in performing them on numpy, riscv, and nevctorized risc v
def test(array_size, real = [], imag = [], hardcoded = False):
    if not hardcoded:
        real =   [i * 2 for i in range(array_size)] 
        imag = [i * 2 for i in range(array_size)]  

    npFFTresult, npFFTcycles, npFFTtime = npFFT(real, imag)
    npIFFTresult, npIFFTcycles,npIFFTtime = npIFFT(real, imag)
    FFTresult, FFTcycles, FFTtime= run('FFT', real, imag, array_size)
    IFFTresults, IFFTcycles, IFFtime = run('IFFT', FFTresult.real, FFTresult.imag, array_size)
    vFFTresult, vFFTcycles, vFFTtime = run('vFFT', real, imag, array_size)
    vIFFTresult, vIFFTcycles, vIFFTtime = run('vIFFT', vFFTresult.real, vFFTresult.imag, array_size)

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
    
# Function to calculate error between two arrays
def calculate_error(type1, type2):
    type1 = np.array(type1)
    type2 = np.array(type2)
        
    # Calculate the error (magnitude of the difference)
    error = np.abs(type1 - type2)
        
    return error


# TESTING
sizes = [2 ** i for i in range(2, 10)]  # From 16 to 8192
results = []

for size in sizes:
    result = test(size)
    results.append({
        'size' : size,
        'npFFTresult' : result[0],
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
        'vFFTresult' : result[12] ,
        'vFFTcycles' : result[13], 
        'vFFTtime': result[14],
        'vIFFTresult': result[15],
        'vIFFTcycles': result[16],
        'vIFFTtime': result[17]
    });           
    

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

df = pd.DataFrame(results)

with PdfPages('FFT_IFFT_Analysis_Reportv2.pdf') as pdf:    
    # Cycle Count Differnce Between FFT (and IFFT diff color) of Different input sizes
    plt.figure(figsize=(10, 5))
    plt.plot(df['size'], df['FFTcycles'], label='FFT Cycles', marker='o')
    plt.plot(df['size'], df['vFFTcycles'], label='vFFT Cycles', marker='o')
    plt.plot(df['size'], pd.DataFrame([i*i for i in df['size']]), label='O(n*2)', marker='o')
    plt.plot(df['size'], pd.DataFrame([i*np.log(i) for i in df['size']]), label='O(n*logn)', marker='o')
    plt.xscale('log')
    plt.ylabel('Cycles Count') #TODO
    plt.xlabel('Input Size (log scale)')
    plt.legend()
    plt.title("Instructions Count Difference Between FFT and vFFT")
    pdf.savefig()
    plt.close()

    
    
    # Run time difference between different input sizes for np, normal, vectorized
    
    
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





