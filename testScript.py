import numpy as np

# calculates FFT using numoy library
def npFFT(real, imag):
    complex_numbers = np.array(real) + 1j * np.array(imag)
    return np.fft.fft(complex_numbers)

# calculates IFFT using numoy library
def npIFFT(real, imag):
    complex_numbers = np.array(real) + 1j * np.array(imag)
    return np.fft.fft(complex_numbers)

# Helper function to format the array elements for assmebly fire
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

# Write array values to  the assembly file
def writeArrayToAssemblyFile(input_file, output_file, real, imag, n, type):
    # Read the input file and find the .data section
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Insert the array values after .data
    for i, line in enumerate(lines):    
        if "call" in line and "XXXX" in line:
            lines[lines.index(line)] = f"    call {type}                      # Apply {type} on the arrays\n"
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

# Runs code of assemblyFiile on Veer, saving the log to logFile and returning cycle count
def runOnVeer(assemblyFile, logFile):
    import subprocess as sp
    import re
    GCC_PREFIX = "riscv32-unknown-elf"
    ABI = "-march=rv32gcv -mabi=ilp32f"
    LINK = "./VeerFiles/link.ld"
    fileName = assemblyFile[assemblyFile.rfind('/') + 1:-2]  # removes .s  and gets file name from the file path
    tempPath = f"./PythonFiles/tempFiles/{fileName}"


    # Commands to run
    commands = [
        f"{GCC_PREFIX}-gcc {ABI} -lgcc -T{LINK} -o {tempPath}.exe {assemblyFile} -nostartfiles -lm",
        f"{GCC_PREFIX}-objcopy -O verilog {tempPath}.exe {tempPath}.hex",
        f"{GCC_PREFIX}-objdump -S {tempPath}.exe > {tempPath}.dis",
        f"whisper -x {tempPath}.hex -s 0x80000000 --tohost 0xd0580000 -f {logFile} --configfile ./VeerFiles/whisper.json"
    ]

    retired_instructions = None  # Variable to store the number of retired instructions

    # Regular expression to find "Retired X instructions"
    instruction_regex = r"Retired\s+(\d+)\s+instructions"

    #print(f"Running {assemblyFile} in Veer")
    # Execute the commands one by one
    for command in commands:
        try:
            result = sp.run(command,capture_output=True, shell=True, check=False, text=True)

            if result.stderr:
                # Search for the "Retired X instructions" pattern in the stderr
                match = re.search(instruction_regex, result.stderr)
                if match:
                    retired_instructions = match.group(1)  # Extract the number
                    #print(result.stderr)
        
        except sp.CalledProcessError as e:
            if "whisper" in command:
                break
            print(f"An error occurred while executing: {command}")
            print(f"Error: {e}")
            exit(-1)
        
    #print(f"Numbers of cycles used: {retired_instructions}")

    return retired_instructions

# onverts HEX code value to float from log file
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

# reads file and gives start and end index of load. basically it finds the pattern i used in assembly coe
# for easily logging of outptu
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


# reads the log file , and saves all the hex values between given indexs
def process_file(file_name):
    start_index, end_index = find_log_index(file_name)
    real = []
    imag = []
    
    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()
        
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
    
# runs FFT, IFFT, vFFT, vIFFT (type), returning the result, and cycles used in a tuple
def run(type, real, imag, array_size):
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
    
    cycles = runOnVeer(assemblyFile, logFile)
    realOutput, imagOutput = process_file(logFile)
    result =  np.array(realOutput) + 1j * np.array(imagOutput)  

    return (result, cycles)

def generate_data(data_type, size):
    """Generate data based on the specified type."""
    if data_type == 'linear':
        return [i for i in range(size)]
    elif data_type == 'constant':
        return [5.0] * size
    elif data_type == 'random_float':
        return np.random.rand(size).tolist()
    elif data_type == 'random_int':
        return np.random.randint(0, 100, size).tolist()
    elif data_type == 'powers_of_two':
        return [2 ** i for i in range(size)]
    elif data_type == 'sine_wave':
        return [np.sin(2 * np.pi * i / size) for i in range(size)]
    elif data_type == 'exponential':
        return [np.exp(i / size) for i in range(size)]
    elif data_type == 'alternating':
        return [(-1) ** i * 5.0 for i in range(size)]
    else:
        raise ValueError("Unsupported data type!")
    

# Performs FFT and IFFT on array of n size, of real and imag. if hardcoded if flase then simple floats will be used
def test(array_size, real = [], imag = [], hardcoded = False):
    if not hardcoded:
        real =   [i * 2 for i in range(array_size)] 
        imag = [i * 2 for i in range(array_size)]  

    npFFTresult = npFFT(real, imag)
    #npIFFTresult = npIFFT(real, imag)
    npIFFTresult = []
    FFTresult, FFTcycles = run('FFT', real, imag, array_size)
    #IFFTresults, IFFTcycles = run('IFFT', FFTresult.real, FFTresult.imag, array_size)
    IFFTresults, IFFTcycles = [], 0
    vFFTresult, vFFTcycles = run('vFFT', real, imag, array_size)
    #vIFFTresult, vIFFTcycles = run('vIFFT', vFFTresult.real, vFFTresult.imag, array_size)
    vIFFTresult, vIFFTcycles = [], 0

    return [npFFTresult, npIFFTresult, FFTresult, FFTcycles, IFFTresults, IFFTcycles, vFFTresult, vFFTcycles, vIFFTresult, vIFFTcycles]




data_types = [
    'linear', 'constant', 'random_float', 'random_int',
    'powers_of_two', 'sine_wave', 'exponential', 'alternating'
]

results_by_type = {data_type: [] for data_type in data_types}
sizes = [2**i for i in range(4, 8)]  # From 16 to 8192

for size in sizes:
    for data_type in data_types:
        real = generate_data(data_type, size)
        imag = generate_data(data_type, size)  
        result = test(size, real, imag)
        results_by_type[data_type].append({
            'size': size,
            'data_type': data_type,
            'npFFT': result[0],
            'npIFFT': result[1],
            'FFT': result[2],
            'FFT_cycles': result[3],
            'IFFT': result[4],
            'IFFT_cycles': result[5],
            'vFFT': result[6],
            'vFFT_cycles': result[7],
            'vIFFT': result[8],
            'vIFFT_cycles': result[9],
        })

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('FFT_IFFT_Analysis_Report.pdf') as pdf:
    # Loop through each data type for reports
    for data_type, results in results_by_type.items():
        # Report: Difference between NumPy FFT and vFFT
        plt.figure()
        plt.title(f'Difference between NumPy FFT and vFFT for {data_type}')
        for result in results:
            plt.plot(result['npFFT'].real, label='NumPy FFT Real', color='blue')
            plt.plot(result['vFFT'].real, label='vFFT Real', color='red', linestyle='--')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        # Report: Difference between NumPy IFFT and vIFFT
        plt.figure()
        plt.title(f'Difference between NumPy IFFT and vIFFT for {data_type}')
        for result in results:
            plt.plot(result['npIFFT'].real, label='NumPy IFFT Real', color='blue')
            plt.plot(result['vIFFT'].real, label='vIFFT Real', color='red', linestyle='--')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        # Report: Comparison of cycles used
        plt.figure()
        plt.title(f'CPU Cycles Comparison: FFT and vFFT for {data_type}')
        sizes = [result['size'] for result in results]  # Extract sizes for plotting
        fft_cycles = [result['FFT_cycles'] for result in results]
        vfft_cycles = [result['vFFT_cycles'] for result in results]
        plt.plot(sizes, fft_cycles, label='FFT Cycles', marker='o')
        plt.plot(sizes, vfft_cycles, label='vFFT Cycles', marker='o')
        plt.xlabel('Array Size')
        plt.ylabel('CPU Cycles')
        plt.xscale('log')
        plt.legend()
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        # Repeat for IFFT and vIFFT cycles
        plt.figure()
        plt.title(f'CPU Cycles Comparison: IFFT and vIFFT for {data_type}')
        ifft_cycles = [result['IFFT_cycles'] for result in results]
        vifft_cycles = [result['vIFFT_cycles'] for result in results]
        plt.plot(sizes, ifft_cycles, label='IFFT Cycles', marker='o')
        plt.plot(sizes, vifft_cycles, label='vIFFT Cycles', marker='o')
        plt.xlabel('Array Size')
        plt.ylabel('CPU Cycles')
        plt.xscale('log')
        plt.legend()
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()


print("Reports and graphs have been saved to 'FFT_IFFT_Analysis_Report.pdf'.")
exit(0)





