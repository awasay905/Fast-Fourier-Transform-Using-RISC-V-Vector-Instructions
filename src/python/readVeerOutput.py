def process_file(file_name, start_index, end_index):
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

        return real, imag
    
    except FileNotFoundError:
        print(f"The file {file_name} does not exist.")
        return real, imag
    
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

def read_file_reverse_and_process(file_path):
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

if __name__ == "__main__":
    file_path = "./veer/tempFiles/logV.txt"  # Replace with your file path
    start, end = read_file_reverse_and_process(file_path)
    realVal, imagVal = process_file(file_path, start, end)
    realVal = hex_to_float(realVal)
    imagVal = hex_to_float(imagVal)
    print("Total", len(realVal), "values found.")
    for i in range(len(realVal)):
        sign = '+' if imagVal[i] >= 0 else '-'
        imag_val_abs = abs(imagVal[i])
        print(f"{realVal[i]:.6f} {sign} {imag_val_abs:.6f}i")
