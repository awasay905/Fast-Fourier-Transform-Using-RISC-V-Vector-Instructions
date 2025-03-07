from python.functions import process_file


file_path = "./veer/tempFiles/logNV2.txt"  # Replace with your log file path
realVal, imagVal = process_file(file_path)
print("Total", len(realVal), "values found.")
for i in range(len(realVal)):
    sign = '+' if imagVal[i] >= 0 else '-'
    imag_val_abs = abs(imagVal[i])
    print(f"{realVal[i]:.6f} {sign} {imag_val_abs:.6f}i")
