from src.python.functions import *


file_path = "./veer/tempFiles/logV2.txt"  # Replace with your log file path
numbers = process_file(file_path)
print("Total", len(numbers), "values found.")
for c in numbers[-5:]:
    print(f"{c.real:12.6f} {'+' if c.imag >= 0 else '-'} {abs(c.imag):10.6f}j")


