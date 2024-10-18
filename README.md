# Fast Fourier Transform using RISC-V Vector Instructions

This project demonstrates the calculation of Fast Fourier Transform (FFT) as well as its Inverse (IFFT) using RISC-V vector instructions. The original FFT code was adapted from StackOverflow (link given at the end), converted to C, and then modified to replace the complex class with real and imaginary part arrays for easier assembly conversion. The Vectorized FFT code can be run on the Veer simulator on Ubuntu, and a Python script is provided to convert the output of Veer to human readable form.

It was made as a project for your course Computer Architecture and Assembly Language. A non-vectorized version of the code is also provided for comparision. 

## Project Structure

- `docs/`: Contains files for project documentation.

- `README.md`: Main project documentation file.

- `results/`: Directory where benchmarking results are saved.

- `src/`: Contains the source code.

- `FFT_V.s`: Vectorized implementation of FFT in RISC-V assembly. (`FFT_V2.s` includes runtime improvements).

- `FFT_NV.s`: Non-vectorized implementation of FFT in RISC-V assembly. (`FFT_NV2.s` includes runtime improvements).

- `fft.c`: Main C code implementing FFT calculation without using complex number libraries.

- `readVeerOutput.py`: Python script to convert hexadecimal values from Veer logs to floating-point numbers.

- `tests/`: Contains code for use by Python wrappers, including both vectorized and non-vectorized FFT/IFFT versions.

- `tools/`: Contains Python utility scripts.

- `functions.py`: Python module with wrapper functions for using RISC-V FFT/IFFT.

- `test.py`: Script to run FFT/IFFT on various sizes and methods for benchmarking speed.

- `veer/`: Contains necessary files for running the Veer simulator.

- `Makefile`: Script to compile and run simulations on the Veer simulator.

- `script.py`: Python script that performs FFT using the provided wrapper functions.


## Prerequisites

Before setting up and running the simulation, ensure you have the following prerequisites installed:

- Linux (tested on Ubuntu 20.04)
- [RISC-V GNU Toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain)
- [VeeR-ISS RISC-V Simulator](https://github.com/chipsalliance/VeeR-ISS)
- [Verilator](https://verilator.org/guide/latest/install.html) (Tested on Version 5.006)

A guide for installing prerequisite is available [here](./docs/VEERGUIDE.md).
## Simulation Instructions

1. Clone this repository to your local machine:

```bash
git clone https://github.com/awasay905/Fast-Fourier-Transform-Using-RISC-V-Vector-Instructions.git
cd Fast-Fourier-Transform-Using-RISC-V-Vector-Instructions
```

2. At the end of the assmebly file, look for .data section, there will be two data named real and imaginary. They are the real and imaginary part of the input you can to perform FFT/IFFT on. Change them according to your need (making sure the imput is of power of 2). After this, change the number of dataSize variable according to the size of data. Save the file

3. To run the FFT simulation on the Veer simulator, execute the following command:

```bash
make clean      (or make cleanNV for non-vectorized code)
make all        (or make allNV for non-vectorized code)
```

    This command will compile the code, run it on the Veer simulator, and generate a log file with the results.

4. After running the simulation, use the provided Python script to convert the hex values in the log file to float. Ensure you have Python installed on your system.

```bash
python3 readVeerOutput.py 
```
## Runtime
The time required to run FFT normally is N * LogN where N is the number of input. With this vectorized version, each operation is done on VLEN (vector length/ amount of element in a vector) element altogether, resulting in N * LogN / VLEN time required. Benchmarking on the VeeR Simulator with VLEN of 8 resulted in almost 5 times less cycles in vectorized code compared to non-vectorized code.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

## Future Plan/Improvment
Although this is the basic FFT without much improvement, it can be made faster if a better version of sin/cos is implemented. Current implementation uses taylor series and takes about 300 cycle for VLEN sin/cos. 

## Acknowledgments
- Thanks to our instructors Dr Salman Zaffar and Dr Zain for guiding us
- Thanks to the contributors on StackOverflow for the [orignal FFT code](https://stackoverflow.com/questions/8801158/fft-in-a-single-c-file).
- Thanks to MERL DSU for guide on verilator

