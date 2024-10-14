# Comprehensive FFT Implementation Flow

## 1. Main Function (_start)

The main function serves as the entry point of the program and orchestrates the entire FFT process.

### Steps:
1. Calls `initHelperVector`
2. Loads data addresses:
   - a0 = address of real[]
   - a1 = address of imag[]
   - a2 = size of arrays (N)
3. Calls `vFFT`
4. Calls `vIFFT`
5. Calls `print`
6. Jumps to `_finish`

### Register Usage:
- a0, a1, a2: Used to pass arguments to functions
- ra: Return address (implicitly used by function calls)

## 2. initHelperVector Function

This function initializes a vector with sequential integers, which is crucial for various operations in the FFT algorithm.

### Steps:
1. Loads address of helperVector into t0
2. Loads size (N) into t1
3. Sets vector length using `vsetvli` instruction
4. Uses `vid.v` instruction to generate an index vector (0, 1, 2, 3, ...)
5. Stores the generated vector to memory

### Register Usage:
- t0: Address of helperVector
- t1: Size (N)
- t2: Vector length returned by vsetvli
- v0: Temporary vector register for index generation

### Vector Instructions:
- `vsetvli t2, t1, e32, m1`: Set vector length based on size N, using 32-bit elements
- `vid.v v0`: Generate index vector
- `vse32.v v0, (t0)`: Store index vector to memory

## 3. logInt Function

This function calculates the base-2 logarithm of the input, which is used to determine the number of stages in the FFT algorithm.

### Steps:
1. Move input N from a0 to t0
2. Initialize result (a0) to 0
3. Loop:
   - Break if t0 is 0
   - Right shift t0 by 1
   - Increment a0
4. Adjust result by subtracting 1

### Register Usage:
- a0: Input N, then output log2(N)
- t0: Loop counter and temporary storage

## 4. vReverse Function

This function performs bit reversal on the input vector, which is a crucial step in the FFT algorithm to reorder the input data.

### Steps:
1. Save ra to stack (as it calls logInt)
2. Call logInt to get log2(N)
3. Initialize registers for bit manipulation
4. Main loop:
   - Perform bit reversal operations using vector instructions
   - Use masked operations to selectively update elements
5. Move result from v28 to v29
6. Restore ra and return

### Register Usage:
- a0: Input N
- t0, t1, t2, t3, t4: Temporary registers for bit manipulation
- v0: Mask vector
- v28: Temporary result vector
- v29: Input and output vector

### Vector Instructions:
- `vand.vx`, `vmsne.vx`, `vor.vx`: Bit manipulation and masking operations
- `vmv.v.v`: Vector move instruction

## 5. vMySin and vMyCos Functions

These functions calculate sine and cosine values for a vector of input angles, used for twiddle factor calculation in the FFT.

### Steps:
1. Perform range reduction on input angles
2. Calculate Taylor series approximation
3. Apply sign correction (for cosine)

### Register Usage:
- t0, t1: Temporary registers for loop control and address calculation
- ft0, ft1, ft2: Floating-point temporary registers
- v0: Mask vector
- v1, v2, v3, v5: Temporary vectors for calculation
- v30 (for vMyCos) or v31 (for vMySin): Input and output vectors

### Vector Instructions:
- `vfmul.vv`, `vfadd.vv`, `vfdiv.vf`: Vector floating-point operations
- `vmflt.vf`, `vmfgt.vf`: Vector comparison operations

## 6. vOrdina Function

This function performs the bit-reversal permutation on the input data, preparing it for the FFT algorithm.

### Steps:
1. Save registers to stack
2. Set up pointers for real and imaginary parts
3. Load helper vector
4. Main loop:
   - Call vReverse to get reversed indices
   - Use reversed indices to reorder data
   - Store reordered data to temporary arrays
5. Second loop to copy data back to original arrays
6. Restore registers and return

### Register Usage:
- a0, a1, a2: Input pointers and size
- t0, t1, t5, t6: Temporary registers and pointers
- a3, a4: Temporary array pointers
- v23, v24, v26, v27, v29: Vector registers for data manipulation

### Vector Instructions:
- `vle32.v`, `vse32.v`: Vector load and store
- `vsll.vi`: Vector shift left logical
- `vloxei32.v`, `vsoxei32.v`: Indexed vector load and store

## 7. vTransform Function

This is the core function that performs the FFT or IFFT, depending on the inverse flag.

### Steps:
1. Call vOrdina to reorder input data
2. Calculate twiddle factors using vMySin and vMyCos
3. Perform butterfly operations in multiple stages
4. Use vector operations for efficient computation

### Register Usage:
- a0, a1, a2: Input pointers and size
- a3: Inverse flag
- t0, t1, t2, t3, s0, s1, s2, s5: Temporary registers
- ft0, ft1, ft2, ft3: Floating-point temporary registers
- v0-v31: Vector registers for various operations

### Vector Instructions:
- Extensive use of vector arithmetic (`vfadd.vv`, `vfsub.vv`, `vfmul.vv`)
- Vector memory operations (`vle32.v`, `vse32.v`, `vloxei32.v`, `vsoxei32.v`)
- Vector masking operations for conditional execution

## 8. vFFT and vIFFT Functions

These functions are wrappers around vTransform, setting the appropriate inverse flag.

### vFFT Steps:
1. Save ra to stack
2. Set inverse flag (a3) to 1
3. Call vTransform
4. Restore ra and return

### vIFFT Steps:
1. Save ra to stack
2. Set inverse flag (a3) to -1
3. Call vTransform
4. Perform scaling of output (divide by N)
5. Restore ra and return

### Register Usage:
- a0, a1, a2: Input pointers and size
- a3: Inverse flag
- t0-t4, ft0: Additional registers used in vIFFT for scaling

## 9. print Function

This function prints the real and imaginary parts of the FFT result.

### Steps:
1. Load size and pointers to real and imaginary arrays
2. Output specific patterns for Python script identification
3. Loop through arrays, loading and "outputting" values
4. Output end patterns

### Register Usage:
- t0, t1, t2, t3: Loop control and array pointers
- ft0, ft1: Floating-point registers for real and imaginary values

## 10. Data Section

The data section contains:
- Input data (real and imaginary parts)
- Temporary arrays for computation
- Constants (PI, TWO_PI, etc.)
- Helper vector space

## 11. Vector Usage and Optimization

- The code extensively uses RISC-V vector instructions for parallelism
- Vector length is dynamically set based on the input size N
- Key optimizations:
  - Vectorized bit reversal in vOrdina
  - Parallel computation of twiddle factors
  - Vectorized butterfly operations in vTransform
  - Use of masked operations for conditional execution

## 12. Register Preservation Strategy

- Most functions save ra when they call other functions
- Leaf functions (logInt, vMySin, vMyCos) don't save/restore registers
- vTransform, being the most complex, uses many registers and relies on caller-save convention
- Helper vector (v22) and constants (v26) are preserved throughout the computation

## 13. Overall Algorithm Flow

1. Initialize helper vector
2. Perform bit-reversal permutation (vOrdina)
3. Calculate twiddle factors (in vTransform)
4. Perform butterfly operations in log2(N) stages (main loop in vTransform)
5. For IFFT, perform additional scaling of the output

This implementation showcases an efficient use of RISC-V vector extensions for FFT computation, balancing between register usage, function modularity, and vectorized operations for optimal performance.