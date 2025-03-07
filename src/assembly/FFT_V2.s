#define STDOUT 0xd0580000
.set vectorSize, 1024              # Change this for max num of 32 bit vector element supported by hafdware

.section .text
.global _start
_start:

# Initialize helper vector and load data addresses
main:                  
    lw a0, size                     # Load size of real/imag arrays into a0
    call setlogN                    # Compute and store log2(size) for shared use by other functions

    la a0, real                     # a0 = address of real[]
    la a1, imag                     # a1 = address of imag[]
    lw a2, size                     # a2 = size of arrays (N)

    # Perform vFFT/vIFFT
    call vFFT
    
    # Print results and finish
    call print
    j _finish 



# Function: setlogN
# Computes log2(N) for a 32-bit unsigned integer in a0 and stores it in `logsize` in memory.
# Inputs:
#   - a0: 32-bit unsigned integer N
# Outputs:
#   - None. Result is saved in memory at 'logsize'
# Clobbers: t0, t1
setlogN:
    clz t0, a0                      # Count leading zeros. Helps in quick log2
    li t1, 31              
    sub t1, t1, t0                  # Subtract clz result from 31 to get log2
    la t0, logsize
    sw t1, 0(t0)                    # Save to memory
    
    jr ra



# Function: reverse
# Reverses the binary digits of a 32-bit integer.
# Inputs:
#   - v26: Input number to reverse.
#   - a0: Number of significant bits to reverse (optional; default 32).
#   - s1: 0x55555555
#   - s2: 0x33333333
#   - s3: 0x0F0F0F0F
#   - s4: 0x00FF00FF
#   - s5: Number of bits to shift
# Outputs:
#   - v29: The reversed binary number.
# Clobbers:
#   - v1, v2
vReverseIndexOffset:
    # Swap odd and even bits
    vsrl.vi v1, v26, 1              # v29 >> 1
    vand.vx v1, v1, s1              # (v29 >> 1) & 0x55555555
    vand.vx v2, v26, s1             # v29 & 0x55555555
    vsll.vi v2, v2, 1               # (v29 & 0x55555555) << 1
    vor.vv v29, v1, v2              # Result back to v29

    # Swap consecutive pairs
    vsrl.vi v1, v29, 2              # v29 >> 2
    vand.vx v1, v1, s2              # (v29 >> 2) & 0x33333333
    vand.vx v2, v29, s2             # v29 & 0x33333333
    vsll.vi v2, v2, 2               # (v29 & 0x33333333) << 2
    vor.vv v29, v1, v2              # Result back to v29

    # Swap nibbles
    vsrl.vi v1, v29, 4              # v29 >> 4
    vand.vx v1, v1, s3              # (v29 >> 4) & 0x0F0F0F0F
    vand.vx v2, v29, s3             # v29 & 0x0F0F0F0F
    vsll.vi v2, v2, 4               # (v29 & 0x0F0F0F0F) << 4
    vor.vv v29, v1, v2              # Result back to v29

    # Swap bytes
    vsrl.vi v1, v29, 8              # v29 >> 8
    vand.vx v1, v1, s4              # (v29 >> 8) & 0x00FF00FF
    vand.vx v2, v29, s4             # v29 & 0x00FF00FF
    vsll.vi v2, v2, 8               # (v29 & 0x00FF00FF) << 8
    vor.vv v29, v1, v2              # Result back to v29

    # Swap 2-byte pairs
    vsrl.vi v1, v29, 16             # v29 >> 16
    vsll.vi v2, v29, 16             # v29 << 16
    vor.vv v29, v1, v2              # Final result in v29

    # Shift by the req bit size
    vsrl.vx v29, v29, s5
    
    ret                             # Return with result in v29



# Function: preload_constants
# Preloads floating-point constants into vector registers for use in trigonometric calculations.
# Values are splatted (copied) onto vector registers because we need them in destructive fma
# Inputs:
#   - None
# Outputs:
#   - Constants loaded into v1 through v13 and ft11.
# Clobbers:
#   - t0, v1-v13, ft11.
preload_constants:
    # Save registers to stack
    addi sp, sp, -52
    fsw fs0, 0(sp)
    fsw fs1, 4(sp)
    fsw fs2, 8(sp)
    fsw fs3, 12(sp)
    fsw fs4, 16(sp)
    fsw fs5, 20(sp)
    fsw fs6, 24(sp)
    fsw fs7, 28(sp)
    fsw fs8, 32(sp)
    fsw fs9, 36(sp)
    fsw fs10, 40(sp)
    fsw fs11, 44(sp)

    # Load addresses of constants into registers
    la      t0, half_pi_hi          # Load address of half_pi_hi

    # Make use of the fact that all float are 4 bytes and stored consecutively
    flw     fs0, 0(t0)              # Load half_pi_hi into fs0
    flw     fs1, 4(t0)              # Load half_pi_lo into fs1
    flw     fs2, 8(t0)              # Load const_2_pi into fs2
    flw     fs3, 12(t0)             # Load const_12582912 into fs3

    # Load cosine coefficients
    flw     fs4, 16(t0)             # Load 2.44677067e-5 into fs4
    flw     fs5, 20(t0)             # Load -1.38877297e-3 into fs5
    flw     fs6, 24(t0)             # Load 4.16666567e-2 into fs6
    flw     fs7, 28(t0)             # Load -5.00000000e-1 into fs7
    flw     fs8, 32(t0)             # Load 1.00000000e+0 into fs8

    # Load sine coefficients
    flw     fs9, 36(t0)             # Load 2.86567956e-6 into fs9
    flw     fs10, 40(t0)            # Load -1.98559923e-4 into fs10
    flw     fs11, 44(t0)            # Load 8.33338592e-3 into fs11
    flw     ft11, 48(t0)            # Load -1.66666672e-1 into ft11

    vfmv.v.f v1, fs0
    vfmv.v.f v2, fs1
    vfmv.v.f v3, fs2
    vfmv.v.f v4, fs3
    vfmv.v.f v5, fs4
    vfmv.v.f v6, fs5
    vfmv.v.f v7, fs6
    vfmv.v.f v8, fs7
    vfmv.v.f v9, fs8
    vfmv.v.f v10, fs9
    vfmv.v.f v11, fs10
    vfmv.v.f v12, fs11
    vfmv.v.f v13, ft11
    fcvt.s.w ft11, zero

    # Restore Register
    flw fs0, 0(sp)
    flw fs1, 4(sp)
    flw fs2, 8(sp)
    flw fs3, 12(sp)
    flw fs4, 16(sp)
    flw fs5, 20(sp)
    flw fs6, 24(sp)
    flw fs7, 28(sp)
    flw fs8, 32(sp)
    flw fs9, 36(sp)
    flw fs10, 40(sp)
    flw fs11, 44(sp)
    addi sp, sp, 52

    ret



# Function: sin_cos_approx
# Calculates sin and cos of the float using chebishev polynomial
# Taken from stackoverflow
# Can not use  v22, v23, v1-v13, ft11
# Input:
#   - a = v21 = angle (a) in radians
# Output:
#   - rs = v30 = sin() (approximation)
#   - rc = v31 = cos() (approximation)
# Clobbers:
#   - v21, t0, t1, ft0, ft1, ft2, ft3
#   - c = v14
#   - j = v15
#   - a = v21
#   - rs = v30
#   - rc = v31
#   - s = v16
#   - sa = v17
#   - t = v18
#   - i = v19 
#   - ic = v20
v_sin_cos_approx:
    # j = fmaf(a, 6.36619747e-1f, 12582912.f) - 12582912.f;
    vmv.v.v   v15, v4               # v15 = 12582912
    vfmacc.vv  v15, v21, v3         # a(v21))*6.36619747e-1f(v3) + 12582912.f(v15)
    vfsub.vv    v15, v15, v4        # j = fmaf(a, 6.36619747e-1f, 12582912.f) - 12582912.f;

    vfnmsac.vv v21, v15, v1         #  a = fmaf (j, -half_pi_hi, a);
    vfnmsac.vv v21, v15, v2         #  a = fmaf (j, -half_pi_lo, a);

    vfcvt.x.f.v v19, v15            #  i = (int) j
    vadd.vi v20, v19, 1             # ic = i + 1

    vfmul.vv  v17, v21, v21         # ft2 = a * a (sa)

    # Approximate cosine.
    vmv.v.v     v14, v5             # c = 2.44677067e-5f
    vfmadd.vv   v14, v17, v6        # c = c * sa + -1.38877297e-3
    vfmadd.vv   v14, v17, v7        # c = c * sa + 4.16666567e-2
    vfmadd.vv   v14, v17, v8        # c = c * sa + -0.5
    vfmadd.vv   v14, v17, v9        # c = c * sa + 1.0

    # Approximate sine.
    vmv.v.v     v16, v10            # v16 = 2.86567956e-6f
    vfmadd.vv   v16, v17, v11       # s = s * sa + -1.98559923e-4
    vfmadd.vv   v16, v17, v12       # s = s * sa + 8.33338592e-3
    vfmadd.vv   v16, v17, v13       # s = s * sa + -0.166666672
    vfmul.vv    v18, v21, v17       # t = a * sa
    vfmadd.vv   v16, v18, v21       # s = s * t + a

    # Check the value of i and adjust the order of sine and cosine if needed
    vand.vi v0, v19, 1              # v0 = i & 1
    vmfne.vf v0, v0, ft11           # Set mask when true i.e not equal too zer0

    #Now we merge c(v14) and s(v16) to rc(v31) and rs(v30)
    vmerge.vvm v30, v16, v14, v0    #v0.mask[i] ? v14[i] : v16[i]
    vmerge.vvm v31, v14, v16, v0    #v0.mask[i] ? v14[i] : v16[i]
 
    vand.vi v0, v19, 2              # t0 = i & 2
    vmfne.vf v0, v0, ft11           # Set mask when true i.e not equal too zer0
    vfsgnjn.vv v30,v30,v30, v0.t    # negate rs where i&2 is true

    vand.vi v0, v20, 2              # t1 = ic & 2
    vmfne.vf v0, v0, ft11           # Set mask when true i.e not equal too zer0
    vfsgnjn.vv v31,v31,v31, v0.t    # Negate cosine if ic & 2 != 0

    ret                             # Return with sine in v30, cosine in v31



# Function: vOrdina
#   Computes bit-reversed indices for an array of elements and performs in-place swaps
#   on the real and imaginary arrays accordingly. This is a key step in the butterfly
#   operation for the Fast Fourier Transform (FFT).
# Inputs:
#   - a0: Base address of the real array
#   - a1: Base address of the imaginary array
#   - a2: Number of elements (N)
# Outputs:
#   - None. The real and imaginary arrays are modified in-place with elements
#     reordered according to their bit-reversed indices.
# Clobbers:
#   - v1, v2: vector registers used for bit manipulation and indexing
#   - t1, t2, t3, t4, t5, t6: Temporary registers used for intermediate calculations
#   - v23, v24, v26, v27, v29: Additional registers used during reordering
vOrdina:
    # Save used callee registers to stack
    addi sp, sp, -28                
    sw a7, 0(sp)
    sw ra, 4(sp)
    sw s1, 8(sp)
    sw s2, 12(sp)
    sw s3, 16(sp)
    sw s4, 20(sp)
    sw s5, 24(sp)

    # Load pointers to temp array. TODO: Replace them with sbrk or via argument
    la t4, real_temp                # t4    = real_temp[] pointer
    la t5, imag_temp                # t5    = imag_temp[] pointer 
    lw a7, logsize                  # log(N) used in reversing
    
    vsetvli t3, a2, e32, m1         # Request vector for a2 length
    vid.v  v26                      # v26 = {0, 1, 2, ... VLEN-1}

    # Load mask and shift bits for reverse. This is required for reverse function
    li s1, 0x55555555
    li s2, 0x33333333
    li s3, 0x0F0F0F0F
    li s4, 0x00FF00FF  
    li s5, 30                       # mask is 30 instead of 32 to add shift by 2 effect
    sub s5, s5, a7                  # a7 (logsize) will never be more than 30

    li t2, 0                
    vOrdinaLoop:                   
    bge t2, a2, endVOrdinaLoop      

    # Bit reverse the index in v26. Output in v29
    call vReverseIndexOffset

    # Load from normal array reversed indexed
    vloxei32.v v23, 0(a0), v29       
    vloxei32.v v24, 0(a1), v29     

    # Generate Index Offset
    vsll.vi v27, v26, 2   

    # Save to temp array normal index
    vsoxei32.v v23, 0(t4), v27            
    vsoxei32.v v24, 0(t5), v27           

    # Increment index and coutner
    vadd.vx v26, v26, t3           
    add t2, t2, t3            
    j vOrdinaLoop
    endVOrdinaLoop:

    vid.v v26                       # v26 = {0, 1, 2, ... VLEN-1}
    vsll.vi v26, v26, 2             # Shift the indexes by 4 so it matches array offsets
    slli t6, t3, 2                  # Shift VLEN by 4. Now  just add shifted vlen to shifted indexes

    li t1, 0              
    vOrdinaLoop2:                   
    bge t1, a2, endvOrdinaLoop2   

    # Indxed Load from temp array
    vloxei32.v v23, 0(t4), v26             
    vloxei32.v v24, 0(t5), v26            

    # Indxed Store to normal array
    vsoxei32.v v23, 0(a0), v26           
    vsoxei32.v v24, 0(a1), v26

    # Incrementing Indexes
    vadd.vx v26, v26, t6           
    add t1, t1, t3                 

    j vOrdinaLoop2              
    endvOrdinaLoop2:

    # Restore registers
    lw a7, 0(sp)
    lw ra, 4(sp)
    lw s1, 8(sp)
    lw s2, 12(sp)
    lw s3, 16(sp)
    lw s4, 20(sp)
    lw s5, 24(sp)
    addi sp, sp, 28              

    jr ra



# Function: vTransform
#   Implements the FFT (Fast Fourier Transform) or IFFT (Inverse FFT) using vectorized
#   operations for performance optimization. It involves the following steps:
#     1. Bit-reversed reordering of elements (via vOrdina).
#     2. Computing twiddle factors (W_real, W_imag) using Euler's formula.
#     3. Nested loops for the butterfly operation, performing in-place transforms.
#
# Inputs:
#   - a0: Base address of the real array
#   - a1: Base address of the imaginary array
#   - a2: Number of elements (N)
#   - a3: Inverse flag (1 for IFFT, 0 for FFT)
#
# Outputs:
#   - None. The real and imaginary arrays are modified in-place to contain the FFT/IFFT result.
#
# Clobbers:
#   - General-purpose: t0-t6, s0-s5, a4-a6
#   - Floating-point: ft1, ft3
#   - Vector: v0, v7-v10, v11-v19, v21, v22, v23-v25, v30, v31
vTransform:
    addi sp, sp, -4  
    sw ra, 0(sp)

    # Call ordina function to bitwise swap reverse indexed elements
    call vOrdina                    

    # Load addresses for W arrays for saving sin/cos
    la t1, W_real                   
    la t2, W_imag                

    # Loop for Sin/Cos Using Euler Formula
    vid.v v22                       # Index vector for help

    # Calculate (inverse) * -2PI/N 
    la t0, NEG_TWO_PI               # Load mem address of -2PI to t0
    flw ft1, 0(t0)                  # Load -2PI to ft1
    mul  t0, a3, a2                 # t0 = (inverse)*N
    fcvt.s.w ft3, t0                # ft3 = N
    fdiv.s ft1, ft1, ft3            # ft1 = ft1 / ft3 = (inverse) -2PI *  / N

    # Preload constatns to be used in sin/cos calculations
    call preload_constants          # uses v1-v13 and t0
    
    srai a4, a2, 1                  # a4    =   N / 2   = a / 2
    vsetvli t0, a4, e32             # Vector for N/2 elements
    li t3, 0                        # t3    = i = 0

    vsincosloop:                    # for loop i = 0; i < N / 2;
    bge t3, a4, endvsincosloop      # as soon as num element t0 >= N/2, break

    vadd.vx v23, v22, t3            # v23 = v22 + i => {i, i+1, i+2, ..., i + VLEN -1}. Rn its i integer
    vfcvt.f.x.v v21, v23            # Convert indexVector 0,1,2 to floats.                     i float
    vfmul.vf v21, v21, ft1          # v21[i] = (inverse * -2.0 * PI  / N )*  i . Now we need cos and sins of this

    # input in v21
    call v_sin_cos_approx          
    # sin in 30, cos in 31

    # Now, we have vector having cos, sin. Now we save to W_real, W_imag
    vsll.vi v23, v23, 2
    vsoxei32.v v31, 0(t1), v23              # W_real[i] = myCos(value);
    vsoxei32.v v30, 0(t2), v23            # W_imag[i] = mySin(value);

    add t3, t3, t0                  # i +=  VLEN
    j vsincosloop
    endvsincosloop:
   
    ##NOW STARTING NESTED LOOP

    li a5, 1                        # a5    = n     = 1
    srai a4, a2, 1                  # a4    = a     = N / 2
    li s0, 0                        # s0    = j     = 0

    lw a3, logsize                     # now a0 have logN
    vid.v v19

    # s3 is vlen*4
    slli s3, t0, 2


    forTransform:                   #int j = 0; j < logint(N); j++
    bge s0, a3, forTransformEnd     # End outer loop
    li s1, 0                        # s1 = i = 0
    
    mul s5, a5, a4                  # s5 = n*a  // shfted it out of loop

    # instead of recalculating (i+n)*4 and i*4
    # precalculate i and i+n, multiply by 4
    # keep adding vlen*4 to them in loop
    # v20 is i
    #v21 is i*4
    #v24 is i+n
    
    vsll.vi v21, v19, 2  # index*4
    vadd.vx v24, v19, a5  # i + n
    vsll.vi v24, v24, 2     # (i+n)*4

    # initializ i*a array then incrmement it in loop end by a*VLEN
    # s4 will be a*vlen
    vmul.vx v25, v19, a4
    mul s4, a4, t0
    # also shift n by 2 to calculate i*4 & n without doing one addition
    slli a6, a5, 2

    vinnerloop:                     # for i = 0; i < N
    bge s1, a2, vinnerloopend       # i  >= num elemenets
    
    # Calculate mask (i & n)
    vand.vx v18, v21, a6           
    vmseq.vx v0, v18, zero         # if (!(i & n)) which means this loop work only when result is 0
    # Start of If block. Every operation is masked wrt v0

    # Calculating k and offest
    vrem.vx v25, v25, s5, v0.t      
    vsll.vi v15, v25, 2, v0.t     

    # Loading values from  windex k
    vloxei32.v v13, 0(t1), v15, v0.t
    vloxei32.v v14, 0(t2), v15, v0.t 

    # Loading values from index i
    vloxei32.v v16, 0(a0), v21 , v0.t   
    vloxei32.v v17, 0(a1)  ,v21 , v0.t  

    # Loading values from index i+n
    vloxei32.v v11, 0(a0), v24, v0.t
    vloxei32.v v12, 0(a1), v24, v0.t

    vfmul.vv v7, v13, v11, v0.t     # v7 = wreal*real[i+n]
    vfnmsac.vv v7, v14, v12, v0.t   # v7 = v7 - v14*v12 = wreal*real[i+n] - wimag*imag[i+n]

    vfmul.vv v8, v13, v12, v0.t     # wrealk*imag{i+n}
    vfmacc.vv v8, v14, v11, v0.t    # v8 = wrealk*imag[i+n] + wrealk*real{i+n}
	
    vfadd.vv v9, v16, v7, v0.t
    vfadd.vv v10, v17, v8, v0.t
    vfsub.vv v5, v16, v7, v0.t
    vfsub.vv v6, v17, v8, v0.t

    # Saving values to index i
    vsoxei32.v v9 , 0(a0), v21, v0.t 
    vsoxei32.v v10, 0(a1), v21, v0.t 

    # Saving values to index i+n
    vsoxei32.v v5 , 0(a0), v24, v0.t  
    vsoxei32.v v6 , 0(a1), v24, v0.t

    # incremenet v21(i) and v24(i+n) by vlen*4 (s3)
    vadd.vx v21, v21, s3
    vadd.vx v24, v24, s3

    # incrmement i*a by a*VLEN
    vadd.vx v25, v25, s4
   
    add s1, s1, t0                  # i += Vlen
    j vinnerloop
    vinnerloopend:

    slli a5, a5, 1                  # n = n * 2 
    srai a4, a4, 1                  # a = a/2 
    addi s0, s0, 1                  # j++
    j forTransform
    forTransformEnd:
   

    lw ra, 0(sp)                    # Restore return address
    addi sp, sp, 4                  # Stack restored
    jr ra



# Function: vFFT
#   Performs the Fast Fourier Transform (FFT) or Inverse FFT (IFFT) on the input real and 
#   imaginary arrays by calling the `vTransform` function. This function sets up the
#   inverse flag for FFT and calls the main transform logic.
#
# Inputs:
#   - a0: Base address of the real array
#   - a1: Base address of the imaginary array
#   - a2: Number of elements (N)
#
# Outputs:
#   - None. The real and imaginary arrays are modified in-place to contain the FFT result.
#
# Registers clobbered:
#   - None explicitly.
vFFT:                     
    addi sp, sp, -4
    sw ra, 0(sp)

    li a3, 1                        # Inverse Flag a3 = 1 for FFT, -1 for IFFT
    call vTransform
   
    lw ra, 0(sp)
    addi sp, sp, 4
    
    jr ra
    

# Function: vIFFT
#   Performs the Inverse Fast Fourier Transform (IFFT) on the input real and imaginary arrays.
#   This function calls the vectorized `vTransform` for the main IFFT computation, then
#   scales the result by dividing each element of the output arrays by the total number of elements (N).
#
# Inputs:
#   - a0: Base address of the real array
#   - a1: Base address of the imaginary array
#   - a2: Number of elements (N)
#
# Outputs:
#   - None. The real and imaginary arrays are modified in-place to contain the scaled IFFT result.
#
# Registers clobbered:
#   - Temporary registers: t0-t4
#   - Floating-point register: ft0
vIFFT:              
    addi sp, sp, -16            # Save a0, a1 etc to stack because these addresses
    sw ra, 0(sp)                # Are modified when dividing
    sw a0, 4(sp)    
    sw a1, 8(sp)
    sw a3, 12(sp)
    
    li a3, -1                        # Inverse Flag. a3 = 1 for FFT, -1 for IFFT
    call vTransform
    
    
    vsetvli t0, a2, e32, m1         # GEt VLEN. Set vector length to a2, acutal lenght stored in t0
    slli t2, t0, 2                  # shift vlen by 2 for offest


    li t1, 0                        # i = 0. starting index
    fcvt.s.w ft0, a2                # Convert N t0 float as we have to divide

    vectorIFFTLoop:                 # for (int i = 0; i < N; i++)
    bge t1, a2, endVectorIFFTLoop   # break when i >= N

    # Load Real/Imag Pair 
    vle32.v v1, 0(a0)               # load t0 real values to vector v1
    vle32.v v2, 0(a1)               # load t0 imag values to vector v2

    # Divide by N
    vfdiv.vf v1, v1, ft0            # v1[i] = v1[i] / ft0 , ft0 is N in input
    vfdiv.vf v2, v2, ft0            # v2[i] = v2[i] / ft0 , ft0 is N in input

    # Save back to memory
    vse32.v v1, 0(a0)               # save result back to meme
    vse32.v v2, 0(a1)               # same as above

    # Increment address by VLEN*4
    add a0, a0, t2                  
    add a1, a1, t2                  

    add t1, t1, t0                 # i += VLEN
    j vectorIFFTLoop
    endVectorIFFTLoop:

    lw ra, 0(sp)
    lw a0, 4(sp)
    lw a1, 8(sp)
    lw a3, 12(sp)
    addi sp, sp, 16

    jr ra



# TODO change la to mv   
# Function: print
# Logs values from real[] and imag[] arrays into registers ft0 and ft1 for debugging and output.
# Inputs:
#   - a0: Base address of real[] array
#   - a1: Base address of imag[] array
#   - a2: Size of array i.e. number of elements to log
# Clobbers: t0,t1, t2,t3 ft0, ft1.
print:        
    addi sp, sp, -8
    sw a0, 0(sp)
    sw a1, 4(sp)    

    li t0, 0x123456                 # Pattern for help in python script
    li t0, 0x234567                 # Pattern for help in python script
    li t0, 0x345678                 # Pattern for help in python script

	li t0, 0		                # load i = 0

    vsetvli t3, a2, e32             # Set vlen to a2, save VLEN in t3
    slli t4, t3, 2                  # vlen*4 for address incre,net

    printloop:
    bge t0, a2, endPrintLoop        # Exit loop if i >= size

    vle32.v v1, 0(a0)                  # Load real[i] into v1
    vle32.v v1, 0(a1)                  # Load imag[i] into v1

    add a0, a0, t4                  # Increment pointer for real[] by VLEN*4
    add a1, a1, t4                  # Increment pointer for imag[] by VLEN*4

    add t0, t0, t3                  # Increment index
    j printloop                     # Jump to start of loop
    endPrintLoop:

    li t0, 0x123456                 # Pattern for help in python script
    li t0, 0x234567                 # Pattern for help in python script
    li t0, 0x345678                 # Pattern for help in python script
	
    lw a0, 0(sp)
    lw a1, 4(sp) 
    addi sp, sp, 8

	jr ra



_finish:
    li x3, 0xd0580000
    addi x5, x0, 0xff
    sb x5, 0(x3)
    beq x0, x0, _finish
.rept 100
    nop
.endr

# PUT INPUT HERE, DO NO CHANHE ABOVE THIS
.data  
real:
.float 0.000000000000, 0.926918268204, 0.919169723988, 0.990578532219,  1.303665637970, 0.808001399040, 0.185667261481, 0.480334579945,  0.692492425442, 0.312599480152, 0.386258661747, 0.549540817738,  -0.265722781420, -1.102399468422, -1.122005224228, -1.308735251427,  -1.611502528191, -0.937327146530, -0.075590953231, -0.065600380301,  0.037101194263, 0.525623679161, 0.314979910851, -0.071582920849,  0.478508591652, 1.037295222282, 0.936154186726, 1.177533030510,  1.518425822258, 0.808750391006, -0.120916709304, -0.308403402567
.float -0.670551419258, -1.275333166122, -0.966756165028, -0.369217187166,  -0.595488190651, -0.761930763721, -0.420918762684, -0.644625961781,  -1.046460032463, -0.436431139708, 0.360446602106, 0.534545481205,  1.028624296188, 1.714458465576, 1.360914587975, 0.638152837753,  0.592721223831, 0.361965686083, -0.267572462559, -0.120013013482,  0.333360344172, -0.090862624347, -0.587445020676, -0.554717063904,  -1.019855380058, -1.715170502663, -1.363277435303, -0.644941747189,  -0.471428960562, 0.049748931080, 0.925872862339, 0.876332044601
.float 0.407106786966, 0.631873965263, 0.745003879070, 0.372024744749,  0.665025115013, 1.282416224480, 0.956648051739, 0.374247580767,  0.259995639324, -0.366485118866, -1.365330219269, -1.387659430504,  -0.949260950089, -1.032721400261, -0.787449836731, -0.044835571200,  -0.087211810052, -0.551729738712, -0.248636275530, 0.104984961450,  -0.010353486985, 0.516224622726, 1.467689752579, 1.492766141891,  1.124644517899, 1.173250198364, 0.692374587059, -0.333901703358,  -0.528410375118, -0.252805680037, -0.558883130550, -0.655717790127
.float -0.211747497320, -0.477967590094, -1.217702984810, -1.154460191727,  -0.874869465828, -1.005793094635, -0.469020694494, 0.664784729481,  0.991863608360, 0.887315750122, 1.229989647865, 1.111086726189,  0.341538578272, 0.282513827085, 0.703803420067, 0.469994395971,  0.271479845047, 0.573745012283, 0.159973710775, -0.868948400021,  -1.164918899536, -1.162372231483, -1.566517591476, -1.324113607407,  -0.332227855921, 0.000774180284, -0.087881229818, 0.359588980675,  0.503921091557, -0.002954887692, 0.165761619806, 0.906659364700
.float 1.000000000000, 0.999952673912, 1.469408035278, 1.213976979256,  0.169858634472, -0.285974532366, -0.445544302464, -1.091424107552,  -1.213793039322, -0.532898485661, -0.428063303232, -0.783507645130,  -0.550538361073, -0.455883175135, -0.970456361771, -0.794189393520,  0.118700794876, 0.494373142719, 0.744690775871, 1.514835119247,  1.636814475060, 0.863516092300, 0.556458950043, 0.544145941734,  -0.049070127308, -0.298704952002, 0.224199354649, 0.172934025526,  -0.468248426914, -0.573485553265, -0.733701348305, -1.514657616615
.float -1.636011600494, -0.874616742134, -0.511112153530, -0.256925642490,  0.626545727253, 1.030739068985, 0.538088619709, 0.475928366184,  0.789236128330, 0.507083296776, 0.431366592646, 1.104689717293,  1.201752066612, 0.545635640621, 0.297718733549, -0.005309666041,  -1.023928642273, -1.516638875008, -1.078535795212, -0.963514864445,  -0.990801572800, -0.315050989389, 0.058984577656, -0.421498954296,  -0.456029921770, 0.039975184947, 0.029148327187, 0.182892262936,  1.142953515053, 1.609780550003, 1.225551366806, 1.143405318260
.float 1.007106781006, 0.044745407999, -0.583949029446, -0.318910360336,  -0.383681237698, -0.716260254383, -0.376896530390, -0.241464272141,  -0.970514059067, -1.283510684967, -0.926972687244, -0.958316743374,  -0.819320738316, 0.242654487491, 0.987362504005, 0.887753307819,  1.070217967033, 1.282708168030, 0.638483047485, 0.176756143570,  0.577790737152, 0.637050211430, 0.266916900873, 0.460974216461,  0.465890675783, -0.486837655306, -1.155774116516, -1.112747192383,  -1.402102708817, -1.564735770226, -0.722950935364, -0.012329882942
.float -0.094815470278, 0.136808797717, 0.559119164944, 0.198326364160,  -0.036523614079, 0.640802979469, 1.048804879189, 0.928915739059,  1.285333991051, 1.468335390091, 0.584242761135, -0.207907393575,  -0.330562144518, -0.811632096767, -1.303490161896, -0.814813077450,  -0.350068032742, -0.679078698158, -0.706414341927, -0.395873308182,  -0.762482702732, -1.011656999588, -0.239129543304, 0.430034935474,  0.576294124126, 1.196114778519, 1.741884827614, 1.190643429756,  0.579422533512, 0.601095318794, 0.232446655631, -0.322906345129
.float -0.000000000000, 0.322906345129, -0.232446655631, -0.601095318794,  -0.579422533512, -1.190643429756, -1.741884827614, -1.196114778519,  -0.576294124126, -0.430034935474, 0.239129543304, 1.011656999588,  0.762482702732, 0.395873308182, 0.706414341927, 0.679078698158,  0.350068032742, 0.814813077450, 1.303490161896, 0.811632096767,  0.330562144518, 0.207907393575, -0.584242761135, -1.468335390091,  -1.285333991051, -0.928915739059, -1.048804879189, -0.640802979469,  0.036523614079, -0.198326364160, -0.559119164944, -0.136808797717
.float 0.094815470278, 0.012329882942, 0.722950935364, 1.564735770226,  1.402102708817, 1.112747192383, 1.155774116516, 0.486837655306,  -0.465890675783, -0.460974216461, -0.266916900873, -0.637050211430,  -0.577790737152, -0.176756143570, -0.638483047485, -1.282708168030,  -1.070217967033, -0.887753307819, -0.987362504005, -0.242654487491,  0.819320738316, 0.958316743374, 0.926972687244, 1.283510684967,  0.970514059067, 0.241464272141, 0.376896530390, 0.716260254383,  0.383681237698, 0.318910360336, 0.583949029446, -0.044745407999
.float -1.007106781006, -1.143405318260, -1.225551366806, -1.609780550003,  -1.142953515053, -0.182892262936, -0.029148327187, -0.039975184947,  0.456029921770, 0.421498954296, -0.058984577656, 0.315050989389,  0.990801572800, 0.963514864445, 1.078535795212, 1.516638875008,  1.023928642273, 0.005309666041, -0.297718733549, -0.545635640621,  -1.201752066612, -1.104689717293, -0.431366592646, -0.507083296776,  -0.789236128330, -0.475928366184, -0.538088619709, -1.030739068985,  -0.626545727253, 0.256925642490, 0.511112153530, 0.874616742134
.float 1.636011600494, 1.514657616615, 0.733701348305, 0.573485553265,  0.468248426914, -0.172934025526, -0.224199354649, 0.298704952002,  0.049070127308, -0.544145941734, -0.556458950043, -0.863516092300,  -1.636814475060, -1.514835119247, -0.744690775871, -0.494373142719,  -0.118700794876, 0.794189393520, 0.970456361771, 0.455883175135,  0.550538361073, 0.783507645130, 0.428063303232, 0.532898485661,  1.213793039322, 1.091424107552, 0.445544302464, 0.285974532366,  -0.169858634472, -1.213976979256, -1.469408035278, -0.999952673912
.float -1.000000000000, -0.906659364700, -0.165761619806, 0.002954887692,  -0.503921091557, -0.359588980675, 0.087881229818, -0.000774180284,  0.332227855921, 1.324113607407, 1.566517591476, 1.162372231483,  1.164918899536, 0.868948400021, -0.159973710775, -0.573745012283,  -0.271479845047, -0.469994395971, -0.703803420067, -0.282513827085,  -0.341538578272, -1.111086726189, -1.229989647865, -0.887315750122,  -0.991863608360, -0.664784729481, 0.469020694494, 1.005793094635,  0.874869465828, 1.154460191727, 1.217702984810, 0.477967590094
.float 0.211747497320, 0.655717790127, 0.558883130550, 0.252805680037,  0.528410375118, 0.333901703358, -0.692374587059, -1.173250198364,  -1.124644517899, -1.492766141891, -1.467689752579, -0.516224622726,  0.010353486985, -0.104984961450, 0.248636275530, 0.551729738712,  0.087211810052, 0.044835571200, 0.787449836731, 1.032721400261,  0.949260950089, 1.387659430504, 1.365330219269, 0.366485118866,  -0.259995639324, -0.374247580767, -0.956648051739, -1.282416224480,  -0.665025115013, -0.372024744749, -0.745003879070, -0.631873965263
.float -0.407106786966, -0.876332044601, -0.925872862339, -0.049748931080,  0.471428960562, 0.644941747189, 1.363277435303, 1.715170502663,  1.019855380058, 0.554717063904, 0.587445020676, 0.090862624347,  -0.333360344172, 0.120013013482, 0.267572462559, -0.361965686083,  -0.592721223831, -0.638152837753, -1.360914587975, -1.714458465576,  -1.028624296188, -0.534545481205, -0.360446602106, 0.436431139708,  1.046460032463, 0.644625961781, 0.420918762684, 0.761930763721,  0.595488190651, 0.369217187166, 0.966756165028, 1.275333166122
.float 0.670551419258, 0.308403402567, 0.120916709304, -0.808750391006,  -1.518425822258, -1.177533030510, -0.936154186726, -1.037295222282,  -0.478508591652, 0.071582920849, -0.314979910851, -0.525623679161,  -0.037101194263, 0.065600380301, 0.075590953231, 0.937327146530,  1.611502528191, 1.308735251427, 1.122005224228, 1.102399468422,  0.265722781420, -0.549540817738, -0.386258661747, -0.312599480152,  -0.692492425442, -0.480334579945, -0.185667261481, -0.808001399040,  -1.303665637970, -0.990578532219, -0.919169723988, -0.926918268204
.float -0.000000000000, 0.926918268204, 0.919169723988, 0.990578532219,  1.303665637970, 0.808001399040, 0.185667261481, 0.480334579945,  0.692492425442, 0.312599480152, 0.386258661747, 0.549540817738,  -0.265722781420, -1.102399468422, -1.122005224228, -1.308735251427,  -1.611502528191, -0.937327146530, -0.075590953231, -0.065600380301,  0.037101194263, 0.525623679161, 0.314979910851, -0.071582920849,  0.478508591652, 1.037295222282, 0.936154186726, 1.177533030510,  1.518425822258, 0.808750391006, -0.120916709304, -0.308403402567
.float -0.670551419258, -1.275333166122, -0.966756165028, -0.369217187166,  -0.595488190651, -0.761930763721, -0.420918762684, -0.644625961781,  -1.046460032463, -0.436431139708, 0.360446602106, 0.534545481205,  1.028624296188, 1.714458465576, 1.360914587975, 0.638152837753,  0.592721223831, 0.361965686083, -0.267572462559, -0.120013013482,  0.333360344172, -0.090862624347, -0.587445020676, -0.554717063904,  -1.019855380058, -1.715170502663, -1.363277435303, -0.644941747189,  -0.471428960562, 0.049748931080, 0.925872862339, 0.876332044601
.float 0.407106786966, 0.631873965263, 0.745003879070, 0.372024744749,  0.665025115013, 1.282416224480, 0.956648051739, 0.374247580767,  0.259995639324, -0.366485118866, -1.365330219269, -1.387659430504,  -0.949260950089, -1.032721400261, -0.787449836731, -0.044835571200,  -0.087211810052, -0.551729738712, -0.248636275530, 0.104984961450,  -0.010353486985, 0.516224622726, 1.467689752579, 1.492766141891,  1.124644517899, 1.173250198364, 0.692374587059, -0.333901703358,  -0.528410375118, -0.252805680037, -0.558883130550, -0.655717790127
.float -0.211747497320, -0.477967590094, -1.217702984810, -1.154460191727,  -0.874869465828, -1.005793094635, -0.469020694494, 0.664784729481,  0.991863608360, 0.887315750122, 1.229989647865, 1.111086726189,  0.341538578272, 0.282513827085, 0.703803420067, 0.469994395971,  0.271479845047, 0.573745012283, 0.159973710775, -0.868948400021,  -1.164918899536, -1.162372231483, -1.566517591476, -1.324113607407,  -0.332227855921, 0.000774180284, -0.087881229818, 0.359588980675,  0.503921091557, -0.002954887692, 0.165761619806, 0.906659364700
.float 1.000000000000, 0.999952673912, 1.469408035278, 1.213976979256,  0.169858634472, -0.285974532366, -0.445544302464, -1.091424107552,  -1.213793039322, -0.532898485661, -0.428063303232, -0.783507645130,  -0.550538361073, -0.455883175135, -0.970456361771, -0.794189393520,  0.118700794876, 0.494373142719, 0.744690775871, 1.514835119247,  1.636814475060, 0.863516092300, 0.556458950043, 0.544145941734,  -0.049070127308, -0.298704952002, 0.224199354649, 0.172934025526,  -0.468248426914, -0.573485553265, -0.733701348305, -1.514657616615
.float -1.636011600494, -0.874616742134, -0.511112153530, -0.256925642490,  0.626545727253, 1.030739068985, 0.538088619709, 0.475928366184,  0.789236128330, 0.507083296776, 0.431366592646, 1.104689717293,  1.201752066612, 0.545635640621, 0.297718733549, -0.005309666041,  -1.023928642273, -1.516638875008, -1.078535795212, -0.963514864445,  -0.990801572800, -0.315050989389, 0.058984577656, -0.421498954296,  -0.456029921770, 0.039975184947, 0.029148327187, 0.182892262936,  1.142953515053, 1.609780550003, 1.225551366806, 1.143405318260
.float 1.007106781006, 0.044745407999, -0.583949029446, -0.318910360336,  -0.383681237698, -0.716260254383, -0.376896530390, -0.241464272141,  -0.970514059067, -1.283510684967, -0.926972687244, -0.958316743374,  -0.819320738316, 0.242654487491, 0.987362504005, 0.887753307819,  1.070217967033, 1.282708168030, 0.638483047485, 0.176756143570,  0.577790737152, 0.637050211430, 0.266916900873, 0.460974216461,  0.465890675783, -0.486837655306, -1.155774116516, -1.112747192383,  -1.402102708817, -1.564735770226, -0.722950935364, -0.012329882942
.float -0.094815470278, 0.136808797717, 0.559119164944, 0.198326364160,  -0.036523614079, 0.640802979469, 1.048804879189, 0.928915739059,  1.285333991051, 1.468335390091, 0.584242761135, -0.207907393575,  -0.330562144518, -0.811632096767, -1.303490161896, -0.814813077450,  -0.350068032742, -0.679078698158, -0.706414341927, -0.395873308182,  -0.762482702732, -1.011656999588, -0.239129543304, 0.430034935474,  0.576294124126, 1.196114778519, 1.741884827614, 1.190643429756,  0.579422533512, 0.601095318794, 0.232446655631, -0.322906345129
.float -0.000000000000, 0.322906345129, -0.232446655631, -0.601095318794,  -0.579422533512, -1.190643429756, -1.741884827614, -1.196114778519,  -0.576294124126, -0.430034935474, 0.239129543304, 1.011656999588,  0.762482702732, 0.395873308182, 0.706414341927, 0.679078698158,  0.350068032742, 0.814813077450, 1.303490161896, 0.811632096767,  0.330562144518, 0.207907393575, -0.584242761135, -1.468335390091,  -1.285333991051, -0.928915739059, -1.048804879189, -0.640802979469,  0.036523614079, -0.198326364160, -0.559119164944, -0.136808797717
.float 0.094815470278, 0.012329882942, 0.722950935364, 1.564735770226,  1.402102708817, 1.112747192383, 1.155774116516, 0.486837655306,  -0.465890675783, -0.460974216461, -0.266916900873, -0.637050211430,  -0.577790737152, -0.176756143570, -0.638483047485, -1.282708168030,  -1.070217967033, -0.887753307819, -0.987362504005, -0.242654487491,  0.819320738316, 0.958316743374, 0.926972687244, 1.283510684967,  0.970514059067, 0.241464272141, 0.376896530390, 0.716260254383,  0.383681237698, 0.318910360336, 0.583949029446, -0.044745407999
.float -1.007106781006, -1.143405318260, -1.225551366806, -1.609780550003,  -1.142953515053, -0.182892262936, -0.029148327187, -0.039975184947,  0.456029921770, 0.421498954296, -0.058984577656, 0.315050989389,  0.990801572800, 0.963514864445, 1.078535795212, 1.516638875008,  1.023928642273, 0.005309666041, -0.297718733549, -0.545635640621,  -1.201752066612, -1.104689717293, -0.431366592646, -0.507083296776,  -0.789236128330, -0.475928366184, -0.538088619709, -1.030739068985,  -0.626545727253, 0.256925642490, 0.511112153530, 0.874616742134
.float 1.636011600494, 1.514657616615, 0.733701348305, 0.573485553265,  0.468248426914, -0.172934025526, -0.224199354649, 0.298704952002,  0.049070127308, -0.544145941734, -0.556458950043, -0.863516092300,  -1.636814475060, -1.514835119247, -0.744690775871, -0.494373142719,  -0.118700794876, 0.794189393520, 0.970456361771, 0.455883175135,  0.550538361073, 0.783507645130, 0.428063303232, 0.532898485661,  1.213793039322, 1.091424107552, 0.445544302464, 0.285974532366,  -0.169858634472, -1.213976979256, -1.469408035278, -0.999952673912
.float -1.000000000000, -0.906659364700, -0.165761619806, 0.002954887692,  -0.503921091557, -0.359588980675, 0.087881229818, -0.000774180284,  0.332227855921, 1.324113607407, 1.566517591476, 1.162372231483,  1.164918899536, 0.868948400021, -0.159973710775, -0.573745012283,  -0.271479845047, -0.469994395971, -0.703803420067, -0.282513827085,  -0.341538578272, -1.111086726189, -1.229989647865, -0.887315750122,  -0.991863608360, -0.664784729481, 0.469020694494, 1.005793094635,  0.874869465828, 1.154460191727, 1.217702984810, 0.477967590094
.float 0.211747497320, 0.655717790127, 0.558883130550, 0.252805680037,  0.528410375118, 0.333901703358, -0.692374587059, -1.173250198364,  -1.124644517899, -1.492766141891, -1.467689752579, -0.516224622726,  0.010353486985, -0.104984961450, 0.248636275530, 0.551729738712,  0.087211810052, 0.044835571200, 0.787449836731, 1.032721400261,  0.949260950089, 1.387659430504, 1.365330219269, 0.366485118866,  -0.259995639324, -0.374247580767, -0.956648051739, -1.282416224480,  -0.665025115013, -0.372024744749, -0.745003879070, -0.631873965263
.float -0.407106786966, -0.876332044601, -0.925872862339, -0.049748931080,  0.471428960562, 0.644941747189, 1.363277435303, 1.715170502663,  1.019855380058, 0.554717063904, 0.587445020676, 0.090862624347,  -0.333360344172, 0.120013013482, 0.267572462559, -0.361965686083,  -0.592721223831, -0.638152837753, -1.360914587975, -1.714458465576,  -1.028624296188, -0.534545481205, -0.360446602106, 0.436431139708,  1.046460032463, 0.644625961781, 0.420918762684, 0.761930763721,  0.595488190651, 0.369217187166, 0.966756165028, 1.275333166122
.float 0.670551419258, 0.308403402567, 0.120916709304, -0.808750391006,  -1.518425822258, -1.177533030510, -0.936154186726, -1.037295222282,  -0.478508591652, 0.071582920849, -0.314979910851, -0.525623679161,  -0.037101194263, 0.065600380301, 0.075590953231, 0.937327146530,  1.611502528191, 1.308735251427, 1.122005224228, 1.102399468422,  0.265722781420, -0.549540817738, -0.386258661747, -0.312599480152,  -0.692492425442, -0.480334579945, -0.185667261481, -0.808001399040,  -1.303665637970, -0.990578532219, -0.919169723988, -0.926918268204
.float

imag:
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,  0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
.float

.set dataSize, 1024

    # DO NOT CHANGE ANYTHING AFTER THIS LINE

    .set halfDataSize, dataSize/2 
    size: .word dataSize
    step: .float 1.0
    logsize: .word 0

    real_temp: 
        .rept dataSize
        .float 0
        .endr

    imag_temp:
        .rept dataSize
        .float 0
        .endr

    W_real:
        .rept halfDataSize
        .float 0
        .endr

    W_imag:
        .rept halfDataSize
        .float 0
        .endr

    PI: .float 3.14159265358979323846
    NEG_PI: .float -3.14159265358979323846
    TWO_PI: .float 6.28318530717958647692
    NEG_TWO_PI: .float -6.28318530717958647692
    HALF_PI: .float 1.57079632679489661923
    NEG_HALF_PI: .float -1.57079632679489661923
    ONE: .float 1
    TERMS: .word 14

    half_pi_hi:    .float 1.57079637e+0  # π/2 high part
    half_pi_lo:    .float -4.37113883e-8 # π/2 low part
    const_2_pi:    .float 6.36619747e-1  # 2/π
    const_12582912: .float 12582912.0    # 1.5 * 2^23
    cos_coeff_0:   .float 2.44677067e-5  # Coefficient for cosine
    cos_coeff_1:   .float -1.38877297e-3
    cos_coeff_2:   .float 4.16666567e-2
    cos_coeff_3:   .float -5.00000000e-1
    cos_coeff_4:   .float 1.00000000e+0
    sin_coeff_0:   .float 2.86567956e-6  # Coefficient for sine
    sin_coeff_1:   .float -1.98559923e-4
    sin_coeff_2:   .float 8.33338592e-3
    sin_coeff_3:   .float -1.66666672e-1
