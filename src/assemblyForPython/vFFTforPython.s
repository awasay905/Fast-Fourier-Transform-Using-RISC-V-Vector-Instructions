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

    # Perform XXXX
    call XXXX
    
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
