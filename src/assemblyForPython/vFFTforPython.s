.section .text
.global _start
_start:

main:    
    la sp, STACK
    la ra, _finish

    lw a0, size                     # Load size of real/imag arrays into a0
    call setlogN                    # Compute and store log2(size) for shared use by other functions

    
    # Compute and save TwiddleFactor      
    la a0, W_real
    la a1, W_imag
    lw a2, size
    call compute_twiddle_factors

    # Perform  vFFT
    la a0, real                     # a0 = address of real[]
    la a1, imag                     # a1 = address of imag[]
    lw a2, size                     # a2 = size of arrays (N)

    call XXXX
    

    # Print Results
    call print

    # End Program
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





# Function: compute_twiddle_factors
# Computes and stores N/2 complex FFT twiddle factors (W_k = cos(theta) - j*sin(theta))
# where theta = dir * -2*pi*k/N, for k = 0 to N/2-1.
# Must be called once before FFT/IFFT operations that require these factors.
# Inputs:
#   - W_real (address loaded into a0 before call)
#   - W_imag (address loaded into a1 before call)
#   - a2: FFT Size (N, must be power-of-2)
# Outputs:
#   - Writes N/2 cosine values to the W_real array (pointed to by a0).
#   - Writes N/2 sine values to the W_imag array (pointed to by a1).
# Clobbers: Various temporary (t*, ft*) and vector (v*) registers used internally.
compute_twiddle_factors:
    addi sp, sp, -16
    sw ra, 0(sp)
    sw a0, 4(sp)
    sw a1, 8(sp)
    sw a4, 12(sp)

    # Setting Up Vector For N/2 Elements
    srai a4, a2, 1                      # a4  = N / 2
    vsetvli t1, a4, e32

    # Preload constatns to be used in sin/cos calculations
    la      t0, half_pi_hi              # Load address of half_pi_hi

    # Make use of the fact that all float are 4 bytes and stored consecutively
    flw     ft0, 0(t0)                  # Load half_pi_hi
    vfmv.v.f v1, ft0                
    flw     ft0, 4(t0)                  # Load half_pi_lo
    vfmv.v.f v20, ft0               
    flw     ft0, 8(t0)                  # Load const_2_pi
    vfmv.v.f v3, ft0                
    flw     ft0, 12(t0)                 # Load const_12582912
    vfmv.v.f v4, ft0                
    flw     ft0, 16(t0)                 # Load cos_coeff_0
    vfmv.v.f v5, ft0                
    flw     ft0, 20(t0)                 # Load cos_coeff_1
    vfmv.v.f v6, ft0                
    flw     ft0, 24(t0)                 # Load cos_coeff_2
    vfmv.v.f v7, ft0                
    flw     ft0, 28(t0)                 # Load cos_coeff_3
    vfmv.v.f v8, ft0                
    flw     ft0, 32(t0)                 # Load cos_coeff_4
    vfmv.v.f v9, ft0                
    flw     ft0, 36(t0)                 # Load sin_coeff_0
    vfmv.v.f v10, ft0               
    flw     ft0, 40(t0)                 # Load sin_coeff_1
    vfmv.v.f v11, ft0               
    flw     ft0, 44(t0)                 # Load sin_coeff_2
    vfmv.v.f v12, ft0               
    flw     ft0, 48(t0)                 # Load sin_coeff_3
    vfmv.v.f v13, ft0               

    # Calculating -2PI/N 
    la t0, NEG_TWO_PI
    flw ft1, 0(t0)
    fcvt.s.w ft3, a2                    # ft3 = N
    fdiv.s ft1, ft1, ft3                # ft1 = -2PI *  / N

    slli t2, t1, 2             # shift for address
    vid.v   v2                          # Index vector 

    li t0, 0                            # t0    = i = 0
    vsincosloop:
        bge t0, a4, endvsincosloop 

        vfcvt.f.x.v v21, v2             # Convert indexVector 0,1,2 to floats.                     i float
        vadd.vx v2, v2, t1              # v23 = v2 + i => {i, i+1, i+2, ..., i + VLEN -1}. Rn its i integer
        vfmul.vf v21, v21, ft1          # v21[i] = (-2.0 * PI  / N )*  i . Now we need cos and sins of this

        vmv.v.v   v15, v4               # v15 = 12582912
        vfmacc.vv  v15, v21, v3         # a(v21))*6.36619747e-1f(v3) + 12582912.f(v15)
        vfsub.vv    v15, v15, v4        # j = fmaf(a, 6.36619747e-1f, 12582912.f) - 12582912.f;

        vfnmsac.vv v21, v15, v1         #  a = fmaf (j, -half_pi_hi, a);
        vfnmsac.vv v21, v15, v20        #  a = fmaf (j, -half_pi_lo, a);

        vfcvt.x.f.v v26, v15            #  i = (int) j
        vadd.vi v22, v26, 1             # ic = i + 1

        vfmul.vv  v17, v21, v21         # sa = a * a (sa)

        # Approximate cosine.
        vmv.v.v     v14, v5             # c = 2.44677067e-5f
        vfmadd.vv   v14, v17, v6        # c = c * sa + -1.38877297e-3
        vfmadd.vv   v14, v17, v7        # c = c * sa + 4.16666567e-2
        vfmadd.vv   v14, v17, v8        # c = c * sa + -0.5
        vfmadd.vv   v14, v17, v9        # c = c * sa + 1.0

        # Approximate sine.
        vmv.v.v     v24, v10            # v24 = 2.86567956e-6f
        vfmadd.vv   v24, v17, v11       # s = s * sa + -1.98559923e-4
        vfmadd.vv   v24, v17, v12       # s = s * sa + 8.33338592e-3
        vfmadd.vv   v24, v17, v13       # s = s * sa + -0.166666672
        vfmul.vv    v18, v21, v17       # t = a * sa
        vfmadd.vv   v24, v18, v21       # s = s * t + a

        # Check the value of i and adjust the order of sine and cosine if needed
        vand.vi v0, v26, 1              # v0 = i & 1
        vmseq.vi v0, v0, 1              # Set mask

        #Now we merge c(v14) and s(v24) to rc(v30) and rs(v28)
        vmerge.vvm v28, v24, v14, v0    #v0.mask[i] ? v14[i] : v24[i]
        vmerge.vvm v30, v14, v24, v0    #v0.mask[i] ? v14[i] : v24[i]
    
        vand.vi v0, v26, 2              # v0 = i & 2
        vmseq.vi v0, v0, 2              # Set mask
        vfsgnjn.vv v28,v28,v28, v0.t    # negate rs where i&2 is true

        vand.vi v0, v22, 2              # a0 = ic & 2
        vmseq.vi v0, v0, 2              # Set mask
        vfsgnjn.vv v30,v30,v30, v0.t    # Negate cosine if ic & 2 != 0
            
        # Save to Memory
        vse32.v v28, 0(a1)              # W_imag[i] = mySin(value);
        vse32.v v30, 0(a0)              # W_real[i] = myCos(value);

        add t0, t0, t1                  # i +=  VLEN
        add a0, a0, t2
        add a1, a1, t2
        j vsincosloop
    endvsincosloop:

    lw ra, 0(sp)
    lw a0, 4(sp)
    lw a1, 8(sp)
    lw a4, 12(sp)
    addi sp, sp, 16
    ret



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
#   - v20, v24, v0, v16, v12: Additional registers used during reordering
vOrdina:
    # Save used callee registers to stack
    addi sp, sp, -36       
    sw ra, 0(sp)
    sw a0, 4(sp)
    sw a1, 8(sp)
    sw a3, 12(sp)
    sw a4, 16(sp)
    sw a5, 20(sp)
    sw a6, 24(sp)
    sw a7, 28(sp)
    sw s0, 32(sp)


    # Load pointers to temp array
    la t0, real_temp
    la t1, imag_temp
    lw s0, logsize
    
    vsetvli t3, a2, e32, m8         # 4-Grouped Vector
    slli t6, t3, 2
    vid.v  v0                      # v0 = {0, 1, 2, ... VLEN-1}

    # Load mask and shift bits for reverse. This is required for reverse function
    li a3, 0x55555555
    li a4, 0x33333333
    li a5, 0x0F0F0F0F
    li a6, 0x00FF00FF  
    li a7, 30                       # mask is 30 instead of 32 to add shift by 2 effect
    sub a7, a7, s0                  # s0 (LOG2_SIZE) will never be more than 30

    li t2, 0                
    vOrdinaLoop:                   
    bge t2, a2, endVOrdinaLoop      

    # Bit reverse the index in v0. Output in v16
    # Swap odd and even bits
    vsrl.vi v16, v0, 1                  # v16 >> 1
    vand.vx v16, v16, a3                # (v16 >> 1) & 0x55555555
    vand.vx v8, v0, a3                  # v16 & 0x55555555
    vsll.vi v8, v8, 1                   # (v16 & 0x55555555) << 1
    vor.vv v16, v16, v8                 # Result back to v16

    # Swap consecutive pairs
    vsrl.vi v8, v16, 2                  # v16 >> 2
    vand.vx v8, v8, a4                  # (v16 >> 2) & 0x33333333
    vand.vx v16, v16, a4                # v16 & 0x33333333
    vsll.vi v16, v16, 2                 # (v16 & 0x33333333) << 2
    vor.vv v16, v8, v16                 # Result back to v16

    # Swap nibbles
    vsrl.vi v8, v16, 4                  # v16 >> 4
    vand.vx v8, v8, a5                  # (v16 >> 4) & 0x0F0F0F0F
    vand.vx v16, v16, a5                # v16 & 0x0F0F0F0F
    vsll.vi v16, v16, 4                 # (v16 & 0x0F0F0F0F) << 4
    vor.vv v16, v8, v16                 # Result back to v16

    # Swap bytes
    vsrl.vi v8, v16, 8                  # v16 >> 8
    vand.vx v8, v8, a6                  # (v16 >> 8) & 0x00FF00FF
    vand.vx v16, v16, a6                # v16 & 0x00FF00FF
    vsll.vi v16, v16, 8                 # (v16 & 0x00FF00FF) << 8
    vor.vv v16, v8, v16                 # Result back to v16

    # Swap 2-byte pairs
    vsrl.vi v8, v16, 16                 # v16 >> 16
    vsll.vi v16, v16, 16                # v16 << 16
    vor.vv v16, v8, v16                 # Final result in v16

    # Shift by the req bit size
    vsrl.vx v16, v16, a7

    # Load from normal array reversed indexed
    vloxei32.v v8, 0(a0), v16       
    vloxei32.v v24, 0(a1), v16     

    # Save to temp array normal index
    vse32.v v8, 0(t0)          
    vse32.v v24, 0(t1)  

    # Increment index and coutner
    vadd.vx v0, v0, t3           
    add t2, t2, t3        
    add t0, t0, t6 
    add t1, t1, t6   
    j vOrdinaLoop
    endVOrdinaLoop:

    slli t6, t3, 2                  # Shift VLEN by 4. Now  just add shifted vlen to shifted indexes
    la t4, real_temp
    la t5, imag_temp
    li t1, 0              
    vOrdinaLoop2:                   
    bge t1, a2, endvOrdinaLoop2   

    # Load from temp array
    vle32.v v8, 0(t4)             
    vle32.v v24, 0(t5)            

    # Store to normal array
    vse32.v v8, 0(a0)           
    vse32.v v24, 0(a1)

    add t1, t1, t3     
    add t4, t4, t6            
    add t5, t5, t6            
    add a0, a0, t6            
    add a1, a1, t6            

    j vOrdinaLoop2              
    endvOrdinaLoop2:

    # Restore registers
    lw ra, 0(sp)
    lw a0, 4(sp)
    lw a1, 8(sp)
    lw a3, 12(sp)
    lw a4, 16(sp)
    lw a5, 20(sp)
    lw a6, 24(sp)
    lw a7, 28(sp)
    lw s0, 32(sp)    
    addi sp, sp, 36       

    ret



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
#   - Vector: v0, v7-v10, v11-v26, v21, v20, v23-v25, v28, v30
vTransform:
    addi sp, sp, -36
    sw ra,0(sp)
    sw a0,4(sp)
    sw a1,8(sp)
    sw a2,12(sp)
    sw a3,16(sp)
    sw a4,20(sp)
    sw a5,24(sp)
    sw a6,28(sp)
    sw s0,32(sp)

    # For Index Reverse (Butterfly)
    call vOrdina                    

    # Convert a3 (integer: 1 or -1) to a float
    fcvt.s.w ft0, a3

    la t1, W_real
    la t2, W_imag
    
    vsetvli t0, a2, e32, m4    

    li a5, 1                        # a5    = n     = 1
    srai a4, a2, 1                  # a4    = a     = N / 2

    li t3, 0                        # t3    = j     = 0
    lw a3, logsize
    

    # t5 is vlen*4
    slli t5, t0, 2

    # k = (i * a) % (n * a); can be done as 
    # k = (i * a) & (N/2 - 1); 
    # Calculate (N/2 - 1) in s0
    addi s0, a4, -1                


    forTransform:                   #runs logN times
        bge t3, a3, forTransformEnd
        li t4, 0                        # t4 = i = 0
        
        # instead of recalculating (i+n)*4 and i*4
        # precalculate i and i+n, multiply by 4
        # keep adding vlen*4 to them in loop
        vid.v v28
        vsll.vi v20, v28, 2  # i*4

        # initializ i*a array then incrmement it in loop end by a*VLEN
        # t6 will be a*vlen
        vmul.vx v24, v28, a4
        mul t6, a4, t0
        # also shift n by 2 to calculate i*4 & n without doing one addition
        slli a6, a5, 2

        vinnerloop:                     # for i = 0; i < N
            bge t4, a2, vinnerloopend       # i  >= num elemenets
            
            # Calculate mask (i & n)
            vand.vx v0, v20, a6           
            vmseq.vx v0, v0, zero         # if (!(i & n)) which means this loop work only when result is 0
            # Start of If block. Every operation is masked wrt v0

            # Calculating k and offest
            # k = (i * a ) & (N/2 -1)
            vand.vx v28, v24, s0, v0.t      
            vsll.vi v28, v28, 2, v0.t     

            # Load from W_array[k]
            vloxei32.v v4, 0(t1), v28, v0.t
            vloxei32.v v28, 0(t2), v28, v0.t 

            # Now v28 contains W_imag for FFT, or -W_imag for IFFT
            vfsgnjx.vf v28, v28, ft0, v0.t

            # Calculate i+n offset *dynamically* using a temporary register (e.g., v16)
            vadd.vx v16, v20, a6, v0.t      # v16 = (i+n)*4 temporarily

            # Load from array[i+n]
            vloxei32.v v8, 0(a0), v16, v0.t
            vloxei32.v v12, 0(a1), v16, v0.t

            vfmul.vv v16, v4, v8, v0.t     # v16 = wreal*real[i+n]
            vfnmsac.vv v16, v28, v12, v0.t   # v16 = v16 - v28*v12 = wreal*real[i+n] - wimag*imag[i+n]

            vfmul.vv v12, v4, v12, v0.t     # wrealk*imag{i+n}
            vfmacc.vv v12, v28, v8, v0.t    # v12 = wrealk*imag[i+n] + wrealk*real{i+n}
            
            # Loading values from index i
            vloxei32.v v4, 0(a0), v20 , v0.t   
            vloxei32.v v28, 0(a1)  ,v20 , v0.t  

            vfadd.vv v8, v4, v16, v0.t
            vfsub.vv v4, v4, v16, v0.t
            vfadd.vv v16, v28, v12, v0.t
            vfsub.vv v28, v28, v12, v0.t

            # Saving values to index i
            vsoxei32.v v8 , 0(a0), v20, v0.t 
            vsoxei32.v v16, 0(a1), v20, v0.t 

            # Calculate i+n offset *again* for storing, reuse a temp (e.g. v8 since its value was just stored)
            vadd.vx v8, v20, a6, v0.t       # v8 = (i+n)*4 temporarily (overwrites B_real[i])

            # Saving values to index i+n
            vsoxei32.v v4 , 0(a0), v8, v0.t  
            vsoxei32.v v28 , 0(a1), v8, v0.t

            # incremenet v20(i) by vlen*4 (t5)
            vadd.vx v20, v20, t5

            # incrmement i*a by a*VLEN
            vadd.vx v24, v24, t6
        
            add t4, t4, t0                  # i += Vlen
            j vinnerloop
        vinnerloopend:

        slli a5, a5, 1                  # n = n * 2 
        srai a4, a4, 1                  # a = a/2 
        addi t3, t3, 1                  # j++
        j forTransform
    forTransformEnd:
   

    lw ra,0(sp)
    lw a0,4(sp)
    lw a1,8(sp)
    lw a2,12(sp)
    lw a3,16(sp)
    lw a4,20(sp)
    lw a5,24(sp)
    lw a6,28(sp)
    lw s0,32(sp)
    addi sp, sp, 36
    jr ra



# Function: vFFT
#   Performs the Fast Fourier Transform (FFT) on the input real and imaginary
#   arrays by calling the `vTransform` function. This function sets up the
#   inverse flag for FFT and calls the main transform logic.
#
# Inputs:
#   - a0: Base address of the real array
#   - a1: Base address of the imaginary array
#   - a2: Number of elements (N)
vFFT:                     
    addi sp, sp, -8
    sw ra, 0(sp)
    sw a3, 8(sp)

    li a3, 1                        # Inverse Flag a3 = 1 for FFT
    call vTransform
   
    lw ra, 0(sp)
    lw a2, 8(sp)
    addi sp, sp, 8
    
    ret



# Function: vIFFT
#   Performs the Inverse Fast Fourier Transform (IFFT) on the input real and imaginary arrays.
#   This function calls the vectorized `vTransform` for the main IFFT computation, then
#   scales the result by dividing each element of the output arrays by the total number of elements (N).
#
# Inputs:
#   - a0: Base address of the real array
#   - a1: Base address of the imaginary array
#   - a2: Number of elements (N)
vIFFT:              
    addi sp, sp, -16
    sw ra, 0(sp)            
    sw a0, 4(sp)    
    sw a1, 8(sp)
    sw a3, 12(sp)
    
    li a3, -1                       # Inverse Flag. a3 = -1 for IFFT
    call vTransform
    
    vsetvli t0, a2, e32, m8
    slli t2, t0, 2         # shift vlen by 2 for offest

    fcvt.s.w ft0, a2                # Convert N t0 float as we have to divide

    li t1, 0                        # i = 0. starting index
    vectorIFFTLoop:
    bge t1, a2, endVectorIFFTLoop

    # Load Real/Imag Pair 
    vle32.v v0, 0(a0)               # load t0 real values to vector v0
    vle32.v v8, 0(a1)               # load t0 imag values to vector v8

    # Divide by N
    vfdiv.vf v0, v0, ft0            # v0[i] = v0[i] / ft0 , ft0 is N in input
    vfdiv.vf v8, v8, ft0            # v8[i] = v8[i] / ft0 , ft0 is N in input

    # Save back to memory
    vse32.v v0, 0(a0)
    vse32.v v8, 0(a1)

    # Increment address by VLEN*4
    add a0, a0, t2                  
    add a1, a1, t2                  

    add t1, t1, t0                  # i += VLEN
    j vectorIFFTLoop
    endVectorIFFTLoop:

    lw ra, 0(sp)            
    lw a0, 4(sp)    
    lw a1, 8(sp)
    lw a3, 12(sp)
    addi sp, sp, 16

    ret



# TODO change la to mv   
# Function: print
# Logs values from real[] and imag[] arrays into registers ft0 and ft1 for debugging and output.
# Inputs:
#   - a0: Base address of real[] array
#   - a1: Base address of imag[] array
#   - a2: Size of array i.e. number of elements to log
# Clobbers: t0,t1, t2,t3 ft0, ft1.
print:        
    addi sp, sp, -12
    sw ra, 0(sp)
    sw a0, 4(sp)    
    sw a1, 8(sp)    

    li t0, 0x123456                 # Pattern for help in python script
    li t0, 0x234567                 # Pattern for help in python script
    li t0, 0x345678                 # Pattern for help in python script

	li t0, 0
    lw a2, size
    vsetvli t3, a2, e32, m8
    slli t4, t3, 2

    printloop:
    bge t0, a2, endPrintLoop

    # Load Real and Imag
    vle32.v v0, 0(a0)
    vle32.v v8, 0(a1)

    # Increment Pointers
    add a0, a0, t4
    add a1, a1, t4

    # Increment index
    add t0, t0, t3                 
    j printloop
    endPrintLoop:

    li t0, 0x123456                 # Pattern for help in python script
    li t0, 0x234567                 # Pattern for help in python script
    li t0, 0x345678                 # Pattern for help in python script
	
    lw ra, 0(sp)
    lw a0, 4(sp)    
    lw a1, 8(sp)   
    addi sp, sp, 12

	ret



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

    NEG_TWO_PI: .float -6.28318530717958647692
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
