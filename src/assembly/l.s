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

    # Perform FFT and IFFT
    call vFFT
    call vIFFT
    
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
# Outputs:
#   - v29: The reversed binary number.
# Clobbers:
#   - t0, v1, v2
# Assumes that mask (0x55555555,0x33333333,0x0F0F0F0F, 0x00FF00FF  ) and shift(30) are saved in s1,s2,s3,s4, s5
vReverseIndexOffset:
    # Swap odd and even bits
    vsrl.vi v1, v26, 1   # v29 >> 1
    vand.vx v1, v1, s1   # (v29 >> 1) & 0x55555555
    vand.vx v2, v26, s1  # v29 & 0x55555555
    vsll.vi v2, v2, 1    # (v29 & 0x55555555) << 1
    vor.vv v29, v1, v2   # Result back to v29

    # Swap consecutive pairs
    vsrl.vi v1, v29, 2       # v29 >> 2
    vand.vx v1, v1, s2       # (v29 >> 2) & 0x33333333
    vand.vx v2, v29, s2       # v29 & 0x33333333
    vsll.vi v2, v2, 2       # (v29 & 0x33333333) << 2
    vor.vv v29, v1, v2        # Result back to v29

    # Swap nibbles
    vsrl.vi v1, v29, 4       # v29 >> 4
    vand.vx v1, v1, s3       # (v29 >> 4) & 0x0F0F0F0F
    vand.vx v2, v29, s3       # v29 & 0x0F0F0F0F
    vsll.vi v2, v2, 4       # (v29 & 0x0F0F0F0F) << 4
    vor.vv v29, v1, v2        # Result back to v29

    # Swap bytes
    vsrl.vi v1, v29, 8       # v29 >> 8
    vand.vx v1, v1, s4       # (v29 >> 8) & 0x00FF00FF
    vand.vx v2, v29, s4       # v29 & 0x00FF00FF
    vsll.vi v2, v2, 8       # (v29 & 0x00FF00FF) << 8
    vor.vv v29, v1, v2        # Result back to v29

    # Swap 2-byte pairs
    vsrl.vi v1, v29, 16      # v29 >> 16
    vsll.vi v2, v29, 16      # v29 << 16
    vor.vv v29, v1, v2        # Final result in v29

    # Save number of bits to reverse in t2
    # bits are in a7
    sub t0, s5, a7  # a7 will never be more than 30
    vsrl.vx v29, v29, t0
    
    ret                            # Return with result in v29



# Function: preload_constants
# Preloads floating-point constants into registers for use in trigonometric calculations.
# Inputs:
#   - None
# Outputs:
#   - Constants loaded into fs0 through fs11 and ft11.
# Clobbers:
#   - t0, fs0-fs11, ft11.
preload_constants:
    # Load addresses of constants into registers
    # Make use of the fact that all float are 4 bytes and stored consecutively
    la      t0, half_pi_hi          # Load address of half_pi_hi
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

    ret



# Function: sin_cos_approx
# Calculates sin and cos of the float using chebishev polynomial
# Taken from stackoverflow
# Input:
#   - a = v21 = angle (a) in radians
# Output:
#   - rs = v30 = sin() (approximation)
#   - rc = v31 = cos() (approximation)
# Clobbers:
#   - t0, t1, ft0, ft1, ft2, ft3
# Can not write to v22, v23, v1-v13
# Help:
#   c = v14
#   j = v15
#   a = v21
#   rs = v30
#   rc = v31
#   s = v16
#   sa = v17
#   t = v18
#   i = v19 
#   ic = v20, 

v_sin_cos_approx:
    # j = fmaf(a, 6.36619747e-1f, 12582912.f) - 12582912.f;
    vmv.v.v   v15, v4 # move 12582912 to v15  because 
    vfmacc.vv  v15, v21, v3 # a(v21))*6.36619747e-1f(v3) + 12582912.f(v15)
    vfsub.vf    v15, v15, fs3   # j = fmaf(a, 6.36619747e-1f, 12582912.f) - 12582912.f;

    vfnmsac.vv v21, v15, v1   #  a = fmaf (j, -half_pi_hi, a);
    vfnmsac.vv v21, v15, v2   #  a = fmaf (j, -half_pi_lo, a);

    vfcvt.x.f.v v19, v15          #  i = (int) j
    vadd.vi v20, v19, 1               # ic = i + 1

    vfmul.vv  v17, v21, v21          # ft2 = a * a (sa)

    # Approximate cosine.
    vmv.v.v   v14, v5  # c =               2.44677067e-5f; # why am i even doing this i can diretcly move flaot to vector splat
    vfmadd.vv v14, v17, v6     # c = c * sa + -1.38877297e-3
    vfmadd.vv v14, v17, v7     # c = c * sa + 4.16666567e-2
    vfmadd.vv v14, v17, v8     # c = c * sa + -0.5
    vfmadd.vv v14, v17, v9     # c = c * sa + 1.0

    # Approximate sine. By default save it to v31
    vmv.v.v     v16, v10          # v16 = 2.86567956e-6f
    vfmadd.vv   v16, v17, v11     # s = s * sa + -1.98559923e-4
    vfmadd.vv   v16, v17, v12     # s = s * sa + 8.33338592e-3
    vfmadd.vv   v16, v17, v13     # s = s * sa + -0.166666672
    vfmul.vv    v18, v21, v17          # t = a * sa
    vfmadd.vv   v16, v18, v21      # s = s * t + a

    # Check the value of i and adjust the order of sine and cosine if needed
    vand.vi v0, v19, 1                     # v0 = i & 1
    vmfne.vf v0, v0, ft11              # Set mask when true i.e not equal too zer0

    #Now we merge c(v14) and s(v16) to rc(v31) and rs(v30)
    vmerge.vvm v30, v16, v14, v0  #v0.mask[i] ? v14[i] : v16[i]
    vmerge.vvm v31, v14, v16, v0  #v0.mask[i] ? v14[i] : v16[i]
 
    vand.vi v0, v19, 2                   # t0 = i & 2
    vmfne.vf v0, v0, ft11              # Set mask when true i.e not equal too zer0
    vfsgnjn.vv v30,v30,v30, v0.t     # negate rs where i&2 is true

    vand.vi v0, v20, 2                   # t1 = ic & 2
    vmfne.vf v0, v0, ft11              # Set mask when true i.e not equal too zer0
    vfsgnjn.vv v31,v31,v31, v0.t                  # Negate cosine if ic & 2 != 0

    ret                              # Return with sine in v30, cosine in v31


vOrdina:                    # Takes real a0, imag in a1, and N in a2. uses all temp registers maybe. i havent checked
    addi sp, sp, -28                # Make space to save registers used
    sw a7, 0(sp)
    sw ra, 4(sp)
    sw s1, 8(sp)
    sw s2, 12(sp)
    sw s3, 16(sp)
    sw s4, 20(sp)
    sw s5, 24(sp)

    la t4, real_temp                # t4    = real_temp[] pointer
    la t5, imag_temp                # t5    = imag_temp[] pointer 
    lw a7, logsize
    
    vsetvli t3, a2, e32, m1         # Request vector for a2 length
    vid.v  v26      # v26 = <i> = {0, 1, 2, ... VLEN-1} => {i, i+1, i+2, ... i + VLEN - 1}

    # Load mask for reverse. This will reducded unnecesaary loadigs in loop
    li s1, 0x55555555
    li s2, 0x33333333
    li s3, 0x0F0F0F0F
    li s4, 0x00FF00FF  
    li s5, 30        # mask is 30 instead of 32 to add shift by 2 effect

    li t2, 0                        # t1 = i = 0
    vOrdinaLoop:                    # Loop which will run N/VLEN times, solving simultanously VLEN elements 
    bge t2, a2, endVOrdinaLoop      # break when t3 >= num of elements as all required elemetns have been operated on

    # reverse uses t0, v1, v2. Input/output in v26
    call vReverseIndexOffset                   # Now V29 have rev(N, <i>), Keep it there for later use 

    # Load from normal array reversed indexed
    vloxei32.v v23, 0(a0), v29       
    vloxei32.v v24, 0(a1), v29     

    # Generate Index Offset
    vsll.vi v27, v26, 2   

    # Save to temp array normal index
    vsoxei32.v v23, 0(t4), v27            
    vsoxei32.v v24, 0(t5), v27           

    # Increment
    vadd.vx v26, v26, t3            # adds VLEN to indexVector, so all indexes increase by VLEN
    add t2, t2, t3                  # i = i + VLEN   
    j vOrdinaLoop
    endVOrdinaLoop:

    vid.v v26
    vsll.vi v26, v26, 2             # Shift the indexes by 4 so it matches array offsets
    slli t6, t3, 2                  # Shift VLEN by 4. Now instead of using 2 insturctions to addlven then shit i will just add shifted vlen to shifter indexes

    li t1, 0                        # t1    = j     = 0
    vOrdinaLoop2:                   # loop from 0 to size of array N
    bge t1, a2, endvOrdinaLoop2     # break when j >= N

    # Indxed Load from temp array
    vloxei32.v v23, 0(t4), v26             
    vloxei32.v v24, 0(t5), v26            

    # Indxed Store to normal array
    vsoxei32.v v23, 0(a0), v26           
    vsoxei32.v v24, 0(a1), v26

    # Incrementing Indexes
    vadd.vx v26, v26, t6            # adds VLEN*4 to indexVector*4, so all indexes increase by VLEN*4
    add t1, t1, t3                  # i = i + VLEN

    j vOrdinaLoop2              
    endvOrdinaLoop2:

    lw a7, 0(sp)
    lw ra, 4(sp)
    lw s1, 8(sp)
    lw s2, 12(sp)
    lw s3, 16(sp)
    lw s4, 20(sp)
    lw s5, 24(sp)
    addi sp, sp, 28                # We use onlt 2 registers in this one

    jr ra


vTransform:                 # Takes real a0, imag in a1, and N in a2, and Inverse Flag in a3
    addi sp, sp, -4         # Save return address for funtion call
    sw ra, 0(sp)

    call vOrdina                    # Call Vectorized Ordina.


    la t1, W_real                   # t1    = W_real[]
    la t2, W_imag                   # t2    = W_imag[]

    # Loop for Sin/Cos (Euler Formula)
    vid.v v22

    la t0, NEG_TWO_PI               # Load mem address of -2PI to t0
    flw ft1, 0(t0)                  # Load -2PI to ft1
    mul  t0, a3, a2                 # t0 = (inverse)*N
    fcvt.s.w ft3, t0                # ft3 = N
    fdiv.s ft1, ft1, ft3            # ft1 = ft1 / ft3 = (inverse) -2PI *  / N


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
    vsoxei32.v v30, 0(t2), v23            # W_imag[i] = mySin(value); hopefully this works

    add t3, t3, t0                  # i +=  VLEN
    j vsincosloop
    endvsincosloop:
   
    ##NOW STARTING NESTED LOOP

    li a5, 1                        # a5    = n     = 1
    srai a4, a2, 1                  # a4    = a     = N / 2
    li s0, 0                        # s0    = j     = 0

    lw a3, logsize                     # now a0 have logN
    vid.v v19

    forTransform:                   #int j = 0; j < logint(N); j++
    bge s0, a3, forTransformEnd     # End outer loop
    li s1, 0                        # s1 = i = 0
    
    mul s5, a5, a4                  # s5 = n*a  // shfted it out of loop

    vinnerloop:                     # for i = 0; i < N
    bge s1, a2, vinnerloopend       # i  >= num elemenets
    
    vadd.vx v20, v19, s1            # v18 = i, i+1, i+2, ....., i + VLEN-1
    vand.vx v18, v20, a5            # v1 & n = (i & n), (i+1 & n), .... (i + VLEN -1   & n)
    vmseq.vx v0, v18, zero         # if (!(i & n)) which means this loop work only when result is 0,
    # THIS IS THE IF BLOCK. EVERY OPERATION WILL BE MASKED wrt v0

    # Loading real[i] and image[i]
    vsll.vi v21, v20, 2 , v0.t                 # s1 = i * 4 = offset, becasue each float 4 byte		
    vloxei32.v v16, 0(a0), v21 , v0.t     # real[i]. v16 = temp_real
    vloxei32.v v17, 0(a1)  ,v21 , v0.t     # imag[i]. v17 = temp_imag

    vmul.vx v15, v20, a4 , v0.t     # v15 = v15*a = i*a
    vrem.vx v15, v15, s5, v0.t      # v15 = v15 % (n*a)

    ## Load W_real[k], but k in int index, so mul by 4 to become offsets
    vsll.vi v15, v15, 2, v0.t       # v15 = v15 * 4. Now i can load values at k
    vloxei32.v v13, 0(t1), v15, v0.t # v13 = wreal[k]
    vloxei32.v v14, 0(t2), v15, v0.t # v14 = wimag[k]

    # Loading real[i + n] and image[i + n]
    vadd.vx v15, v20, a5, v0.t      # v15 = i+n, i+1+n, ...., i+VLEN-1+n

    vsll.vi v15, v15, 2, v0.t       # v15 = v15 * 4. Now i can load values at i + n i think
    vloxei32.v v11, 0(a0), v15, v0.t # real [i+n]
    vloxei32.v v12, 0(a1), v15, v0.t # imag[i+n] 

    vfmul.vv v7, v13, v11, v0.t     # v7 = wreal*real[i+n]
    vfnmsac.vv v7, v14, v12, v0.t   # v7 = v7 - v14*v12 = wreal*real[i+n] - wimag*imag[i+n]

    vfmul.vv v8, v13, v12, v0.t     # wrealk*imag{i+n}
    vfmacc.vv v8, v14, v11, v0.t    # v8 = wrealk*imag[i+n] + wrealk*real{i+n}
	
    vfadd.vv v9, v16, v7, v0.t
    vfadd.vv v10, v17, v8, v0.t
    vfsub.vv v5, v16, v7, v0.t
    vfsub.vv v6, v17, v8, v0.t

    # SAVE To realtempi, real/temp i+n
    vsoxei32.v v9, 0(a0),v21, v0.t  #  save to real[i]
    vsoxei32.v v10, 0(a1),v21, v0.t # imag[i]

    vsoxei32.v v5, 0(a0),v15, v0.t  # ad mask 
    vsoxei32.v v6, 0(a1),v15, v0.t  # ad mask 
   
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


vFFT:                       # Takes real a0, imag in a1, and N a2. Uses no t registers
    addi sp, sp, -4
    sw ra, 0(sp)

    li a3, 1                        # Inverse Flag a3 = 1 for FFT, -1 for IFFT
    call vTransform
   
    lw ra, 0(sp)
    addi sp, sp, 4
    
    jr ra
    

vIFFT:                      # Takes real a0, imag in a1, and N a2. USES t0-4 and ft0
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
.data  
    # PUT INPUT HERE, DO NO CHANHE ABOVE THIS

    real: .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8


    imag: .float 0,0,0,0, 1,1,1,1, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0
          .float 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0
          .float 0,0,0,0, 1,1,1,1, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0
          .float 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8

    .set dataSize, 1024          # THIS IS N

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
