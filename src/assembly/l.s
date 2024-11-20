#define STDOUT 0xd0580000
.set vectorSize, 1024              # Change this for max num of 32 bit vector element supported by hafdware

.section .text
.global _start
_start:

# Initialize helper vector and load data addresses
main:
                              
    lw a0, size                     # Load size of real/imag arrays into a0
    call setlogN                    # Compute and store log2(size) for shared use by other functions


    call initHelperVector
    la a0, real                     # a0 = address of real[]
    la a1, imag                     # a1 = address of imag[]
    lw a2, size                     # a2 = size of arrays (N)

    # Perform FFT and IFFT
    call vFFT
    call vIFFT
    
    # Print results and finish
    call print
    j _finish 

# Initialize helper vector with sequential integers (0,1,2,3..)
initHelperVector:
    la t0, helperVector
    lw t1, size
    vsetvli t2, t1, e32, m1         # Set vector length once
    vid.v v0                        # Generate index vector
    vse32.v v0, (t0)
    ret



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
#   - v29: Input number to reverse.
#   - a0: Number of significant bits to reverse (optional; default 32).
# Outputs:
#   - v29: The reversed binary number.
# Clobbers:
#   - t0, v1, v2
vReverseIndexOffset:
    # Swap odd and even bits
    li t0, 0x55555555    # Pattern for odd/even bits
    vsrl.vi v1, v29, 1   # v29 >> 1
    vand.vx v1, v1, t0   # (v29 >> 1) & 0x55555555
    vand.vx v2, v29, t0  # v29 & 0x55555555
    vsll.vi v2, v2, 1    # (v29 & 0x55555555) << 1
    vor.vv v29, v1, v2   # Result back to v29

    # Swap consecutive pairs
    li t0, 0x33333333    # Pattern for pairs
    vsrl.vi v1, v29, 2       # v29 >> 2
    vand.vx v1, v1, t0       # (v29 >> 2) & 0x33333333
    vand.vx v2, v29, t0       # v29 & 0x33333333
    vsll.vi v2, v2, 2       # (v29 & 0x33333333) << 2
    vor.vv v29, v1, v2        # Result back to v29

    # Swap nibbles
    li t0, 0x0F0F0F0F    # Pattern for nibbles
    vsrl.vi v1, v29, 4       # v29 >> 4
    vand.vx v1, v1, t0       # (v29 >> 4) & 0x0F0F0F0F
    vand.vx v2, v29, t0       # v29 & 0x0F0F0F0F
    vsll.vi v2, v2, 4       # (v29 & 0x0F0F0F0F) << 4
    vor.vv v29, v1, v2        # Result back to v29

    # Swap bytes
    li t0, 0x00FF00FF    # Pattern for bytes
    vsrl.vi v1, v29, 8       # v29 >> 8
    vand.vx v1, v1, t0       # (v29 >> 8) & 0x00FF00FF
    vand.vx v2, v29, t0       # v29 & 0x00FF00FF
    vsll.vi v2, v2, 8       # (v29 & 0x00FF00FF) << 8
    vor.vv v29, v1, v2        # Result back to v29

    # Swap 2-byte pairs
    vsrl.vi v1, v29, 16      # v29 >> 16
    vsll.vi v2, v29, 16      # v29 << 16
    vor.vv v29, v1, v2        # Final result in v29

    # Save number of bits to reverse in t2
    # bits are in a7
    li t0, 30       # make is 30 instead of 32 to add shift by 2 effect
    sub t0, t0, a0  # a0 will never be more than 30
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

    ret



# Function: sin_cos_approx
# Calculates sin and cos of the float using chebishev polynomial
# Taken from stackoverflow
# Input:
#   - v30 = angle (a) in radians
# Output:
#   - v30 = sin() (approximation)
#   - v31 = cos() (approximation)
# Clobbers:
#   - t0, t1, ft0, ft1, ft2, ft3
# Help:
#   - i = v2 , ic = v3, j = v1, a = v30, sa = v4, c = v5 t = ft3, temp = v4
sin_cos_approx:
    # j = fmaf(a, 6.36619747e-1f, 12582912.f) - 12582912.f;
    vfmv.vf   v1, fs3           # Loads fs3 to v1. we need it for fma for accuracy
    vfmacc.vf  v1, fs2, v30
    vfsub.vf  v1, v1, fs3                

    vfnmsac.vf   v30,  fs0, v1       # a = a - j * half_pi_hi
    vfnmsac.vf   v30,  fs1, v1       # a = a - j * half_pi_lo

    vfcvt.rtz.x.f.v v2, v1           #  i = (int) j
    vadd.vi v3, v2, 1               # ic = i + 1

    vfmul.vv  v4, v30, v30          # ft2 = a * a (sa)

    # Approximate cosine.
    vfmv.vf   v4, fs4
    fmadd.s   v30, fs4, ft2, fs5     # c = c * sa + -1.38877297e-3
    fmadd.s   v30, v30, ft2, fs6     # c = c * sa + 4.16666567e-2
    fmadd.s   v30, v30, ft2, fs7     # c = c * sa + -0.5
    fmadd.s   v30, v30, ft2, fs8     # c = c * sa + 1.0

    # Approximate sine. By default save it to v31
    fmadd.s   v31, fs9, ft2, fs10     # s = s * sa + -1.98559923e-4
    fmadd.s   v31, v31, ft2, fs11     # s = s * sa + 8.33338592e-3
    fmadd.s   v31, v31, ft2, ft11     # s = s * sa + -0.166666672
    fmul.s    ft3, ft1, ft2           # t = a * sa
    fmadd.s   v31, v31, ft3, ft1      # s = s * a

    # Check the value of i and adjust the order of sine and cosine if needed
    andi t2, t0, 1                     # t2 = i & 1
    beqz t2, ifsincos                   # If i & 1 == 0, jump to ifsincos
    j adjust_sign                       # Jump to adjust_sign

    ifsincos:
        fmv.s ft0, v30                  # Swap sine and cosine
        fmv.s v30, v31
        fmv.s v31, ft0

    adjust_sign:
        andi t0, t0, 2                  # t0 = i & 2
        beqz t0, sign1done               # If i & 2 == 0, skip sign flip
        fneg.s v30, v30                  # Negate sine if i & 2 != 0

    sign1done:
        andi t1, t1, 2                  # t1 = ic & 2
        beqz t1, sign2done              # If ic & 2 == 0, skip sign flip
        fneg.s v31, v31                  # Negate cosine if ic & 2 != 0

    sign2done:
        ret                              # Return with sine in v30, cosine in v31


# Input:  v31 = angles in radians
# Output: v31 = sine values
vMySin:       # Uses t0, t1, ft0, ft1, ft2
    # all vectors except v20, v21 and v22 are free to use
    # Range Reduction
    la t0, NEG_HALF_PI              # t0 = *NEG_HALF_PI
    la t1, NEG_PI                   # t1 = *NEG_PI
    flw ft1, 0(t0)                  # ft1 = NEG_HALF_PI
    flw ft2, 0(t1)                  # ft2 = NEG_PI

    vmflt.vf v0, v31, ft1          # compares if x<NEG_HALF_PI and set true bits in mask
    ## NOW WE DO THE IF CONDITION
    vfrsub.vf v31, v31, ft2, v0.t  # v31 = -PI - x . for the if = condition x < neg_half_pi

    ## NOW CLEAR MARK AND CHECK FOR OTHER CONDITION
    vmnot.m v0, v0
    ## NEGATE NEG_HALF_PI and NEG_PI
    fneg.s  ft1, ft1            # ft1 = HALF_PI
    fneg.s ft2, ft2             # ft2 = PI
    vmfgt.vf v0, v31, ft1, v0.t          # compares if x>HALF_PI and set true bits in mask
    
    ## Now we do if condition
    vfrsub.vf v31, v31, ft2, v0.t  # v31 = PI - x for the if condition


    vfmul.vv v2, v31, v31           # v2 = x2 = x*x
    vfneg.v v2, v2                  # v2 = -x2
    vmv.v.v v3, v31                # v3   = term  = x
    # sum = v31
    li t0, 1                    # t0    = i  = 1
    fcvt.s.w ft1, t0 # ft1 = 1      # ft1 = factorial = 1

    lw t1, TERMS              # t1 = TERMS
    vSinFor:                        # for loop i <= 2*TERMS + 1
    bgt t0, t1, vSinForEnd          # Break when i > 2*TERMS + 1

    # only t0 and t1 used till here. v2, v3, v4,  v31 used
    slli t4, t0, 1                  # i*2 = t4
    addi t5, t4, 1                  # i*2 + 1 = t5
    mul t5, t5, t4                  # (i*2) * (i*2 + 1)  
    fcvt.s.w ft2, t5                # converts above to float
    fmul.s ft1, ft1, ft2            # factorial = facotial*

    # now t4, t5, ft2 free    
    vfmul.vv v3, v3, v2             # term = teerm * -x2
    # v5 free
    vfdiv.vf v5, v3, ft1
    vfadd.vv v31, v31, v5

    addi t0, t0, 1                  # i     = i + 1
    j vSinFor
    vSinForEnd:
    vmv.v.v v19, v31

    jr ra                           # Return to caller

# Input:  v30 = angles in radians
# Output: v30 = cosine values
vMyCos:                     # Takes input v30 of floats, and vector length a0. Returns cos(v30) in v30
    vmv.v.i v1, 1           # v1 = sign = 1
    # all vectors except v20, v21 and v22 are free to use
    # Range Reduction
    la t0, NEG_HALF_PI              # t0 = *NEG_HALF_PI
    la t1, NEG_PI
    flw ft1, 0(t0)                  # ft1 = NEG_HALF_PI
    flw ft2, 0(t1)

    vmflt.vf v0, v30, ft1          # compares if x<NEG_HALF_PI and set true bits in mask
    ## NOW WE DO THE IF CONDITION
    vfrsub.vf v30, v30, ft2, v0.t  # v30 = -PI - x . for the if = condition x < neg_half_pi
    vrsub.vx v1, v1, zero, v0.t # v1[i] = 0 - v1[i]  // negate for sign

    ## NOW CLEAR MARK AND CHECK FOR OTHER CONDITION
    fneg.s ft1, ft1     # ft1 = HALF_PI
    fneg.s ft2, ft2     # ft2 = PI
    vmnot.m v0, v0
    vmfgt.vf v0, v30, ft1, v0.t          # compares if x>HALF_PI and set true bits in mask
    
    ## Now we do if condition
    vfrsub.vf v30, v30, ft2, v0.t  # v30 = PI - x for the if condition
    vrsub.vx v1, v1, zero, v0.t # v1[i] = 0 - v1[i]  // negate for sign


    vfmul.vv v2, v30, v30           # v2 = x2 = x*x
    vfsgnjn.vv v2,v2,v2             # negates x2. v2 = -x2
    li t0, 1
    fcvt.s.w ft1, t0 # t0 = 1       # ft1 = factorial = 1
    vfmv.v.f v3, ft1                # v3   = term  = 1
    vfmv.v.f v30, ft1                # v30   = sum   = 1

    lw t1, TERMS                  # t1 = TERMS
    vCosFor:                        # for loop i <= 2*TERMS + 1
    bgt t0, t1, vCosForEnd          # Break when i > 2*TERMS + 1

    # only t0 and t1 used till here. v2, v3, v4,  v31 used
    slli t4, t0, 1                  # i*2 = t4
    addi t5, t4, -1                  # i*2 - 1 = t5
    mul t5, t5, t4                  # (i*2) * (i*2 - 1)  
    fcvt.s.w ft2, t5                # converts above to float
    fmul.s ft1, ft1, ft2            # factorial = facotial*

    # now t4, t5, ft2 free    
    vfmul.vv v3, v3, v2
    # v5 free
    vfdiv.vf v5, v3, ft1
    vfadd.vv v30, v30, v5

    addi t0, t0, 1                  # i     = i + 2
    j vCosFor
    vCosForEnd:
    
    vfcvt.f.x.v v1, v1  # converts sign to float
    vfmul.vv v20, v30, v1

    jr ra                           # Return to caller



vOrdina:                    # Takes real a0, imag in a1, and N in a2. uses all temp registers maybe. i havent checked
    addi sp, sp, -36                # Make space to save registers used
    sw a0, 0(sp)
    sw ra, 4(sp)
    sw a3, 12(sp)
    sw a4, 16(sp)


    mv t5, a0                       # t4 = real address
    mv t6, a1                       # t5 = imag
    la a3, real_temp                # a3    = real_temp[] pointer
    la a4, imag_temp                # a4    = imag_temp[] pointer 
    lw a0, logsize
    vsetvli a6, a2, e32, m1         # Request vector for a2 length
    
    la t1, helperVector             # Load addres of vector 0,1,2,3... 
    vle32.v v26, 0(t1)              # v26 = <i> = {0, 1, 2, ... VLEN-1} => {i, i+1, i+2, ... i + VLEN - 1}

    li t3, 0                        # t1 = i = 0
    vOrdinaLoop:                    # Loop which will run N/VLEN times, solving simultanously VLEN elements 
    bge t3, a2, endVOrdinaLoop      # break when t3 >= num of elements as all required elemetns have been operated on

    # Call vReverseIndexOffset. Uses t0, v1, v2
    vmv.v.v v29, v26                # v29 = v26 = <i> for vReverseIndexOffset input
    call vReverseIndexOffset                   # Now V29 have rev(N, <i>), Keep it there for later use 

    # Generate Index Offset
    vsll.vi v27, v26, 2             

    # Load from normal array reversed indexed
    vloxei32.v v23, 0(t5), v29      # Load into v23 real[rev_index] 
    vloxei32.v v24, 0(t6), v29      # Load into v24 imag[rev_index]

    # Save to temp array normal index
    vsoxei32.v v23, 0(a3) , v27             # real_temp[i] = real[rev_index];
    vsoxei32.v v24, 0(a4)  , v27            # imag_temp[i] = imag[rev_index];

    # Increment
    vadd.vx v26, v26, a6            # adds VLEN to helperVector, so all indexes increase by VLEN
    add t3, t3, a6                  # i = i + VLEN   
    j vOrdinaLoop
    endVOrdinaLoop:

    la t1, helperVector             # Load addres of vector 0,1,2,3... 
    vle32.v v26, 0(t1)              # v26 = <j> = {0, 1, 2, ... VLEN-1} => {i, i+1, i+2, ... i + VLEN - 1}

    li t1, 0                        # t1    = j     = 0
    vOrdinaLoop2:                   # loop from 0 to size of array N
    bge t1, a2, endvOrdinaLoop2     # break when j >= N

    vsll.vi v27, v26, 2                  # Multiply i by 4 to get starting offset
    vloxei32.v v23, 0(a3) ,v27             # v23 = real_temp[i]
    vloxei32.v v24, 0(a4), v27              # v24 = imag_temp[i]

    vsoxei32.v v23, 0(t5) , v27             # real[i] = realtemp[i], well its j but nvm
    vsoxei32.v v24, 0(t6), v27
    vadd.vx v26, v26, a6            # adds VLEN to helperVector, so all indexes increase by VLEN

    add t1, t1, a6                  # i = i + VLEN
    j vOrdinaLoop2              
    endvOrdinaLoop2:

    lw a0, 0(sp)
    lw ra, 4(sp)
    lw a3, 12(sp)
    lw a4, 16(sp)
    addi sp, sp, 36                # We use 9 registers in this one

    jr ra


vTransform:                 # Takes real a0, imag in a1, and N in a2, and Inverse Flag in a3

    addi sp, sp, -4         # Save return address for funtion call
    sw ra, 0(sp)

    call vOrdina                    # Call Vectorized Ordina.

    lw ra, 0(sp)                    # Restore return address
    addi sp, sp, 4                  # Stack restored

    la t1, W_real                   # t1    = W_real[]
    la t2, W_imag                   # t2    = W_imag[]

    # Loop for Sin/Cos (Euler Formula)
    la t0, helperVector             # Helper Vector is a vector of sequential number, hardcoded
    vle32.v v22, 0(t0)              # v22 = {0, 1, 2, 3, 4 .. VLEN -1}

    la t0, NEG_TWO_PI               # Load mem address of -2PI to t0
    flw ft1, 0(t0)                  # Load -2PI to ft1

    fcvt.s.w ft3, a2                # ft3 = N
    fdiv.s ft1, ft1, ft3            # ft1 = ft1 / ft3 = -2PI *  / N

    fcvt.s.w ft3, a3                # inverse
    fmul.s ft1, ft1, ft3            # Multiply by inverse. If a3 is -1, then IFFT is done, else FFT

    srai a4, a2, 1                  # a4    =   N / 2   = a / 2
    vsetvli t0, a4, e32             # Vector for N/2 elements
    li t3, 0                        # t3    = i = 0
    vsincosloop:                    # for loop i = 0; i < N / 2;
    bge t3, a4, endvsincosloop      # as soon as num element t0 >= N/2, break

    vadd.vx v23, v22, t3            # v23 = v22 + i => {i, i+1, i+2, ..., i + VLEN -1}. Rn its i integer
    vfcvt.f.x.v v21, v23            # Convert helperVector 0,1,2 to floats.                     i float
    vfmul.vf v21, v21, ft1          # v21[i] = (inverse * -2.0 * PI  / N )*  i . Now we need cos and sins of this

    addi sp, sp, -24                 # Save return address for funtion call + registers sued for loop
    fsw ft1, 0(sp)
    sw t3, 4(sp)
    sw ra, 8(sp)
    sw t1, 12(sp)
    sw t2, 16(sp)
    sw t0, 20(sp)

    vmv.v.v v30, v21                # Load v21 to v30 to pass to myCos
    call vMyCos                     # v30 = cos(v21) 

    vmv.v.v v31, v21                # Load v21 to v31 to pass to mySin
    call vMySin                     # v31 = sin(v21)

    flw ft1, 0(sp)
    lw t3, 4(sp)
    lw ra, 8(sp)
    lw t1, 12(sp)
    lw t2, 16(sp)
    lw t0, 20(sp)
    addi sp, sp, 24



    # Now, we have vector having cos, sin. Now we save to W_real, W_imag
    vsll.vi v23, v23, 2
    vsoxei32.v v20, 0(t1), v23              # W_real[i] = myCos(value);
    vsoxei32.v v19, 0(t2)  , v23            # W_imag[i] = mySin(value); hopefully this works

    add t3, t3, t0                  # i +=  VLEN
    j vsincosloop
    endvsincosloop:
    
    ##NOW STARTING NESTED LOOP

    li a5, 1                        # a5    = n     = 1
    srai a4, a2, 1                  # a4    = a     = N / 2
    li s0, 0                        # s0    = j     = 0

    lw a3, logsize                     # now a0 have logN

    vsetvli t0, a2, e32             # set vector  
    la s2, helperVector             # First make vector of i
    vle32.v v19, (s2)               # v18 = 0, 1, 2  VLEN-1
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
# Clobbers: t0,t1, t2, ft0, ft1.
print:                      
    li t0, 0x123456                 # Pattern for help in python script
    li t0, 0x234567                 # Pattern for help in python script
    li t0, 0x345678                 # Pattern for help in python script

    la t1, real                       # Move address to temp register to avoid stacking
    la t2, imag                       # Move address to temp register to avoid stacking
	li t0, 0		                # load i = 0

    printloop:
    bge t0, a2, endPrintLoop        # Exit loop if i >= size

    flw ft0, 0(t1)                  # Load real[i] into fa0
    flw ft1, 0(t2)                  # Load imag[i] into fa1

    addi t1, t1, 4                  # Increment pointer for real[]
    addi t2, t2, 4                  # Increment pointer for imag[]

    addi t0, t0, 1                  # Increment index
    j printloop                     # Jump to start of loop
    endPrintLoop:

    li t0, 0x123456                 # Pattern for help in python script
    li t0, 0x234567                 # Pattern for help in python script
    li t0, 0x345678                 # Pattern for help in python script
	
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

    .set dataSize, 512          # THIS IS N

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

    helperVector:
        .rept vectorSize
        .word 0
        .endr

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
