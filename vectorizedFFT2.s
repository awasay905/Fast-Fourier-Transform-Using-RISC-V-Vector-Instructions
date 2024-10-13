#define STDOUT 0xd0580000
.set vectorSize, 1024              # Change this for max num of 32 bit vector element supported by hafdware

.section .text
.global _start
_start:

# Initialize helper vector and load data addresses
main:
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
    li t1, vectorSize
    vsetvli t2, t1, e32, m1         # Set vector length once
    vid.v v0                        # Generate index vector
    vse32.v v0, (t0)
    ret
    
# Calculate log base 2 of input
# Input:  a0 = N
# Output: a0 = log2(N)
logInt:
    mv t0, a0                       # t0 = N (loop counter)
    li a0, 0                        # a0 = result
    logLoop:
    beqz t0, logLoopEnd
    srli t0, t0, 1
    addi a0, a0, 1
    j logLoop
    logLoopEnd:
    addi a0, a0, -1                 # Adjust result
    ret
    
# Bit-reverse the elements of a vector
# Input:  v29 = input vector, a0 = N
# Output: v29 = bit-reversed vector
vReverse:
    # Save ra as we're calling another function
    addi sp, sp, -4
    sw ra, 0(sp)
    
    call logInt                     # a0 = log2(N)
    
    lw ra, 0(sp)
    addi sp, sp, 4

    li t1, 1                        # 1 for use in bit shift
    li t0, 1                        # t0 = jbit position counter
    vmv.v.x v28, zero               # v28 = p = result vector (initially 0)

    vReverseLoop:                   # For j <= logN
    bgt t0, a0, vReverseEnd         # Break when j > logN

    sub t2, a0, t0                  # t2    = logN - j
    sll t3, t1, t2                  # t3    = (1 << (LogN - j))
    
    vand.vx v0, v29, t3             # v0   = v29 & t2 = n & (1 << (LogN - j))
    vmsne.vx v0, v0, zero           # v0    = 1 if (n & (1 << (LogN - j))) else 0

    addi t4, t0, -1                 # t4    = t0 - 1   = j - 1
    sll t3, t1, t4                  # t3    = t5 << t4  = 1 << (j - 1)

    vor.vx v28, v28, t3, v0.t       # v28 = v28 | 1 << ( j - 1) Set bit in result if mask is true

    addi t0, t0, 1                  # Move to next bit
    j vReverseLoop
    vReverseEnd:

    vmv.v.v v29, v28                # Move to v29

    jr ra


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

    la t1, TERMS
    lw t1, 0(t1)                    # t1 = TERMS
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

    la t1, TERMS
    lw t1, 0(t1)                    # t1 = TERMS
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
    vfmul.vv v30, v30, v1

    jr ra                           # Return to caller



vOrdina:                    # Takes real a0, imag in a1, and N in a2. uses all temp registers maybe. i havent checked
    addi sp, sp, -36                # Make space to save registers used
    sw a0, 0(sp)
    sw a3, 12(sp)
    sw a4, 16(sp)

    mv t5, a0                       # t4 = real address
    mv t6, a1                       # t5 = imag
    la a3, real_temp                # a3    = real_temp[] pointer
    la a4, imag_temp                # a4    = imag_temp[] pointer 

    vsetvli t0, a2, e32, m1         # Request vector for a2 length
    
    la t1, helperVector             # Load addres of vector 0,1,2,3... 
    vle32.v v26, 0(t1)              # v26 = <i> = {0, 1, 2, ... VLEN-1} => {i, i+1, i+2, ... i + VLEN - 1}

    li t1, 0                        # t1 = i = 0
    vOrdinaLoop:                    # Loop which will run N/VLEN times, solving simultanously VLEN elements 
    bge t1, a2, endVOrdinaLoop      # break when t1 >= num of elements as all required elemetns have been operated on

    addi sp, sp, -8                # Call Reverse function which will calculate reverse(N, n) of helper vector 
    sw ra, 0(sp)
    sw t1, 4(sp)
    
    mv a0, a2                       # a0 = N, input for reverse function. a0 is overwritten but no worriesv
    vmv.v.v v29, v26                # v29 = v26 = <i> for vReverse input

    call vReverse                   # Now V29 have rev(N, <i>), Keep it there for later use 

    lw ra, 0(sp)
    lw t1, 4(sp)
    addi sp, sp, 8                # Stack restored. Function call ended


    vsll.vi v29, v29, 2             # Multiply index by 4 for offset
    vloxei32.v v23, 0(t5), v29      # Load into v23 real[rev_index] 
    vloxei32.v v24, 0(t6), v29      # Load into v24 imag[rev_index]

    vsll.vi v27, v26, 2             # mul i by 4 for alignment
    vsoxei32.v v23, 0(a3) , v27             # real_temp[i] = real[rev_index];
    vsoxei32.v v24, 0(a4)  , v27            # imag_temp[i] = imag[rev_index];

    vadd.vx v26, v26, t0            # adds VLEN to helperVector, so all indexes increase by VLEN

    add t1, t1, t0                  # i = i + VLEN   
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
    vadd.vx v26, v26, t0            # adds VLEN to helperVector, so all indexes increase by VLEN

    add t1, t1, t0                  # i = i + VLEN
    j vOrdinaLoop2              
    endvOrdinaLoop2:

    lw a0, 0(sp)
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
    srai a4, a2, 1                  # a4    =   N / 2   = a / 2
    vsetvli t0, a4, e32             # Vector for N/2 elements

    li t3, 0                        # t3    = i = 0

    la t4, helperVector             # Helper Vector is a vector of sequential number, hardcoded
    vle32.v v22, 0(t4)              # v22 = {0, 1, 2, 3, 4 .. VLEN -1}

    la t4, NEG_TWO_PI               # Load mem address of -2PI to t4
    flw ft1, 0(t4)                  # Load -2PI to ft1
    fcvt.s.w ft3, a2                # ft3 = N
    fdiv.s ft1, ft1, ft3            # ft1 = ft1 / ft3 = -2PI *  / N
    fcvt.s.w ft3, a3                # inverse
    fmul.s ft1, ft1, ft3               # Multiply by inverse. If a3 is -1, then IFFT is done, else FFT
    vsincosloop:                    # for loop i = 0; i < N / 2;
    bge t3, a4, endvsincosloop      # as soon as num element t0 >= N/2, break

    vadd.vx v23, v22, t3            # v21 = v22 + i => {i, i+1, i+2, ..., i + VLEN -1}. Rn its i integer
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
    vmv.v.v v20, v30                # Save v30 to v20 for later use

    vmv.v.v v31, v21                # Load v21 to v31 to pass to mySin
    call vMySin                     # v31 = sin(v21)
    vmv.v.v v19, v31                # Save v31 to v19 for later use

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

    addi sp, sp, -8                 ## CALL LOG INT TO GET LOG OF N
    sw a0, 0(sp)                    # save a0 because we will pass argument from it
    sw ra, 4(sp)
    
    mv a0, a2                       # logN input is N which was in a2
    call logInt                     # now a0 have logN
    mv a3, a0                       # a3    = logN 
    
    lw a0, 0(sp)                    # restore a0 to real address
    lw ra, 4(sp)
    addi sp, sp, 8                  # LOGINT CALL END HERE LOL 
  
    vsetvli t0, a2, e32             # set vector  
    la s2, helperVector             # First make vector of i
    vle32.v v19, (s2)               # v18 = 0, 1, 2  VLEN-1
    forTransform:                   #int j = 0; j < logint(N); j++
    bge s0, a3, forTransformEnd     # End outer loop
  
    li s1, 0                        # s1 = i = 0
    

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
    mul s5, a5, a4                  # s5 = n*a
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
    addi sp, sp, -8
    sw ra, 0(sp)
    sw a3, 4(sp)

    li a3, 1                        # Inverse Flag a3 = 1 for FFT, -1 for IFFT
    call vTransform
   
    lw ra, 0(sp)
    lw a3, 0(sp)
    addi sp, sp, 8
    
    jr ra
    

vIFFT:                      # Takes real a0, imag in a1, and N a2. USES t0-4 and ft0
    addi sp, sp, -8
    sw ra, 0(sp)
    sw a3, 4(sp)
    
    li a3, -1                        # Inverse Flag. a3 = 1 for FFT, -1 for IFFT
    call vTransform
    
    lw ra, 0(sp)
    lw  a3, 4(sp)
    addi sp, sp, 8
    
    vsetvli t0, a2, e32, m1         # Set vector length to a2, acutal lenght stored in t0
    fcvt.s.w ft0, a2                # Convert N t0 float as we have to divide

    li t1, 0                        # i = 0. starting index
    slli t2, t0, 2                  # shift vlen by 2 for offest
    mv t3, a0                       # moves real address to t3
    mv t4, a1                       # moves imag addrress to t4
    vectorIFFTLoop:                 # for (int i = 0; i < N; i++)
    bge t1, a2, endVectorIFFTLoop   # break when i >= N

    vle32.v v3, 0(t3)               # load t0 real values to vector v3
    vle32.v v4, 0(t4)               # load t0 imag values to vector v4

    vfdiv.vf v3, v3, ft0            # v3[i] = v3[i] / ft0 , ft0 is N in input
    vfdiv.vf v4, v4, ft0            # v4[i] = v4[i] / ft0 , ft0 is N in input

    vse32.v v3, 0(t3)               # save result back to meme
    vse32.v v4, 0(t4)               # same as above

    add t3, t3, t2                  # real + VLEN
    add t4, t4, t2                  # imag + VLEN

    add t1, t1, t0                 # i += VLEN
    j vectorIFFTLoop
    endVectorIFFTLoop:

    jr ra

    
print:                      # Writes real/imag in the ft0 and ft1 register for log. uses t0-3, ft0-1	
	la t1, size                     # t1 has base address of word size
	lw t3, 0(t1)                    # load size to register t3
	
	la t1, real                     # now t1 has address of reals
	la t2, imag	                    # now t2 has address of imag
	
    li t0, 0x123456                 # Pattern for help in python script
    li t0, 0x234567                 # Pattern for help in python script
    li t0, 0x345678                 # Pattern for help in python script
	li t0, 0		                # load i = 0
	
	printloop:
	bge t0, t3, endPrintLoop
	
	flw ft0, 0(t1)
	flw ft1, 0(t2)
	
	addi t1, t1, 4
	addi t2, t2, 4
	
	addi t0, t0, 1
	j printloop
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

    imag: .float 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0
          .float 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0
          
    .set dataSize, 128          # THIS IS N

    # DO NOT CHANGE ANYTHING AFTER THIS LINE

    .set halfDataSize, dataSize/2 
    size: .word dataSize
    step: .float 1.0

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
    TERMS: .word 10

    helperVector:
        .rept vectorSize
        .word 0
        .endr
