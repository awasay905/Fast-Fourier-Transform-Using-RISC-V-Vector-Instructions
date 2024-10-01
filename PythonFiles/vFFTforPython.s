#define STDOUT 0xd0580000
.set SINCOSITER, 50
.set vectorSize, 1024              # Change this for max num of 32 bit vector element supported by hafdware

.section .text
.global _start
_start:

main:                       # Main Function. Calls FFT and IFFT
    call initHelperVector           # Initialize HelperVector for later uses
    la a0, real                     # a0 holds address of array real[]
    la a1, imag                     # a1 holds address of array imag[]
    la a2, size                     # use a2 to load size of arrays real/img
    lw a2, 0(a2)                    # a2 holds the size of arrays real/imag
    la a3, step                     # use a3 to load step for FFT/IFFT
    flw fa3, 0(a3)                  # 3 holds the step for FFT/IFFT
    
    call XXXX                       # Apply XXXX on the arrays
    
	call print                      # Writes down value of arrays to regsiter for helping in log
    j _finish                       # End program
    
initHelperVector:                   # Makes helperVEcotr = {0,1,2,3 ...VLEN-1}\
    addi sp, sp, -12
    sw t0, 0(sp)
    sw t1, 4(sp)
    sw t2, 8(sp)

    la t0, helperVector             # Load base address
    li t1, vectorSize               # Load size of vector
    li t2, 0                        # t2 = i = 0

    helperVectorLoop:
    bge t2, t1, endHelperVectorLoop # for i <= vectorSize
    sw t2, 0(t0)                    # Save i to helperVector
    addi t0, t0, 4                  # Add 4 offset to address
    addi t2, t2, 1                  # i = i + 1
    j helperVectorLoop

    endHelperVectorLoop:

    lw t0, 0(sp)
    lw t1, 4(sp)
    lw t2, 8(sp)
    addi sp, sp, 12

    jr ra
    
logInt:                     # Takes input N(a0), returns its log-base 2 in a0
    addi sp, sp, -4                 # Make space to save registers used
    sw t0, 0(sp)                    # Save t0 as we will be using it
    
    add t0, a0, zero                # t0 = k = N    , loop counter
    add a0, zero, zero              # a0 = i = 0

    logLoop:                        # while(k) i.e when k is non-zero
    beq t0, zero, logLoopEnd        # If k becomes 0, end loop
    srai t0, t0, 1                  # Shift right k by 1. K = k >> 1
    addi a0, a0, 1                  # i++. Increment i
    j logLoop
    logLoopEnd:

    addi a0, a0, -1                 # Return i - 1
    lw t0, 0(sp)                    # Restore t0
    addi sp, sp, 4                  # Restore stack pointer
    jr ra                           # Return to caller

vReverse:                   # Takes vector n in v29, array size N in a0 and vector size in a1. Returns bit reverse of vector in v29
    addi sp, sp, -20                # Make space to save registers used
    sw t0, 0(sp)
    sw t1, 4(sp)
    sw t2, 8(sp)
    sw t3, 12(sp)
    sw t4, 16(sp)

    vsetvli t0, a1, e32, m1         # Set vector lenght to a1
    vmv.v.i v28, 0                  # V28   = <p>   = 0

    addi sp, sp, -8                 # Save values before function call
    sw a0, 0(sp)
    sw ra, 4(sp)
    
    call logInt                     # Getting log of a0.    a0  = logN
    mv t2, a0                       # t2    = logN 
    
    lw a0, 0(sp)
    lw ra, 4(sp)
    addi sp, sp, 8                  # Restore Stack and values

    li t1, 1                        # j = 1 
    vReverseFor:                    # For j <= logN
    bgt t1, t2, vReverseForEnd      # Break when j > logN

    sub t3, t2, t1                  # t3    = logN - j
    li t4, 1                        # t4    = 1
    sll t3, t4, t3                  # t3    = (1 << (LogN - j))

    vand.vx v27, v29, t3            # v27   = v29 & t3 = n & (1 << (LogN - j))
    vmsne.vx v0, v27, zero          # v0    = 1 if (n & (1 << (LogN - j))) else 0

    li t3, 1                        # t3    = 1
    sub t4, t1, t3                  # t4    = t1 - t3   = j - 1
    sll t4, t3, t4                  # t4    = t3 << t4  = 1 << (j - 1)

    vor.vx v28, v28, t4, v0.t       # v28   = v28 | 1 << ( j - 1) , if condition

    addi t1, t1, 1                  # j++
    j vReverseFor
    vReverseForEnd:

    vmv.v.v v29, v28                # Move to v29
    
    lw t0, 0(sp)
    lw t1, 4(sp)
    lw t2, 8(sp)
    lw t3, 12(sp)
    lw t4, 16(sp)
    addi sp, sp, 20                 # Restore Stack

    jr ra

vMySin:                     # Takes input v31 of floats, and vector length a0. Returns sin(v31) in v31
    addi sp, sp, -28                # Make space to save registers used
    sw a1, 0(sp)
    sw t0, 4(sp)
    sw t1, 8(sp)
    sw t2, 12(sp)
    sw t3, 16(sp)
    sw t4, 20(sp)
    fsw ft1, 24(sp)

    vsetvli a1, a0, e32, m1         # Set vector lenght to a0. No need to loop as we know it is always a single vector
    vmv.v.v v30, v31                # v30   = term  = x
    vmv.v.v v29, v31                # v29   = sum   = x
    li t0, -1                       # t0    = sign  = -1
    li t1, 3                        # t1    = i     = 3
    li t2, SINCOSITER                       # t2    = itertaions    = 2*TERMS + 1   (RUNNING FOR 21 ITERATiONS)
    li t4, -1                       # t4    = -1    for sign changing

    vSinFor:                        # for loop i <= 2*TERMS + 1
    bgt t1, t2, vSinForEnd          # Break when i > 2*TERMS + 1

    addi t3, t1, -1                 # t3    =  i - 1
    mul t3, t3, t1                  # t3    = (i-1)*t1  = (i - 1) * i
    fcvt.s.w ft1, t3                # ft1   = (i - 1) * i in float

    vfmul.vv v28, v31, v31          # Multiply x * x and save to v28
    vfmul.vv v28, v30, v28          # Term * (x * x) and save to v28
    vfdiv.vf v30, v28, ft1          # Term  = (Term * (x * x) ) / (i - 1)*i

    fcvt.s.w ft1, t0                # convert sign to float
    vfmacc.vf v29, ft1, v30         # v29 = v29 + v30*ft1  => sum = sum + term*sign

    mul t0, t0, t4                  # sign  = -sgin
    addi t1, t1, 2                  # i     = i + 2
    j vSinFor
    vSinForEnd:

    vmv.v.v v31, v29                # return sum in v31

    lw a1, 0(sp)
    lw t0, 4(sp)
    lw t1, 8(sp)
    lw t2, 12(sp)
    lw t3, 16(sp)
    lw t4, 20(sp)
    flw ft1, 24(sp)
    addi sp, sp, 28                 # Restore stack 
    jr ra                           # Return to caller

vMyCos:                     # Takes input v30 of floats, and vector length a0. Returns cos(v30) in v30
    addi sp, sp, -28                # Make space to save registers used
    sw a1, 0(sp)
    sw t0, 4(sp)
    sw t1, 8(sp)
    sw t2, 12(sp)
    sw t3, 16(sp)
    sw t4, 20(sp)
    fsw ft1, 24(sp)

    vsetvli a1, a0, e32, m1         # Set vector lenght to a0. No need to loop as we know it is always a single vector
    vmv.v.i v29, 1                  # v29   = term  = 1 in integer
    vfcvt.f.x.v v29, v29            # v29 converted to float
    vmv.v.i v28, 1                  # v28   = term  = 1 in integer
    vfcvt.f.x.v v28, v28            # v28 converted to float

    li t0, -1                       # t0    = sign  = -1
    li t1, 2                        # t1    = i     = 2
    li t2, SINCOSITER                       # t2    = itertaions    = 2*TERMS + 1   (RUNNING FOR 21 ITERATiONS)
    li t4, -1                       # t4    = -1 for sign changing

    vCosFor:                        # for loop i <= 2*TERMS + 1
    bgt t1, t2, vCosForEnd          # Break when i > 2*TERMS + 1

    addi t3, t1, -1                 # t3    =  i - 1
    mul t3, t3, t1                  # t3    = (i-1)*t1  = (i - 1) * i
    fcvt.s.w ft1, t3                # ft1   = (i - 1) * i in float

    vfmul.vv v27, v30, v30          # Multiply x * x and save to v27
    vfmul.vv v27, v29, v27          # Term * (x * x) and save to v27
    vfdiv.vf v29, v27, ft1          # Term  = (Term * (x * x) ) / (i - 1)*i

    fcvt.s.w ft1, t0                # convert sign to float
    vfmacc.vf v28, ft1, v29         # v28 = v28 + v29*ft1  => sum = sum + term*sign

    mul t0, t0, t4                  # sign  = -sgin
    addi t1, t1, 2                  # i     = i + 2
    j vCosFor
    vCosForEnd:

    vmv.v.v v30, v28                # return sum in v30

    lw a1, 0(sp)
    lw t0, 4(sp)
    lw t1, 8(sp)
    lw t2, 12(sp)
    lw t3, 16(sp)
    lw t4, 20(sp)
    flw ft1, 24(sp)
    addi sp, sp, 28                 # Restore stack 
    jr ra                           # Return to caller

vOrdina:                    # Takes real a0, imag in a1, and N in a2
    addi sp, sp, -36                # Make space to save registers used
    sw a0, 0(sp)
    sw a1, 4(sp)
    sw a2, 8(sp)
    sw a3, 12(sp)
    sw a4, 16(sp)
    sw t0, 20(sp)
    sw t1, 24(sp)
    sw t2, 28(sp)
    sw t3, 32(sp)

    la a3, real_temp                # a3    = real_temp[] pointer
    la a4, imag_temp                # a4    = imag_temp[] pointer 

    vsetvli t0, a2, e32, m1         # Request vector for a2 length
    
    la t1, helperVector             # Load addres of vector 0,1,2,3... 
    vle32.v v26, 0(t1)              # v26 = <i> = {0, 1, 2, ... VLEN-1} => {i, i+1, i+2, ... i + VLEN - 1}

    li t1, 0                        # t1 = i = 0
    vOrdinaLoop:                    # Loop which will run N/VLEN times, solving simultanously VLEN elements 
    bge t1, a2, endVOrdinaLoop      # break when t1 >= num of elements as all required elemetns have been operated on

    addi sp, sp, -16                # Call Reverse function which will calculate reverse(N, n) of helper vector 
    sw a0, 0(sp)
    sw ra, 4(sp)
    sw a1, 8(sp)
    sw a2, 12(sp)

    mv a0, a2                       # a0 = N, input for reverse function
    mv a1, t0                       # a1 = vector size = VLEN which should be 8
    vmv.v.v v29, v26                # v29 = v26 = <i> for vReverse input

    call vReverse                   # Now V29 have rev(N, <i>), Keep it there for later use 

    lw a0, 0(sp)
    lw ra, 4(sp)
    lw a1, 8(sp)
    lw a2, 12(sp)
    addi sp, sp, 16                 # Stack restored. Function call ended

    vsll.vi v29, v29, 2             # Multiply index by 4 for offset
    
    vloxei32.v v23, 0(a0), v29      # Load into v23 real[rev_index] 
    vloxei32.v v24, 0(a1), v29      # Load into v24 imag[rev_index]

    slli t1, t1, 2                  # Multiply i by 4 to get starting offset
    add t2, a3, t1                  # t2 = real_temp + 4*i, # these are staring addresses for storing
    add t3, a4, t1                  # t3 = imag_temp + 4*i, # we store sequentially
    srli t1, t1, 2                  # Divide i by 4 to restore it
    vse32.v v23, 0(t2)              # real_temp[i] = real[rev_index];
    vse32.v v24, 0(t3)              # imag_temp[i] = imag[rev_index];

    vadd.vx v26, v26, t0            # adds VLEN to helperVector, so all indexes increase by VLEN

    add t1, t1, t0                  # i = i + VLEN   
    j vOrdinaLoop
    endVOrdinaLoop:

    mv t0, a2                       # Request for N element vector
    vsetvli t0, t0, e32             # Set vector length, acutal lenght stored in t0

    li t1, 0                        # t1    = j     = 0
    vOrdinaLoop2:                   # loop from 0 to size of array N
    bge t1, a2, endvOrdinaLoop2     # break when j >= N

    slli t1, t1, 2                  # Multiply i by 4 to get starting offset
    add t2, a3, t1                  # t2 = real_temp + 4*i, # these are staring addresses for storing
    add t3, a4, t1                  # t3 = imag_temp + 4*i, # we do not use above approach beause we store sequentially
    
    vle32.v v23, 0(t2)              # v23 = real_temp[i]
    vle32.v v24, 0(t3)              # v24 = imag_temp[i]

    add t2, a0, t1                  # t2 = real + 4*i     # i is incremented by VLEN dont forget
    add t3, a1, t1                  # t3 = imag + 4*i 
    vse32.v v23, 0(t2)              # real[i] = realtemp[i], well its j but nvm
    vse32.v v24, 0(t3)

    srli t1, t1, 2                  # Divide i by 4 to restore it

    add t1, t1, t0                  # i = i + VLEN
    j vOrdinaLoop2              
    endvOrdinaLoop2:

    lw a0, 0(sp)
    lw a1, 4(sp)
    lw a2, 8(sp)
    lw a3, 12(sp)
    lw a4, 16(sp)
    lw t0, 20(sp)
    lw t1, 24(sp)
    lw t2, 28(sp)
    lw t3, 32(sp)
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

    la t4, PI                       # Load mem address of PI to t4
    flw ft1, 0(t4)                  # Load PI to ft1
    li t4, -2                       # Load -2, for euler formula
    mul t4, t4, a3                  # Multiply by inverse. If a3 is -1, then IFFT is done, else FFT
    fcvt.s.w ft2, t4                # Convert it to float , in ft2
    fcvt.s.w ft3, a2                # ft3 = N
    fmul.s ft1, ft1, ft2            # ft1 = ft1*ft2 = PI * (inverse ) * -2
    fdiv.s ft1, ft1, ft3            # ft1 = ft1 / ft3 = PI * inverse * -2 / N

    vsincosloop:                    # for loop i = 0; i < N / 2;
    bge t3, a4, endvsincosloop      # as soon as num element t0 >= N/2, break

    vadd.vx v21, v22, t3            # v21 = v22 + i => {i, i+1, i+2, ..., i + VLEN -1}. Rn its i integer
    vfcvt.f.x.v v21, v21            # Convert helperVector 0,1,2 to floats.                     i float
    vfmul.vf v21, v21, ft1          # v21[i] = (inverse * -2.0 * PI  / N )*  i . Now we need cos and sins of this

    addi sp, sp, -4                 # Save return address for funtion call
    sw ra, 0(sp)

    vmv.v.v v30, v21                # Load v21 to v30 to pass to myCos
    call vMyCos                     # v30 = cos(v21) 
    vmv.v.v v20, v30                # Save v30 to v20 for later use

    vmv.v.v v31, v21                # Load v21 to v31 to pass to mySin
    call vMySin                     # v31 = sin(v21)
    vmv.v.v v19, v31                # Save v31 to v19 for later use

    lw ra, 0(sp)
    addi sp, sp, 4

    # Now, we have vector having cos, sin. Now we save to W_real, W_imag
    slli t3, t3, 2                  # Multiply i by 4 to gget offset
    add t5, t1, t3                  # t5 = wreal + i
    add t6, t2, t3                  # t6 = w + i
    
    vse32.v v20, 0(t5)              # W_real[i] = myCos(value);
    vse32.v v19, 0(t6)              # W_imag[i] = mySin(value); hopefully this works

    srli t3, t3, 2                  # Restore i
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
  
    
    forTransform:                   #int j = 0; j < logint(N); j++
    bge s0, a3, forTransformEnd     # End outer loop
  
    li s1, 0                        # s1 = i = 0
    vsetvli t0, a2, e32             # set vector  

    vinnerloop:                     # for i = 0; i < N
    bge s1, a2, vinnerloopend       # i  >= num elemenets

    la s2, helperVector             # First make vector of i
    vle32.v v18, (s2)               # v18 = 0, 1, 2  VLEN-1
    vadd.vx v18, v18, s1            # v18 = i, i+1, i+2, ....., i + VLEN-1
    vand.vx v18, v18, a5            # v1 & n = (i & n), (i+1 & n), .... (i + VLEN -1   & n)
    vmseq.vx v18, v18, zero         # if (!(i & n)) which means this loop work only when result is 0,

	vmclr.m v0                      # clear mask register
	vmmv.m v0, v18                  # copy mask from v18 to v0

    # THIS IS THE IF BLOCK. EVERY OPERATION WILL BE MASKED wrt v0

    # Loading real[i] and image[i]
    slli s1, s1, 2                  # s1 = i * 4 = offset, becasue each float 4 byte		
    add s3, a0, s1                  # s3 = real + offset
    add s4, a1, s1                  # s4 = imag + offset

    vmv.v.i v16, 0                  # put all zeros in vector, might be redundant, remove and test later
    vle32.v v16, 0(s3)   , v0.t     # real[i]. v16 = temp_real
    vmv.v.i v17, 0                  # put all zeros in vector, might be redundant, remove and test later
    vle32.v v17, 0(s4)   , v0.t     # imag[i]. v17 = temp_imag
    
    srli s1, s1, 2                  # restore i

    la s2, helperVector
    vmv.v.i v15, 0

    vle32.v v15, 0(s2) , v0.t       # v15 = 0, 1, 2, ..., Vlen-1
    vadd.vx v15, v15, s1 , v0.t     # v15 = i, i+1, ....m i + VLEN -1
    vmul.vx v15, v15, a4 , v0.t     # v15 = v15*a = i*a
    mul s5, a5, a4                  # s5 = n*a
    vrem.vx v15, v15, s5, v0.t      # v15 = v15 % (n*a)

    ## Load W_real[k], but k in int index, so mul by 4 to become offsets
    vsll.vi v15, v15, 2, v0.t       # v15 = v15 * 4. Now i can load values at k
    vmv.v.i v13, 0                  # put all zeros in vector, might be redundant, remove and test later
    vloxei32.v v13, 0(t1), v15, v0.t # v13 = wreal[k]
    vmv.v.i v14, 0                  # put all zeros in vector, might be redundant, remove and test later
    vloxei32.v v14, 0(t2), v15, v0.t # v14 = wimag[k]

    # Loading real[i + n] and image[i + n]
    la s2, helperVector
    vmv.v.i v15, 0                  # put all zeros in vector, might be redundant, remove and test later
    vle32.v v15, 0(s2), v0.t        # v15 = 0, 1, 2, ..., Vlen-1
    vadd.vx v15, v15, s1, v0.t      # v15 = i, i+1, ....m i + VLEN -1
    vadd.vx v15, v15, a5, v0.t      # v15 = i+n, i+1+n, ...., i+VLEN-1+n

    vsll.vi v15, v15, 2, v0.t       # v15 = v15 * 4. Now i can load values at i + n i think
    vmv.v.i v11, 0                  # put all zeros in vector, might be redundant, remove and test later
    vmv.v.i v12, 0                  # put all zeros in vector, might be redundant, remove and test later
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
    la s2, helperVector
    vle32.v v15, 0(s2), v0.t        # v15 = 0, 1, 2, ..., Vlen-1
    vadd.vx v15, v15, s1, v0.t      # v15 = i, i+1, ....m i + VLEN -1
    vsll.vi v15, v15, 2, v0.t       # v15 = v15 * 4. Now i can save values at i , i think
    
    vsoxei32.v v9, 0(a0),v15, v0.t  #  save to real[i]
    vsoxei32.v v10, 0(a1),v15, v0.t # imag[i]

    vsra.vi v15, v15, 2, v0.t      # v15 = v15 / 4. 
    vadd.vx v15, v15, a5, v0.t     # v15 = i+n, i+1+n, ...., i+VLEN-1+n
    vsll.vi v15, v15, 2, v0.t      # v15 = v15 * 4. Now i can save values at i+n , i think
    
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


vFFT:                       # Takes real a0, imag in a1, and N a2. Also d in fa3
    addi sp, sp, -24
    sw a0, 0(sp)
    sw a1, 4(sp)
    sw a2, 8(sp)
    sw a3, 12(sp)
    sw ra, 16(sp)
    fsw fa3, 20(sp)

    li a3, 1                        # Inverse Flag a3 = 1 for FFT, -1 for IFFT
    call vTransform
   
    lw a0, 0(sp)
    lw a1, 4(sp)
    lw a2, 8(sp)
    lw a3, 12(sp)
    lw ra, 16(sp)
    flw fa3, 20(sp)
    addi sp, sp, 24
    
    mv t0, a2                       # Request for N element vector
    vsetvli t0, t0, e32, m1         # Set vector length, acutal lenght stored in t0

    li t1, 0                        # i = 0. starting index
    vectorFFTLoop:                  # for (int i = 0; i < N; i++)
    bge t1, a2, endVectorFFTLoop    # break when i >= N

    slli t1, t1, 2                  # Multiply t1 by 4 for offest
    add s0, a0, t1                  # real + i
    add s1, a1, t1                  # imag + i
    
    vle32.v v3, 0(s0)               # load t0 real values to vector v3
    vle32.v v4, 0(s1)               # load t0 imag values to vector v4

    vfmul.vf v3, v3, fa3            # v3[i] = v3[i] * fa3 , fa3 is d in input
    vfmul.vf v4, v4, fa3            # v4[i] = v4[i] * fa3 , fa3 is d in input

    vse32.v v3, 0(s0)               # save result back to meme
    vse32.v v4, 0(s1)               # same as above

    srli t1, t1, 2                  # Restore i
    add t1, t1, t0                  # i += VLEN
    j vectorFFTLoop
    endVectorFFTLoop:
    
    jr ra
    
vIFFT:                      # Takes real a0, imag in a1, and N a2. Also d in fa3
    addi sp, sp, -24
    sw a0, 0(sp)
    sw a1, 4(sp)
    sw a2, 8(sp)
    sw a3, 12(sp)
    sw ra, 16(sp)
    fsw fa3, 20(sp)
    
    li a3, -1                        # Inverse Flag. a3 = 1 for FFT, -1 for IFFT
    call vTransform
    
    lw a0, 0(sp)
    lw a1, 4(sp)
    lw a2, 8(sp)
    lw a3, 12(sp)
    lw ra, 16(sp)
    flw fa3, 20(sp)
    addi sp, sp, 24
    
    mv t0, a2                       # Request for N element vector
    vsetvli t0, t0, e32, m1         # Set vector length, acutal lenght stored in t0
    fcvt.s.w fa3, a2                # Convert N t0 float as we have to divide

    li t1, 0                        # i = 0. starting index
    vectorIFFTLoop:                 # for (int i = 0; i < N; i++)
    bge t1, a2, endVectorIFFTLoop   # break when i >= N

    slli t1, t1, 2                  # i = i*4 for offser
    add s0, a0, t1                  # real + i
    add s1, a1, t1                  # imag + i

    vle32.v v3, 0(s0)               # load t0 real values to vector v3
    vle32.v v4, 0(s1)               # load t0 imag values to vector v4

    vfdiv.vf v3, v3, fa3            # v3[i] = v3[i] / fa3 , fa3 is N in input
    vfdiv.vf v4, v4, fa3            # v4[i] = v4[i] / fa3 , fa3 is N in input

    vse32.v v3, 0(s0)               # save result back to meme
    vse32.v v4, 0(s1)               # same as above

    srli t1, t1, 2                  # Restore i
    add t1, t1, t0                 # i += VLEN
    j vectorIFFTLoop
    endVectorIFFTLoop:

    jr ra

    
print:                      # Writes real/imag in the ft0 and ft1 register for log
	addi sp, sp, -24
	sw t0, 0(sp)
	sw s0, 4(sp)
	sw a0, 8(sp)
	sw a1, 12(sp)
	fsw fa0, 16(sp)
	fsw fa1, 20(sp)
	
	la a0, size                     # a0 has base address of word size
	lw s0, 0(a0)                    # load size to register s0
	
	la a0, real                     # now a0 has address of reals
	la a1, imag	                    # now a1 has address of imag
	
    li t0, 0x123456                 # Pattern for help in python script
    li t0, 0x234567                 # Pattern for help in python script
    li t0, 0x345678                 # Pattern for help in python script
	li t0, 0		                # load i = 0
	
	printloop:
	bge t0, s0, endPrintLoop
	
	flw fa0, 0(a0)
	flw fa1, 0(a1)
	
	addi a0, a0, 4
	addi a1, a1, 4
	
	addi t0, t0, 1
	j printloop
	endPrintLoop:

    li t0, 0x123456                 # Pattern for help in python script
    li t0, 0x234567                 # Pattern for help in python script
    li t0, 0x345678                 # Pattern for help in python script
	
	lw t0, 0(sp)
	lw s0, 4(sp)
	lw a0, 8(sp)
	lw a1, 12(sp)
	flw fa0, 16(sp)
	flw fa1, 20(sp)
	addi sp, sp, 24
	
	jr ra

_finish:
    li x3, 0xd0580000
    addi x5, x0, 0xff
    sb x5, 0(x3)
    beq x0, x0, _finish
.rept 100
    nop
.endr

# PUT INPUT HERE, DO NO CHANGE THIS LINE
# DO NOT CHANGE ANYTHING AFTER THIS LINE
.data  



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

    helperVector:
        .rept vectorSize
        .word 0
        .endr
