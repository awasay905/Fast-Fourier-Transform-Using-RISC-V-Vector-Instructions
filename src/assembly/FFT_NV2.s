#define STDOUT 0xd0580000

.section .text
.global _start
_start:

main:                               # Main Function to Call FFT/IFFT
    la a0, real                     # a0 points to real[]
    la a1, imag                     # a1 points to imag[]
    lw a2, size                    # a2 has size of real/imag arrays  
    
    call FFT                        # Apply FFT on the arrays
    call IFFT                       # Apply IFFT on the arrays
    
	call print                      # Writes arrays values to register for logging
    j _finish                       # End program


logint:                             # Returns log(N) base 2 where N=a0
    clz a0, a0            # Count leading zeros of rs1, store result in rd
    li t0, 31              # Load 31 (32-bit word size - 1) into temporary register t0
    sub a0, t0, a0         # Subtract clz result from 31 to get log2(n)
    
    jr ra
 
increment_reversed:
    # Step 1: Create mask where first `bits` are 0 and the rest are 1
    li      t2, -1               # Load -1 into t0
    sll     t0, t2, a7          # Shift -1 left by `bits` (a7) to create -1U << bits

    # Step 2: OR the number with the mask
    or      a6, a6, t0          # num = num | mask

    # Step 4: Count leading zeros in the negated number
    not     t0, a6              # t0 = ~num
    clz     t1, t0              # t1 = __builtin_clz(~num), count leading zeros

    # Step 5: Calculate the position of the first 0-bit from the left
    li      t0, 31              # Load 31 into t0 (number of bits in uint32_t)
    sub     t1, t0, t1          # t1 = 32 - leading_zeros -1. . t1 = first_zero_index - 1

    # Step 7: Set the bit at the first_zero_index - 1
    bset a6, a6, t1

    # Step 8: Clear all bits to the left of the first_zero_index
    addi    t1, t1, 1           # first_zero_index
    sll     t0, t2, t1          # t0 = -1U << (first_zero_index)
    andn     a6, a6, t0          # num &= ~((-1U << (first_zero_index)))

    # Return the modified num
    ret


reverse:                            # Reverse the binary digits of the number. Takes input N (in a0) and n (in a1).
    # Save return address
    addi    sp, sp, -4             # Allocate stack space
    sw      ra, 0(sp)              # Save return address

    call logint             # Call logint, result in a0

    lw      ra, 0(sp)              # Restore return address
    addi    sp, sp, 4              # Deallocate stack space
    
    # Move log result to t1 and restore n to a0
    mv      t1, a0                 # Save log result in t1
    mv      a0, a1                 # Move n to a0

    # Clear upper bits of n
    li      t0, 1                  
    sll     t0, t0, t1             # t0 = 1 << log(N)
    addi    t0, t0, -1             # Create mask of log(N) bits
    and     a0, a0, t0             # Clear upper bits of n

    # Save number of bits to reverse in t2
    mv      t2, t1

    # First swap pass: swap adjacent bits
    li      t0, 0x55555555          # Load mask for odd bits
    and     t1, a0, t0              # Extract odd bits
    slli    t1, t1, 1               # Shift odd bits left
    srli    a0, a0, 1               # Shift input right
    and     a0, a0, t0              # Extract even bits
    or      a0, a0, t1              # Combine bits

    # Second swap pass: swap bit pairs
    li      t0, 0x33333333          # Load mask for 2-bit groups
    and     t1, a0, t0              # Extract lower bits of pairs
    slli    t1, t1, 2               # Shift lower bits left
    srli    a0, a0, 2               # Shift input right
    and     a0, a0, t0              # Extract upper bits of pairs
    or      a0, a0, t1              # Combine bits

    # Third swap pass: swap nibbles
    li      t0, 0x0F0F0F0F          # Load mask for 4-bit groups
    and     t1, a0, t0              # Extract lower nibbles
    slli    t1, t1, 4               # Shift lower nibbles left
    srli    a0, a0, 4               # Shift input right
    and     a0, a0, t0              # Extract upper nibbles
    or      a0, a0, t1              # Combine nibbles

    # Final swap pass: reverse bytes
    slli    t1, a0, 24              # Shift leftmost byte to rightmost
    li      t0, 0xFF00
    and     t0, a0, t0              # Extract second byte
    slli    t0, t0, 8               # Shift it left by 8
    or      t1, t1, t0              # Combine with first result
    
    srli    t0, a0, 8               # Shift right by 8
    li      a0, 0xFF00
    and     t0, t0, a0              # Extract second-to-last byte
    or      t1, t1, t0              # Combine with previous result
    
    srli    t0, a0, 24              # Shift rightmost byte to leftmost
    or      a0, t1, t0              # Final combination

    # Shift right to align the reversed bits
    li      t0, 32                  # Total bits in word
    sub     t0, t0, t2              # Calculate shift amount
    srl     a0, a0, t0              # Shift right to align

    
    ret                            # Return with result in a0
    

mySin:                              # Returns sin(x) where x=fa0
    # Range Reduction to [0, pi/2] for accuracy

    # Compare if x(fa0) < NEG_HALF_PI(ft0)
    la t0, NEG_HALF_PI              # Load constants
    flw ft0, 0(t0)                  # ft0 = NEG_HALF_PI
    flt.s t0, fa0, ft0              # t0 = fa0 < ft0
    bnez t0 , lessThanhalfPI        # Result is 1 if true (not equal to zero)
    
    # Compare if x(fa0) > HALF_PI(ft0)
    la t0, HALF_PI                  # Load constants
    flw ft0, 0(t0)                  # ft0 = NEG_HALF_PI
    fle.s t0, fa0, ft0              #  checking x <= halfpi, reverse condition
    beqz t0, moreThanhalfPI         # if false, go to if blok
    j doneRangeReduction            # ff true, we will skip 

    lessThanhalfPI:                 # if block for x < NEG_HALF_PI
    la t0, NEG_PI
    flw ft1, 0(t0)                  # ft1 = NEG_PI
    fsub.s fa0, ft1, fa0            # fa0 = -PI -x
    j doneRangeReduction

    moreThanhalfPI:                 # if block for x > HALF_PI
    la t0, PI
    flw ft1, 0(t0)                  # ft1 = PI
    fsub.s fa0, ft1, fa0            # fa0 = PI - x
    j doneRangeReduction

    doneRangeReduction:
    fmul.s ft0, fa0, fa0            # ft0 = x2 = x*x
    fmv.s ft1, fa0                  # ft1 = term = x
    fmv.s ft2, fa0                  # ft2 = sum = x
    la t0, ONE
    flw ft3, 0(t0)                  # ft3 = factorail = 1.0
    
    la t0, TERMS
    lw t4, 0(t0)                    # t4 = TERMS for taylors
    li t1, 1                        # t1 = i = 1
    sinfor:
    bgt t1, t4, sinforend
    
    #START HERE
    #factorial =factorial * (2*i) * (2*i + 1);
    #Multipli i by 2
    li t0, 2 # t0 = 2
    mul t2, t0, t1 # t2 = 2*i
    li t0, 1 # t0 = 1
    add t3, t2, t0 # t3 = (2*i) + 1
    mul t2, t2,  t3 # t2 = (2*i) * (2*i + 1)
    # now convert to float
    fcvt.s.w ft4, t2    # now ft4 =  (2*i) * (2*i + 1). t2, t3 free to use
    fmul.s ft3, ft3, ft4 # facortila done, ft4  free

    # term =term * -x2; //negative x2
    ###fneg.s ft4, ft0 # ft4 = -x2
    fmul.s ft1,  ft1, ft0 # term = term * (-x2)
    fneg.s ft1, ft1

    # float next_term = term / factorial; ft4 is free to use dont forget
    fdiv.s ft4, ft1, ft3

    # sum =sum +  next_term;
    fadd.s ft2, ft2, ft4
   
    # LOop stuff
    addi t1, t1, 1 # i ++
    j sinfor
    sinforend:
    
    fmv.s fa0, ft2 # return sum(ft2)

    jr ra


myCos:          # Returns cos(x) where x=fa0
    # Range Reduction to [0, pi/2] for accuracy
    li t4, 1 # t4 = 1 = sign
    # Load constants
    la t0, NEG_HALF_PI
    flw ft0, 0(t0) # ft0 = neg_halfPI

    # Compare if x (fa0) < -halfpi (ft0)
    flt.s t0, fa0, ft0 # t0 = fa0 < ft0
    bnez t0 , lessThanneghalfPIcos 
    
    # Compare x (fa0) > halfpi (ft0)
    la t0, HALF_PI
    flw ft0, 0(t0) # ft0 = halfPI
    fle.s t0, fa0, ft0 # checks if x <= halfpi, reverse condition
    beqz t0, moreThanhalfPIcos # if false, go to if blok
    j doneRangeReductioncos # ff true, we will skip 


    lessThanneghalfPIcos:         # If block for X < -Half_PI
    la t0, NEG_PI
    flw ft1, 0(t0)  # ft1 = -PI
    fsub.s fa0, ft1, fa0 # ft1 = -PI -x
    li t4, -1 # sign = -1
    j doneRangeReductioncos

    moreThanhalfPIcos:         # If block for x > Half PI
    la t0, PI
    flw ft1, 0(t0)      # ft1 = PI
    fsub.s fa0, ft1, fa0    # x = PI - x
    li t4, -1 # sign = -1
    j doneRangeReductioncos

    doneRangeReductioncos:
    
    fmul.s ft0, fa0, fa0 # ft0 = x2 = x*x
    la t0, ONE
    flw ft1, 0(t0)  # ft1 = term = 1
    fmv.s ft2, ft1 # ft2 = sum = 1
    fmv.s ft3, ft1 # ft3 =  faotiraln = 1.0
    
    la t0, TERMS
    lw t5, 0(t0)   # t5 = TERMS for taylors
    li t1, 1 # t1 = i = 1
    cosfor:
    bgt t1, t5, cosforend
    
    #START HERE
    #factorial =factorial * (2*i) * (2*i - 1);
    #Multipli i by 2
    li t0, 2 # t0 = 2
    mul t2, t0, t1 # t2 = 2*i
    li t0, 1 # t0 = 1
    sub t3, t2, t0 # t3 = (2*i) - 1
    mul t2, t2,  t3 # t2 = (2*i) * (2*i - 1)
    # now convert to float
    fcvt.s.w ft4, t2    # now ft4 =  (2*i) * (2*i - 1). t2, t3 free to use
    fmul.s ft3, ft3, ft4 # facortila done, ft4  free

    # term =term * -x2; //negative x2
    ###fneg.s ft4, ft0 # ft4 = -x2
    fmul.s ft1,  ft1, ft0 # term = term * (-x2)
    fneg.s ft1, ft1

    # float next_term = term / factorial; ft4 is free to use dont forget
    fdiv.s ft4, ft1, ft3

    # sum =sum +  next_term;
    fadd.s ft2, ft2, ft4
   
    # LOop stuff
    addi t1, t1, 1 # i ++
    j cosfor
    cosforend:

    fcvt.s.w ft4, t4 # convert sign to float
    fmul.s ft2, ft2, ft4 # mul sum by sign
    fmv.s fa0, ft2 # return sum(ft2)

    jr ra


ordina: # it receives base address of real[] a0, imag[] a1, and an int N a2
    addi sp, sp, -12
    sw ra, 0(sp)
    sw a0, 4(sp)
    sw a1, 8(sp)
    
    la s0, real_temp
    la s1, imag_temp
    lw a7, logsize

    li t6,  0 # i
    li a6,  0 # rev(i)
    
    forordina:
    bge t6, a2, endforordina

    call reverse  # is saved in to a6. Do not save ra. it is saved in parent funciton


    slli t2, t6, 2  # i*4
    slli t3, a6, 2  # rev_index*4
    
    add t4, a0, t3  # real array index rev_index
    add t5, s0, t2  # real_temp array i
    flw ft1, 0(t4)
    fsw ft1, 0(t5)
    
    add t4, a1, t3  # imag array index rev_index
    add t5, s1, t2  # imag_temp array i
    flw ft1, 0(t4)
    fsw ft1, 0(t5)
    

    addi t6, t6, 1
    j forordina
    endforordina:

    
    addi t1, zero, 0  # j
    forordina2:
    bge t1, a2, endforordina2
    slli t2, t1, 2  # t2 = t1*4 for index
    add t3, t2, s0
    flw ft1, 0(t3)  # load real_temp
    add t3, t2, a0
    fsw ft1, 0(t3)  # save to real
    
    add t3, t2, s1
    flw ft1, 0(t3)  # load imag_temp
    add t3, t2, a1
    fsw ft1, 0(t3)  # save to real
    
    addi t1, t1, 1
    j forordina2
    endforordina2:
    
    lw ra, 0(sp)
    lw a0, 4(sp)
    lw a1, 8(sp)
    addi sp, sp, 12

    jr ra


transform:
 # it receives base address of real[] a0, imag[] a1, and an int N a2, inverse flag in a3
    addi sp, sp, -4
    sw ra, 0(sp)

    call ordina

    lw ra, 0(sp)
    addi sp, sp, 4
    
    la s0, W_real
    la s1, W_imag
    
    # call to cos and sin in loop
    addi a6, zero, 0            # a6 = i = 0
    srli s3, a2, 1              # s3 = N/2
    
    # Calculating (inverse)*-2*PI/N
    la a7, NEG_TWO_PI
    flw fa3, 0(a7)              # fa3 = -2*PI
    fcvt.s.w fa7, a3            # put inverse (-1) in fa7
    fmul.s fa3, fa3, fa7        # fa3 = (inverse)*-2*PI
    fcvt.s.w fa7, a2            # fa7 = N
    fdiv.s fa3, fa3, fa7 # fa3 = (inverse)*-2*PI/N

    sincosfor:
    bge a6, s3, sincosforend
    
    fcvt.s.w fa6, a6 # fa6 = i
    fmul.s fa0, fa3, fa6    # fa0 is mulvalue
    
    # now call myCos but save fa0 first
    addi sp, sp, -8
    sw ra, 0(sp)
    fsw fa0, 4(sp)
    
    call myCos          # Now fa0 has myCos
    fmv.s fs4, fa0 # fs4 = mycos
    
    lw ra, 0(sp)
    flw fa0, 4(sp)
    addi sp, sp, 8
    
    # now call mySin but no need to save fa0 this time, because it will not be used
    addi sp, sp, -4
    sw ra, 0(sp)
    
    call mySin               # Now fa0 has mySin
    fmv.s fs5, fa0 # fs5 = mysin
    
    lw ra, 0(sp)
    addi sp, sp, 4

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## NOW JUST SAVE WORD TO W_real I
    ## FIRST MAKE OFFSET i*4
    slli a7, a6, 2  # a7 = i*4
    add a7, a7, s0 # W_real address
    fsw fs4, 0(a7)
    
    slli a7, a6, 2  # a7 = i*4
    add a7, a7, s1 # W_imag address
    fsw fs5, 0(a7)
    
    addi a6, a6, 1  # i++
    j sincosfor
    sincosforend:
    
    ###########################
    
    addi s2, zero, 1  # s2 = n = 1
    srli s3, a2, 1  # s3 = a = N/2
    addi t0, zero, 0  # t0 = j = 0
    
    lw s4, logsize 
    transformfor1:
    bge t0, s4, endtransformfor1
   
   ## SECOND LOOP 
    addi t1, zero, 0 # t1 = i = 0
    transformfor2:
    bge t1, a2, transformfor2end
    
    and t2, t1, s2 # t2 = i AND n
    # xori t2, t2, -1 # t2 = !t2
    bne t2, zero, transformelse
    transformif:
    slli t3, t1, 2 #  i*4 offeset
    add t4, a0, t3 # real base + offset
    flw ft0, 0(t4)
    add t4, a1, t3 # imag base + offset
    flw ft1, 0(t4)
    
    mul t4, t1, s3 # i * a
    mul t5, s2, s3 # n * a
    rem t4, t4, t5 # k = t4 % t5
    
    slli t4, t4, 2 # offset, k * 4
    
    add t5, s0, t4 # W_real
    flw ft2, 0(t5)
    add t5, s1, t4  # W_imag
    flw ft3, 0(t5)
    
    
    add t5, t1, s2 # i + n
    slli t5, t5, 2 # offset
    # now load real[i+n] and image[i+n]
    add t6, t5, a0
    flw ft4, 0(t6)  # real[i+n
    add t6, t5, a1
    flw ft5, 0(t6)  # imag[i+n]
    
    fmul.s ft6, ft2, ft4 # W_real*real(i+n)
    fmul.s ft7, ft3, ft5 #  W_imag*imag(i+n)
    fsub.s fs1, ft6, ft7 #  W_real*real(i+n) - W_imag*imag(i+n)
    
    fmul.s ft6, ft2, ft5 # W_real*imag(i+n)
    fmul.s ft7, ft3, ft4 #  W_imag*real(i+n)
    fadd.s fs2, ft6, ft7 #  W_real*imag(i+n) + W_imag*real(i+n)
    
    fadd.s fs3, ft0, fs1 # save to real [i]
    
    slli s8, t1, 2 # i*4
    add s9, s8, a0 # real base + offset = real[i]
    fsw fs3, 0(s9) # real[i] = fs3 = ft0+fs1 = temp_real + temp1_real
    
    fadd.s fs3, ft1, fs2 # save to imag [i]
    
    slli s8, t1, 2 # i*4
    add s9, s8, a1 # imag base + offset = imag[i]
    fsw fs3, 0(s9) # imag[i] = fs3 = ft1+fs2 = temp__img + temp1_img
    
    
    fsub.s fs3, ft0, fs1 # save to real [i+n]
    
    add a7, t1, s2 # i + n
    slli s8, a7, 2 # (i+n)*4
    add s9, s8, a0 # real base + offset = real[i+n]
    fsw fs3, 0(s9) # real[i+n] = fs3 = ft0-fs1 = temp_real - temp1_real
    
    fsub.s fs3, ft1, fs2 # save to imag [i+n]
    
    add a7, t1, s2 # i + n
    slli s8, a7, 2 # (i+n)*4
    add s9, s8, a1 # real base + offset = real[i+n]
    fsw fs3, 0(s9) # imag[i+n] = fs3 = ft1-fs2 = temp_real - temp1_real
    
    transformelse:
    
    addi t1, t1, 1
    j transformfor2
    transformfor2end:
    
    ## SECOND LOOP ENDS HERE
    
    slli s2, s2, 1  # n = n * 2
    srli s3, s3, 1  # a = a / 2
    addi t0, t0, 1
    j transformfor1
    endtransformfor1:
    jr ra
    
    
FFT: # takes input real a0, imag a1, N a2
    addi sp, sp, -4
    sw ra, 0(sp)
    
    li a3, 1 # 0 is false, no inverse
    call transform
    
    lw ra, 0(sp)
    addi sp, sp, 4
    
    jr ra
    

IFFT: # takes input real a0, imag a1, N a2
    addi sp, sp, -4
    sw ra, 0(sp)
    
    li a3, -1 # 1 is true so inverse
    call transform
    
    lw ra, 0(sp)
    addi sp, sp, 4
    
    addi t0, zero, 0 # i = 0
    forloopifft:
    bge t0, a2, endforloopifft
    slli t1, t0, 2  # offset
    # lets do real first
    add t2, a0, t1 # real base a0 + offest t1
    flw ft2, 0(t2)
    fcvt.s.w ft3, a2 # N is in ft3
    fdiv.s ft2, ft2, ft3 # div by N
    fsw ft2, 0(t2)
    
    # now do imag
    add t2, a1, t1 # imag base a0 + offest t1
    flw ft2, 0(t2)
    fcvt.s.w ft3, a2 # N is in ft3
    fdiv.s ft2, ft2, ft3 # div by N
    fsw ft2, 0(t2)
    
    
    addi t0, t0, 1
    j forloopifft
    
    endforloopifft:
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
    TERMS: .word 12


    .align 2
    BitReverseTable:  # 256-byte lookup table
    .byte 0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0, 0x70, 0xF0
    .byte 0x08, 0x88, 0x48, 0xC8, 0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98, 0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8
    .byte 0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64, 0xE4, 0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4
    .byte 0x0C, 0x8C, 0x4C, 0xCC, 0x2C, 0xAC, 0x6C, 0xEC, 0x1C, 0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC
    .byte 0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2, 0x72, 0xF2
    .byte 0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A, 0xFA
    .byte 0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6, 0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6
    .byte 0x0E, 0x8E, 0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE, 0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE
    .byte 0x01, 0x81, 0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71, 0xF1
    .byte 0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9, 0x19, 0x99, 0x59, 0xD9, 0x39, 0xB9, 0x79, 0xF9
    .byte 0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5, 0x15, 0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5
    .byte 0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD, 0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD
    .byte 0x03, 0x83, 0x43, 0xC3, 0x23, 0xA3, 0x63, 0xE3, 0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3
    .byte 0x0B, 0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB, 0x3B, 0xBB, 0x7B, 0xFB
    .byte 0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97, 0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7
    .byte 0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F, 0xEF, 0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF


