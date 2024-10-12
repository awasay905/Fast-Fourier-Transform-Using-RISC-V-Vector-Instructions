#define STDOUT 0xd0580000

.section .text
.global _start
_start:

main:                               # Main Function to Call FFT/IFFT
    la a0, real                     # a0 points to real[]
    la a1, imag                     # a1 points to imag[]
    la a2, size 
    lw a2, 0(a2)                    # a2 has size of real/imag arrays  
    
    call XXXX                        # Apply XXXX on the arrays
    
	call print                      # Writes arrays values to register for logging
    j _finish                       # End program


logint:                             # Returns log(N) base 2 where N=a0
    add t0, a0, zero                # k = N
    add a0, zero, zero              # i = 0

    logloop:
    beq t0, zero, logloopend
    srai t0, t0, 1
    addi a0, a0, 1
    j logloop
    logloopend:

    addi a0, a0, -1  
    jr ra
 
    
reverse:                            # Reverse the binary digits of the number. Takes input N (in a0) and n (in a1).
    addi sp, sp, -4
    sw ra, 0(sp)
    
    call logint                     # Now a0 = log2(N)

    lw ra, 0(sp)
    addi sp, sp, 4

    addi t0, zero, 1                # j = 1
    add t1, zero, zero              # p = 0
    
    forloopreverse:
    bgt t0, a0, forloopreverseend
    
    sub t2, a0, t0
    addi t3, zero, 1
    sll t3, t3, t2
    and t3, a1, t3
    beq t3, zero, elses3
    ifs3:
    addi t4, t0, -1
    addi t5, zero, 1
    sll t5, t5, t4
    or t1, t1, t5
    elses3:
    
    addi t0, t0, 1
    j forloopreverse
    
    forloopreverseend:
    add a0, t1, zero                # return p
    
    jr ra
    

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
    addi sp, sp, -20
    sw s0, 0(sp)
    sw s1, 4(sp)
    sw a0, 8(sp)
    sw a1, 12(sp)
    sw ra, 16(sp)
    
    la s0, real_temp
    la s1, imag_temp
    
    addi t0, zero, 0 # i
    
    forordina:
    bge t0, a2, endforordina
    
    addi sp, sp, -16
    sw a0, 0(sp)
    sw a1, 4(sp)
    sw ra, 8(sp)
    sw t0, 12(sp) 
    
    add a0, a2, zero # a0 = N
    add a1, t0, zero # a1 = i 
    
    call reverse
    # now a0 have reverse index, save it to t1
    add t1, a0, zero
    
    lw a0, 0(sp)
    lw a1, 4(sp)
    lw ra, 8(sp)
    lw t0, 12(sp)
    addi sp, sp, 16
    
    slli t2, t0, 2  # i*4
    slli t3, t1, 2  # rev_index*4
    
    add t4, a0, t3  # real array index rev_index
    add t5, s0, t2  # real_temp array i
    flw ft1, 0(t4)
    fsw ft1, 0(t5)
    
    add t4, a1, t3  # imag array index rev_index
    add t5, s1, t2  # imag_temp array i
    flw ft1, 0(t4)
    fsw ft1, 0(t5)
    
    addi t0, t0, 1
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
    
    lw s0, 0(sp)
    lw s1, 4(sp)
    lw a0, 8(sp)
    lw a1, 12(sp)
    lw ra, 16(sp)
    addi sp, sp, 20

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
    
    # for loop  need logint(N)
    addi sp, sp, -12
    sw ra, 0(sp)
    sw a0, 4(sp)
    sw t0, 8(sp)
    
    #call here 
    add a0, zero, a2
    call logint
    # now a0 have logInt, transfer it to s4
    add s4, a0, zero
    
    lw ra, 0(sp)
    lw a0, 4(sp)
    lw t0, 8(sp)
    addi sp, sp, 12
    ## call end, s4 have logint(N)
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
    addi sp, sp, -16
    sw a0, 0(sp)
    sw a1, 4(sp)
    sw a2, 8(sp)
    sw ra, 12(sp)
    
    addi a3, zero, 1 # 0 is false, no inverse
    call transform
    
    lw a0, 0(sp)
    lw a1, 4(sp)
    lw a2, 8(sp)
    lw ra, 12(sp)
    addi sp, sp, 16
    
    jr ra
    

IFFT: # takes input real a0, imag a1, N a2
    addi sp, sp, -16
    sw a0, 0(sp)
    sw a1, 4(sp)
    sw a2, 8(sp)
    sw ra, 12(sp)
    
    addi a3, zero, -1 # 1 is true so inverse
    call transform
    
    lw a0, 0(sp)
    lw a1, 4(sp)
    lw a2, 8(sp)
    lw ra, 12(sp)
    addi sp, sp, 16
    
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


    imag: .float 0,0,0,0, 1,1,1,1, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0
          .float 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0
          .float 0,0,0,0, 1,1,1,1, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0
          .float 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8
          .float 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0
          .float 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0

    .set dataSize, 256          # THIS IS N

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
    TERMS: .word 12

