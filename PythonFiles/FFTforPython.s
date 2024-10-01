#define STDOUT 0xd0580000

.section .text
.global _start
_start:

main:
    la a0, real                     # a0 holds address of array real[]
    la a1, imag                     # a1 holds address of array imag[]
    la a2, size                     # use a2 to load size of arrays real/img
    lw a2, 0(a2)                    # a2 holds the size of arrays real/imag
    la a3, step                     # use a3 to load step for FFT/IFFT
    flw fa3, 0(a3)                  # 3 holds the step for FFT/IFFT
    
    call XXXX                        # Apply XXXX on the arrays
    
	call print                      # Writes down value of arrays to regsiter for helping in log
    j _finish                       # End program

logint:     # takes input N (a0), returns its log base 2 in a0
    addi sp, sp, -4
    sw t0, 0(sp)
    
    add t0, a0, zero                # k = N
    add a0, zero, zero              # i = 0

    logloop:
    beq t0, zero, logloopend
    srai t0, t0, 1
    addi a0, a0, 1
    j logloop
    logloopend:

    addi a0, a0, -1  
    lw t0, 0(sp)
    addi sp, sp, 4
    jr ra
    
    
reverse:    # takes input N(a0) and n(a1). reverse the number in binray
    addi sp, sp, -28
    sw ra, 0(sp)
    sw s0, 4(sp)
    sw s1, 8(sp)
    sw s2, 12(sp)
    sw s3, 16(sp)
    sw s4, 20(sp)
    sw s5, 24(sp)
    
    call logint                     # now a0 have log2(N)
    addi s0, zero, 1                # j = 1
    add s1, zero, zero              # p = 0
    
    forloopreverse:
    bgt s0, a0, forloopreverseend
    
    sub s2, a0, s0
    addi s3, zero, 1
    sll s3, s3, s2
    and s3, a1, s3
    beq s3, zero, elses3
    ifs3:
    addi s4, s0, -1
    addi s5, zero, 1
    sll s5, s5, s4
    or s1, s1, s5
    elses3:
    
    addi s0, s0, 1
    j forloopreverse
    
    forloopreverseend:
    add a0, s1, zero                # return p
    
    
    lw ra, 0(sp)
    lw s0, 4(sp)
    lw s1, 8(sp)
    lw s2, 12(sp)
    lw s3, 16(sp)
    lw s4, 20(sp)
    lw s5, 24(sp)
    addi sp, sp, 28
    jr ra
    
mySin:      # takes input x = fa0 and returns sin in fa0
    addi sp, sp, -52
    fsw ft0, 0(sp)
    fsw ft1, 4(sp)
    fsw ft2, 8(sp)
    fsw ft3, 12(sp)
    fsw ft4, 16(sp)
    fsw ft5, 20(sp)
    fsw fa1, 24(sp)
    sw t0, 28(sp)
    sw t1, 32(sp)
    sw t2, 36(sp)
    sw t3, 40(sp)
    sw t4, 44(sp)
    sw s0, 48(sp)
    
    fcvt.s.w ft0, zero  # make ft0=0
    fadd.s ft1, ft0, fa0  # ft1 = term = x
    fadd.s fa1, ft0, fa0  # fa1 = sum = x
    addi t0, zero, -1  # t0 = sign = -1
    
    addi s0, zero, 21
    addi t1, zero, 3 # i = 3
    sinfor:
    bgt t1, s0, sinforend
    
    addi t3, t1, -1  # i - 1
    mul t3, t3, t1   # (i-1)*i
    fcvt.s.w ft4, t3 # convert aboove to float
    fmul.s ft5, fa0, fa0
    fdiv.s ft5, ft5, ft4
    fmul.s ft1, ft1, ft5
    fcvt.s.w ft2, t0 # convert sign to float
    fmul.s ft3,ft2, ft1 # sign * term
    fadd.s fa1, fa1, ft3
    
    addi t2, zero, -1
    mul t0, t0, t2  # sign = -1 * sign
    addi t1, t1, 2 # i += 2
    j sinfor
    sinforend:
    
    fadd.s fa0, fa1, ft0
    
    
    flw ft0, 0(sp)
    flw ft1, 4(sp)
    flw ft2, 8(sp)
    flw ft3, 12(sp)
    flw ft4, 16(sp)
    flw ft5, 20(sp)
    flw fa1, 24(sp)
    lw t0, 28(sp)
    lw t1, 32(sp)
    lw t2, 36(sp)
    lw t3, 40(sp)
    lw t4, 44(sp)
    lw s0, 48(sp)
    addi sp, sp, 52
    jr ra

myCos:
    addi sp, sp, -60
    fsw ft0, 0(sp)
    fsw ft1, 4(sp)
    fsw ft2, 8(sp)
    fsw ft3, 12(sp)
    fsw ft4, 16(sp)
    fsw ft5, 20(sp)
    fsw fa1, 24(sp)
    sw t0, 28(sp)
    sw t1, 32(sp)
    sw t2, 36(sp)
    sw t3, 40(sp)
    sw t4, 44(sp)
    sw t5, 48(sp)
    sw t6, 52(sp)
    sw s0, 56(sp)
    
    
    fcvt.s.w ft0, zero  # make ft0=0
    addi t6, zero, 1
    fcvt.s.w ft6, t6  # ft6 is 1 in float
    fadd.s ft1, ft0, ft6  # ft1 = term = 1
    fadd.s fa1, ft0, ft6  # fa1 = sum = 1
    addi t0, zero, -1  # t0 = sign = -1
    
    addi s0, zero, 20
    addi t1, zero, 2 # i = 2
    cosfor:
    bgt t1, s0, cosforend
    
    addi t3, t1, -1  # i - 1
    mul t3, t3, t1   # (i-1)*i
    fcvt.s.w ft4, t3 # convert aboove to float
    fmul.s ft5, fa0, fa0
    fdiv.s ft5, ft5, ft4
    fmul.s ft1, ft1, ft5
    fcvt.s.w ft2, t0 # convert sign to float
    fmul.s ft3,ft2, ft1 # sign * term
    fadd.s fa1, fa1, ft3
    
    addi t2, zero, -1
    mul t0, t0, t2  # sign = -1 * sign
    addi t1, t1, 2 # i += 2
    j cosfor
    cosforend:
    
    fadd.s fa0, fa1, ft0
    
    flw ft0, 0(sp)
    flw ft1, 4(sp)
    flw ft2, 8(sp)
    flw ft3, 12(sp)
    flw ft4, 16(sp)
    flw ft5, 20(sp)
    flw fa1, 24(sp)
    lw t0, 28(sp)
    lw t1, 32(sp)
    lw t2, 36(sp)
    lw t3, 40(sp)
    lw t4, 44(sp)
    lw t5, 48(sp)
    lw t6, 52(sp)
    lw s0, 56(sp)
    addi sp, sp, 60
    jr ra

ordina: # it receives base address of real[] a0, imag[] a1, and an int N a2
    addi sp, sp, -44
    sw s0, 0(sp)
    sw s1, 4(sp)
    sw t0, 8(sp)
    sw a0, 12(sp)
    sw a1, 16(sp)
    sw ra, 20(sp)
    sw t1, 24(sp)
    sw t3, 28(sp)
    sw t4, 32(sp)
    sw t5, 36(sp)
    fsw ft1, 40(sp)
    
    la s0, real_temp
    la s1, imag_temp
    
    addi t0, zero, 0 # i
    
    forordina:
    bge t0, a2, endforordina
    
    addi sp, sp, -12
    sw a0, 0(sp)
    sw a1, 4(sp)
    sw ra, 8(sp)
    
    add a0, a2, zero # a0 = N
    add a1, t0, zero # a1 = i 
    
    call reverse
    # now a0 have reverse index, save it to t1
    add t1, a0, zero
    
    lw a0, 0(sp)
    lw a1, 4(sp)
    lw ra, 8(sp)
    addi sp, sp, 12
    
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
    lw t0, 8(sp)
    lw a0, 12(sp)
    lw a1, 16(sp)
    lw ra, 20(sp)
    lw t1, 24(sp)
    lw t3, 28(sp)
    lw t4, 32(sp)
    lw t5, 36(sp)
    flw ft1, 40(sp)
    addi sp, sp, 44
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
    #############################
    addi a6, zero, 0 # a6 = i = 0
    srli s3, a2, 1  # s3 = N/2
    sincosfor:
    bge a6, s3, sincosforend
    
    addi a7, zero, -2
    mul a7, a7, a3  # multiply a7 by a3. for fft a3 is 1, for ifft a3 is -1
    fcvt.s.w fa3, a7  # fa3 = -2.0
    la a7, PI
    flw fa4, 0(a7)  # fa4 = PI
    fcvt.s.w fa6, a6 # fa6 = i
    fcvt.s.w fa7, a2 # fa7 = N
    
    fmul.s fa5, fa3, fa4
    fmul.s fa2, fa5, fa6
    
    fdiv.s fa1, fa2, fa7  
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # now call myCos by first moving fa1 to fa0 
    fcvt.s.w ft0, zero
    fadd.s fa0, fa1, ft0
    
    addi sp, sp, -8
    sw ra, 0(sp)
    fsw fa1, 4(sp)
    
    call myCos
    # Now fa0 has myCos
    fcvt.s.w ft0, zero
    fadd.s fs4, ft0, fa0 # fs4 = mycos
    
    lw ra, 0(sp)
    flw fa1, 4(sp)
    addi sp, sp, 8
    
    ##############################################$%%%%%%%%%%%
    # now call mySin by first moving fa1 to fa0 
    fcvt.s.w ft0, zero
    fadd.s fa0, fa1, ft0
    
    addi sp, sp, -4
    sw ra, 0(sp)
    
    call mySin
    # Now fa0 has mySin 
    fadd.s fs5, ft0, fa0 # fs5 = mysin
    
    lw ra, 0(sp)
    addi sp, sp, 4
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## NOW JUST SAVE WORD TO W_real I
    ## FIRST MAKE OFFSET i*4
    la s0, W_real 
    slli a7, a6, 2  # a7 = i*4
    add a7, a7, s0 # W_real address
    fsw fs4, 0(a7)
    
    la s1, W_imag
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
    addi sp, sp, -8
    sw ra, 0(sp)
    sw a0, 4(sp)
    
    #call here 
    add a0, zero, a2
    call logint
    # now a0 have logInt, transfer it to s4
    add s4, a0, zero
    
    lw ra, 0(sp)
    lw a0, 4(sp)
    addi sp, sp, 8
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
    
    
FFT: # takes input real a0, imag a1, N a2, d fa3
    addi sp, sp, -24
    sw a0, 0(sp)
    sw a1, 4(sp)
    sw a2, 8(sp)
    sw ra, 12(sp)
    fsw fa3, 16(sp)
    sw a3, 20(sp)
    
    addi a3, zero, 1 # 0 is false, no inverse
    call transform
    
    lw a0, 0(sp)
    lw a1, 4(sp)
    lw a2, 8(sp)
    lw ra, 12(sp)
    flw fa3, 16(sp)
    lw a3, 20(sp)
    addi sp, sp, 24
    
    addi t0, zero, 0 # i = 0
    forloopfft:
    bge t0, a2, endforloopfft
    slli t1, t0, 2  # offset
    # lets do real first
    add t2, a0, t1 # real base a0 + offest t1
    flw ft2, 0(t2)
    fmul.s ft2, ft2, fa3 # mul by d
    fsw ft2, 0(t2)
    
    # now do imag
    add t2, a1, t1 # imag base a0 + offest t1
    flw ft2, 0(t2)
    fmul.s ft2, ft2, fa3 # mul by d
    fsw ft2, 0(t2)
    
    
    addi t0, t0, 1
    j forloopfft
    
    endforloopfft:
    jr ra
    

IFFT: # takes input real a0, imag a1, N a2, d fa3
    addi sp, sp, -24
    sw a0, 0(sp)
    sw a1, 4(sp)
    sw a2, 8(sp)
    sw ra, 12(sp)
    fsw fa3, 16(sp)
    sw a3, 20(sp)
    
    addi a3, zero, -1 # 1 is true so inverse
    call transform
    
    lw a0, 0(sp)
    lw a1, 4(sp)
    lw a2, 8(sp)
    lw ra, 12(sp)
    flw fa3, 16(sp)
    lw a3, 20(sp)
    addi sp, sp, 24
    
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


# PUT INPUT HERE, DO NO CHANGE THIS LINE
# DO NOT CHANGE ANYTHING AFTER THIS LINE
.data  

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

