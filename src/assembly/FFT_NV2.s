#define STDOUT 0xd0580000

.section .text
.global _start
_start:

main:                               
    lw a0, size                     # Load size of real/imag arrays into a0
    call setlogN                    # Compute and store log2(size) for shared use by other functions

    la a0, real                     # Load address of real[] into a0
    la a1, imag                     # Load address of imag[] into a1
    lw a2, size                     # Load size of real/imag arrays into a2
    call FFT                        # Perform FFT on real[] and imag[] arrays

    la a0, real                     # Load address of real[] into a0
    la a1, imag                     # Load address of imag[] into a1
    lw a2, size                     # Load size of real/imag arrays into a2
    call IFFT                       # Perform IFFT on real[] and imag[] arrays

    la a0, real                     # Load address of real[] into a0
    la a1, imag                     # Load address of imag[] into a1
    lw a2, size                     # Load size of real/imag arrays into a2
    call print                      # Log the values of the arrays for debugging or display
    j _finish                       # Jump to program finish/exit



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



reverse:                            # Reverse the binary digits of the number. Takes input n (in t6)
    # use t0, t1, t2
    # Swap odd and even bits
    mv a6, t6
    li t0, 0x55555555    # Pattern for odd/even bits
    srli t1, a6, 1       # v >> 1
    and t1, t1, t0       # (v >> 1) & 0x55555555
    and t2, a6, t0       # v & 0x55555555
    slli t2, t2, 1       # (v & 0x55555555) << 1
    or a6, t1, t2        # Result back to a6

    # Swap consecutive pairs
    li t0, 0x33333333    # Pattern for pairs
    srli t1, a6, 2       # v >> 2
    and t1, t1, t0       # (v >> 2) & 0x33333333
    and t2, a6, t0       # v & 0x33333333
    slli t2, t2, 2       # (v & 0x33333333) << 2
    or a6, t1, t2        # Result back to a6

    # Swap nibbles
    li t0, 0x0F0F0F0F    # Pattern for nibbles
    srli t1, a6, 4       # v >> 4
    and t1, t1, t0       # (v >> 4) & 0x0F0F0F0F
    and t2, a6, t0       # v & 0x0F0F0F0F
    slli t2, t2, 4       # (v & 0x0F0F0F0F) << 4
    or a6, t1, t2        # Result back to a6

    # Swap bytes
    li t0, 0x00FF00FF    # Pattern for bytes
    srli t1, a6, 8       # v >> 8
    and t1, t1, t0       # (v >> 8) & 0x00FF00FF
    and t2, a6, t0       # v & 0x00FF00FF
    slli t2, t2, 8       # (v & 0x00FF00FF) << 8
    or a6, t1, t2        # Result back to a6

    # Swap 2-byte pairs
    srli t1, a6, 16      # v >> 16
    slli t2, a6, 16      # v << 16
    or a6, t1, t2        # Final result in a6

    # Save number of bits to reverse in t2
    # bits are in a7
    li t1, 32
    sub t1, t1, a7
    srl a6, a6, t1
    
    ret                            # Return with result in a0
    

preload_constants:
    # Load addresses of constants into registers
    # Make use of the fact that all float are 4 bytes and stored consecutively
    la      t0, half_pi_hi          # Load address of half_pi_hi
    flw     fs0, 0(t0)             # Load value into fs0
    flw     fs1, 4(t0)             # Load value into fs1
    flw     fs2, 8(t0)             # Load value into fs2
    flw     fs3, 12(t0)             # Load value into fs3

    # Load cosine coefficients
    flw     fs4, 16(t0)             # Load value into fs4
    flw     fs5, 20(t0)             # Load value into fs5
    flw     fs6, 24(t0)             # Load value into fs6
    flw     fs7, 28(t0)             # Load value into fs7
    flw     fs8, 32(t0)             # Load value into fs8

    # Load sine coefficients
    flw     fs9, 36(t0)             # Load value into fs9
    flw     fs10, 40(t0)            # Load value into fs10
    flw     fs11, 44(t0)            # Load value into fs11
    flw     ft11, 48(t0)            # Load value into ft11

    ret

sin_cos_approx:
    # Input: fa0 = a
    # Output: fa0 = sin, fa1 = cos

    # j = fmaf(a, 6.36619747e-1f, 12582912.f) - 12582912.f;
    fmadd.s ft5, fa0, fs2, fs3
    fsub.s ft5, ft5, fs3

    # a = fmaf(j, -half_pi_hi, a)
    fnmsub.s   ft6, ft5, fs0, fa0     # a = a - j * half_pi_hi

    # a = fmaf(j, -half_pi_lo, a)
    fnmsub.s   ft6, ft5, fs1, ft6     # a = a - j * half_pi_lo

    # Compute i = (int) j and i = i + 1
    fcvt.w.s t0, ft5                # Convert j to integer in t0
    addi    t1, t0, 1              # ic = i + 1
    

    # Compute sa = a * a
    fmul.s  ft7, ft6, ft6          # ft1 = a * a (sa)

    # Approximate cosine on [-π/4, +π/4] using polynomial approximation
    # c = 2.44677067e-5
    fmadd.s   fa2, fs4, ft7, fs5     # c = c * sa + -1.38877297e-3
    fmadd.s   fa2, fa2, ft7, fs6     # c = c * sa + 4.16666567e-2
    fmadd.s   fa2, fa2, ft7, fs7     # c = c * sa + -0.5
    fmadd.s   fa0, fa2, ft7, fs8     # c = c * sa + 1.0

    # Approximate sine on [-π/4, +π/4] using polynomial approximation
    fmadd.s   fa4, fs9, ft7, fs10     # s = s * sa + -1.98559923e-4
    fmadd.s   fa4, fa4, ft7, fs11     # s = s * sa + 8.33338592e-3
    fmadd.s   fa4, fa4, ft7, ft11     # s = s * sa + -0.166666672
    fmul.s ft9, ft6, ft7            # t = a * sa
    fmadd.s   fa1, fa4, ft9, ft6     # s = s * a

    #t0 is for sin . fa4. i
    #t1 is for cos . fa2. ic

    # r = (i & 1) ? c : s;
    andi    t5, t0, 1              # t0 = i & 1
    beqz    t5, ifsincos1        # If i & 1 == 0, jump to ifsincos1
    j       adjust_sign            # Jump to adjust_sign

ifsincos1:
    fmv.s fa4, fa0
    fmv.s fa0, fa1
    fmv.s   fa1, fa4

adjust_sign:
    andi    t0, t0, 2              # t0 = i & 2
    beqz    t0, sign1done           # If i & 2 == 0, skip sign flip
    fneg.s  fa0, fa0               # r = -r
    
sign1done:

    andi t1, t1, 2
    beqz t1, sign2done
    fneg.s fa1, fa1

sign2done:

    ret


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

    slli t3, a6, 2  # rev_index*4
    
    add t4, a0, t3  # real array index rev_index
    flw ft1, 0(t4)
    fsw ft1, 0(s0)
    addi s0, s0, 4  # real_temp array i

    
    add t4, a1, t3  # imag array index rev_index
    flw ft1, 0(t4)
    fsw ft1, 0(s1)
    addi s1, s1, 4  # imag_temp array i

    addi t6, t6, 1
    j forordina
    endforordina:

    la s0, real_temp
    la s1, imag_temp

    
    addi t1, zero, 0  # j
    forordina2:
    bge t1, a2, endforordina2

    flw ft1, 0(s0)  # load real_temp
    fsw ft1, 0(a0)  # save to real
    
    flw ft1, 0(s1)  # load imag_temp
    fsw ft1, 0(a1)  # save to real


    addi s0, s0, 4
    addi a0, a0, 4
    addi s1, s1, 4
    addi a1, a1, 4

    
    addi t1, t1, 1
    j forordina2
    endforordina2:
    
    lw ra, 0(sp)
    lw a0, 4(sp)
    lw a1, 8(sp)
    addi sp, sp, 12

    jr ra


 
transform:      # it receives base address of real[] a0, imag[] a1, and an int N a2, inverse flag in a3
    addi sp, sp, -4   # save ra to stack. only once because it is safe?
    sw ra, 0(sp)

    call ordina
    
    la s0, W_real
    la s1, W_imag
    
    addi a6, zero, 0            # a6 = i = 0
    fcvt.s.w fa6, a6        # fa6 = i convert i to float to use in sin/cos
    li s3, 1
    fcvt.s.w fa5, s3        # createa a floating 1 to keep adding to i. used to avoid converting i to float in loop
    srli s3, a2, 1              # s3 = N/2
    
    # Calculating (inverse)*-2*PI/N
    la a7, NEG_TWO_PI
    flw fa3, 0(a7)              # fa3 = -2*PI
    mul a7, a2, a3              # a7 = (inverse)*N
    fcvt.s.w fa7, a7            # fa7 = (inverse)*N
    fdiv.s fa3, fa3, fa7        # fa3 = (inverse)*-2*PI/N

    call preload_constants

    sincosfor:
    bge a6, s3, sincosforend
    
    fmul.s fa0, fa3, fa6    # fa0 is mulvalue

    call sin_cos_approx          # Now fa1 has myCos . RA is not saved because it is saved once in this function
    fsw fa1, 0(s0)      # save output to wreal
    fsw fa0, 0(s1)          # save output to wreal

    addi s0, s0, 4 # increment addreess
    addi s1, s1, 4 # increment addreess

    addi a6, a6, 1  # i++
    fadd.s fa6, fa6, fa5    # floating i increment
    j sincosfor
    sincosforend:

    ########
    #Instead of creating offest and adding it to base address or wreal and wimag
    # just keep incrementing them by 4 (word size)
    # and after loop end restore them
    la s0, W_real
    la s1, W_imag
    
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
    bne t2, zero, transformelse
    transformif:

    slli t3, t1, 2 #  i*4 offeset
    add s9, a0, t3 # real base + offset
    flw ft0, 0(s9)  # real[i]
    add s10, a1, t3 # imag base + offset
    flw ft1, 0(s10)  # imag[i]
    
    mul t4, t1, s3 # i * a
    mul t5, s2, s3 # n * a
    rem t4, t4, t5 # k = t4 % t5
    
    slli t4, t4, 2 # offset, k * 4

    add t5, s0, t4 # W_real
    flw ft2, 0(t5)  # w-real[k]
    add t5, s1, t4  # W_imag
    flw ft3, 0(t5) #w_imag[k]
    
    
    add t5, t1, s2 # i + n
    slli t5, t5, 2 # offset
    add s7, t5, a0
    flw ft4, 0(s7)  # real[i+n]
    add s8, t5, a1
    flw ft5, 0(s8)  # imag[i+n]

    # ft0 real[i]
    # ft1 imag[i]
    # ft2 wreal[k]
    # ft3 wimag[k]
    # ft4 real[i+n]
    # ft5 imag[i+n]

    fmul.s ft7, ft3, ft5 #  W_imag*imag(i+n)
    fmsub.s fs1, ft2, ft4, ft7   # #  W_real*real(i+n) - W_imag*imag(i+n)

    fmul.s ft7, ft3, ft4 #  W_imag*real(i+n)
    fmadd.s fs2, ft2, ft5, ft7 #  W_real*imag(i+n) + W_imag*real(i+n)

    # s7        real[i+n]
    # s8        imag[i+n]
    # s9        real[i]
    # s10       imag[i]

    fadd.s fs3, ft0, fs1 # save to real [i]
    fsw fs3, 0(s9) # real[i] = fs3 = ft0+fs1 = temp_real + temp1_real
    
    fadd.s fs3, ft1, fs2 # save to imag [i]
    fsw fs3, 0(s10) # imag[i] = fs3 = ft1+fs2 = temp__img + temp1_img
    
    fsub.s fs3, ft0, fs1 # save to real [i+n]
    fsw fs3, 0(s7) # real[i+n] = fs3 = ft0-fs1 = temp_real - temp1_real
    
    fsub.s fs3, ft1, fs2 # save to imag [i+n]
    fsw fs3, 0(s8) # imag[i+n] = fs3 = ft1-fs2 = temp_real - temp1_real
    
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


    lw ra, 0(sp)
    addi sp, sp, 4
    jr ra
   
    
# FFT:
#   Performs the FFT on real and imaginary inputs.
# Inputs:
#   a0 = real array address
#   a1 = imaginary array address
#   a2 = size (N)
# Output:
#   In-place modification of arrays (a0, a1).
# Clobbers:
#   a3    
FFT:                                       
    addi sp, sp, -4
    sw ra, 0(sp)
    
    li a3, 1                   # Set a3 to 1 (indicates non-inverse FFT)
    call transform             # Call the 'transform' function (performs FFT)
    
    lw ra, 0(sp)
    addi sp, sp, 4
    
    jr ra
    

# IFFT:
#   Performs the IFFT on real and imaginary inputs.
# Inputs:
#   a0 = real array address
#   a1 = imaginary array address
#   a2 = size (N)
# Output:
#   In-place modification of arrays (a0, a1).
# Clobbers:
#   a3    
IFFT: # takes input real a0, imag a1, N a2
    addi sp, sp, -4
    sw ra, 0(sp)
    
    li a3, -1 # 1 is true so inverse
    call transform
    
    lw ra, 0(sp)
    addi sp, sp, 4
    
    addi t0, zero, 0 # i = 0
    fcvt.s.w ft3, a2 # N is in ft3

    forloopifft:
    bge t0, a2, endforloopifft
    # lets do real first
    flw ft2, 0(a0)
    fdiv.s ft2, ft2, ft3 # div by N
    fsw ft2, 0(a0)
    
    # now do imag
    flw ft2, 0(a1)
    fdiv.s ft2, ft2, ft3 # div by N
    fsw ft2, 0(a1)
    
    addi a0, a0, 4  # increment address by word size
    addi a1, a1, 4
    addi t0, t0, 1
    j forloopifft
    
    endforloopifft:
    jr ra
    


# Function: print
# Logs values from real[] and imag[] arrays into registers ft0 and ft1 for debugging and output.
# Inputs:
#   - a0: Base address of real[] array
#   - a1: Base address of imag[] array
#   - a2: Size of array i.e. number of elements to log
# Clobbers: t0,t1, t2, ft0, ft1. a0 and a1 are modified
print:                      
    li t0, 0x123456                 # Pattern for help in python script
    li t0, 0x234567                 # Pattern for help in python script
    li t0, 0x345678                 # Pattern for help in python script

    mv t1, a0                       # Move address to temp register to avoid stacking
    mv t2, a1                       # Move address to temp register to avoid stacking
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



# Function: _finish
# VeeR Related function which writes to to_host which stops the simulator
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
