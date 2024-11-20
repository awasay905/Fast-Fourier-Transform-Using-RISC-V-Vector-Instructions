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
    call IFFT                       # Perform IFFT on real[] and imag[] arrays
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



# Function: reverse
# Reverses the binary digits of a 32-bit integer.
# Inputs:
#   - a0: Input number to reverse.
#   - a1: Number of significant bits to reverse (optional; default 32).
# Outputs:
#   - a0: The reversed binary number.
# Clobbers:
#   - t0, t1, t2. 
reverse:                            
    # Swap odd and even bits
    li t0, 0x55555555    # Pattern for odd/even bits
    srli t1, a0, 1       # v >> 1
    and t1, t1, t0       # (v >> 1) & 0x55555555
    and t2, a0, t0       # v & 0x55555555
    slli t2, t2, 1       # (v & 0x55555555) << 1
    or a0, t1, t2        # Result back to a0

    # Swap consecutive pairs
    li t0, 0x33333333    # Pattern for pairs
    srli t1, a0, 2       # v >> 2
    and t1, t1, t0       # (v >> 2) & 0x33333333
    and t2, a0, t0       # v & 0x33333333
    slli t2, t2, 2       # (v & 0x33333333) << 2
    or a0, t1, t2        # Result back to a0

    # Swap nibbles
    li t0, 0x0F0F0F0F    # Pattern for nibbles
    srli t1, a0, 4       # v >> 4
    and t1, t1, t0       # (v >> 4) & 0x0F0F0F0F
    and t2, a0, t0       # v & 0x0F0F0F0F
    slli t2, t2, 4       # (v & 0x0F0F0F0F) << 4
    or a0, t1, t2        # Result back to a0

    # Swap bytes
    li t0, 0x00FF00FF    # Pattern for bytes
    srli t1, a0, 8       # v >> 8
    and t1, t1, t0       # (v >> 8) & 0x00FF00FF
    and t2, a0, t0       # v & 0x00FF00FF
    slli t2, t2, 8       # (v & 0x00FF00FF) << 8
    or a0, t1, t2        # Result back to a0

    # Swap 2-byte pairs
    srli t1, a0, 16      # v >> 16
    slli t2, a0, 16      # v << 16
    or a0, t1, t2        # Final result in a0

    # Save number of bits to reverse in t2
    # bits are in a7
    li t1, 32
    sub t1, t1, a1
    srl a0, a0, t1
    
    ret                            # Return with result in a0
    


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
#   - fa0 = angle (a) in radians
# Output:
#   - fa0 = sin(a) (approximation)
#   - fa1 = cos(a) (approximation)
# Clobbers:
#   - t0, t1, ft0, ft1, ft2, ft3
# Help:
#   - i = t0, ic = t1, j = ft0, a = ft1, sa = ft2, t = ft3
sin_cos_approx:
    fmadd.s ft0, fa0, fs2, fs3          # j = fmaf(a, 6.36619747e-1f, 12582912.f)
    fsub.s ft0, ft0, fs3                # j = fmaf(a, 6.36619747e-1f, 12582912.f) - 12582912.f;

    fnmsub.s   ft1, ft0, fs0, fa0       # a = a - j * half_pi_hi
    fnmsub.s   ft1, ft0, fs1, ft1       # a = a - j * half_pi_lo

    fcvt.w.s t0, ft0                #  i = (int) j
    addi    t1, t0, 1              # ic = i + 1

    fmul.s  ft2, ft1, ft1          # ft2 = a * a (sa)

    # Approximate cosine. By default save it to fa0
    fmadd.s   fa0, fs4, ft2, fs5     # c = c * sa + -1.38877297e-3
    fmadd.s   fa0, fa0, ft2, fs6     # c = c * sa + 4.16666567e-2
    fmadd.s   fa0, fa0, ft2, fs7     # c = c * sa + -0.5
    fmadd.s   fa0, fa0, ft2, fs8     # c = c * sa + 1.0

    # Approximate sine. By default save it to fa1
    fmadd.s   fa1, fs9, ft2, fs10     # s = s * sa + -1.98559923e-4
    fmadd.s   fa1, fa1, ft2, fs11     # s = s * sa + 8.33338592e-3
    fmadd.s   fa1, fa1, ft2, ft11     # s = s * sa + -0.166666672
    fmul.s    ft3, ft1, ft2           # t = a * sa
    fmadd.s   fa1, fa1, ft3, ft1      # s = s * a

    # Check the value of i and adjust the order of sine and cosine if needed
    andi t2, t0, 1                     # t2 = i & 1
    beqz t2, ifsincos                   # If i & 1 == 0, jump to ifsincos
    j adjust_sign                       # Jump to adjust_sign

    ifsincos:
        fmv.s ft0, fa0                  # Swap sine and cosine
        fmv.s fa0, fa1
        fmv.s fa1, ft0

    adjust_sign:
        andi t0, t0, 2                  # t0 = i & 2
        beqz t0, sign1done               # If i & 2 == 0, skip sign flip
        fneg.s fa0, fa0                  # Negate sine if i & 2 != 0

    sign1done:
        andi t1, t1, 2                  # t1 = ic & 2
        beqz t1, sign2done              # If ic & 2 == 0, skip sign flip
        fneg.s fa1, fa1                  # Negate cosine if ic & 2 != 0

    sign2done:
        ret                              # Return with sine in fa0, cosine in fa1



# Function: ordina
# Reorders real[] and imag[] arrays based on bit-reversed indices.
# Inputs:
#   - a0: Base address of real[] array
#   - a1: Base address of imag[] array
#   - a2: Size of the arrays (N)
# Outputs:
#   - real[] and imag[] reordered in place
# Clobbers:
#   - t0, t1, t2, t3, t4, t5, ft0, ft1
ordina: 
    addi sp, sp, -20
    sw ra, 0(sp)
    sw a0, 4(sp)
    sw a1, 8(sp)
    sw a3, 12(sp)
    sw a4, 16(sp)
    
    la t4, real_temp               # Load address of temporary real array
    la t5, imag_temp               # Load address of temporary imag array
    mv a3, a0                      # Copy real[] base to a3
    mv a4, a1                      # Copy imag[] base to a4

    lw a1, logsize

    li t3,  0 # i
    forordina:
    bge t3, a2, endforordina

    mv a0, t3                      # Move i to a0 for reverse function
    call reverse                   # Compute bit-reversed index for i

    # Generate Reversed Index Offset
    slli t2, a0, 2  
    add t0, a3, t2  
    add t1, a4, t2 

    # Load from real array
    flw ft0, 0(t0)
    flw ft1, 0(t1)

    # Save to temp array
    fsw ft0, 0(t4)
    fsw ft1, 0(t5)

    # Increment Address
    addi t4, t4, 4  
    addi t5, t5, 4  

    addi t3, t3, 1
    j forordina
    endforordina:

    la t4, real_temp
    la t5, imag_temp
    
    addi t0, zero, 0  # i
    forordina2:
    bge t0, a2, endforordina2

    # Load from temp array
    flw ft0, 0(t4)  
    flw ft1, 0(t5) 

    # Save to normal array
    fsw ft0, 0(a3)  
    fsw ft1, 0(a4)  

    # Increment address
    addi t4, t4, 4
    addi t5, t5, 4
    addi a3, a3, 4
    addi a4, a4, 4

    addi t0, t0, 1
    j forordina2
    endforordina2:
    
    lw ra, 0(sp)
    lw a0, 4(sp)
    lw a1, 8(sp)
    lw a3, 12(sp)
    lw a4, 16(sp)
    addi sp, sp, 20

    jr ra


 
transform:      # it receives base address of real[] a0, imag[] a1, and an int N a2, inverse flag in a3
    addi sp, sp, -4 
    sw ra, 0(sp)

    call ordina 

    la s0, W_real
    la s1, W_imag


    call preload_constants 
    
    # Calculating (inverse)*-2*PI/N
    mul t0, a2, a3              # t0 = (inverse)*N
    fcvt.s.w ft0, t0            # ft0 = (inverse)*N
    la t0, NEG_TWO_PI
    flw ft4, 0(t0)              # ft4 = -2*PI
    fdiv.s ft4, ft4, ft0        # ft4 = (inverse)*-2*PI/N

    addi a6, zero, 0            # a6 = i = 0
    fcvt.s.w fa6, a6        # fa6 = i convert i to float to use in sin/cos
    li s3, 1
    fcvt.s.w fa5, s3        # createa a floating 1 to keep adding to i. used to avoid converting i to float in loop
    srli s3, a2, 1              # s3 = N/2

    sincosfor:
    bge a6, s3, sincosforend
    
    # Call sin_cos_approx. cos returned in fa1, sin in fa0
    fmul.s fa0, fa3, fa6    # fa0 is mulvalue
    call sin_cos_approx      

    # Save cos/sin to W array
    fsw fa1, 0(s0)      
    fsw fa0, 0(s1)    

    # Increment Address
    addi s0, s0, 4 
    addi s1, s1, 4 

    # Increment Loop index + float
    addi a6, a6, 1  
    fadd.s fa6, fa6, fa5   
    j sincosfor
    sincosforend:

    # ft0 real[i]
    # ft1 imag[i]
    # ft2 wreal[k]
    # ft3 wimag[k]
    # ft4 real[i+n]
    # ft5 imag[i+n]

    # t0    j for outer loop
    # t1    i for inner loop
    # t2    ( i AND n) for condition
    # t3    temp. used in addr. cacl. in loop
    # t4    temp. used in addr. cacl. in loop
    # t5    n
    # t6    a  
    # a0    real base address
    # a1    imag base address
    # a2    N size
    # a3    inverse flag. Useless after sin/cos. now stride. (n*a)
    # a4    real[i]
    # a5    imag[i]
    # a6    real[i+n]
    # a7    imag[i+n]   

    

    la s0, W_real
    la s1, W_imag
    
    addi t5, zero, 1                        # n = 1
    srli t6, a2, 1                          # a = N/2
    lw s4, logsize                          # s4 = log(N)

    addi t0, zero, 0                        # j = 0
    transformloop1:
    bge t0, s4, endtransformloop1
   
    # Calculating Stride (n * 1)
    mul a3, t5, t6

    addi t1, zero, 0                        # i = 0                 
    transformforloop2:
    bge t1, a2, transformforloop2end
    
    and t2, t1, t5                          # t2 = i AND n
    bne t2, zero, transformelse             # Skip conditon

    transformif:
    # Load real[i] and imag[i]
    slli t3, t1, 2                          #  Calculate array index offset
    add a4, a0, t3                          # real base + offset
    add a5, a1, t3                         # imag base + offset
    flw ft0, 0(a4)                          # real[i]
    flw ft1, 0(a5)                         # imag[i]
    
    # Calculate k ((i * a) % stride)
    mul t3, t1, t6                          
    rem t3, t3, a3                         

    # Load real[k] and imag[k]
    slli t3, t3, 2 # offset, k * 4
    add t4, s0, t3 # W_real
    add t3, s1, t3  # W_imag
    flw ft2, 0(t4)  # w-real[k]
    flw ft3, 0(t3) #w_imag[k]
    
    # Load real[i+n] and imag[i+n]
    add t4, t1, t5 # i + n
    slli t4, t4, 2 # offset
    add a6, t4, a0
    add a7, t4, a1
    flw ft4, 0(a6)  # real[i+n]
    flw ft5, 0(a7)  # imag[i+n]

    # Apply Transformation on Loaded Values
    fmul.s ft6, ft3, ft5 #  W_imag*imag(i+n)
    fmsub.s ft6, ft2, ft4, ft6   # #  W_real*real(i+n) - W_imag*imag(i+n)

    fmul.s ft7, ft3, ft4 #  W_imag*real(i+n)
    fmadd.s ft7, ft2, ft5, ft7 #  W_real*imag(i+n) + W_imag*real(i+n)

    fadd.s ft8, ft0, ft6                # ft8 = temp_real + temp1_real
    fadd.s ft9, ft1, ft7                # ft9 = temp__img + temp1_img
    fsub.s ft10, ft0, ft6               # ft10 = temp_real - temp1_real
    fsub.s ft11, ft1, ft7               # ft11 = temp_real - temp1_real

    # Save Values Back
    fsw ft8, 0(a4)                      # real[i]
    fsw ft9, 0(a5)                      # imag[i] 
    fsw ft10, 0(a6)                     # real[i+n]
    fsw ft11, 0(a7)                     # imag[i+n]
    
    transformelse:
    
    addi t1, t1, 1                      # i += 1
    j transformforloop2
    transformforloop2end:
    
    slli t5, t5, 1                      # n *= 2
    srli t6, t6, 1                      # a /= 2
    addi t0, t0, 1                      # j += 1
    j transformloop1
    endtransformloop1:

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
#   None   
FFT:                                       
    addi sp, sp, -8
    sw ra, 0(sp)
    sw a3, 4(sp)
    
    li a3, 1                            # Set a3 to 1 (indicates non-inverse FFT)
    call transform                      # Call the 'transform' function (performs FFT)
    
    lw ra, 0(sp)
    lw a3, 4(sp)
    addi sp, sp, 8
    
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
#   t0, ft0, ft1, ft2
IFFT: 
    addi sp, sp, -16            # Save a0, a1 etc to stack because these addresses
    sw ra, 0(sp)                # Are modified when dividing
    sw a0, 4(sp)    
    sw a1, 8(sp)
    sw a3, 12(sp)
    
    li a3, -1                   # Set a3 to -1 (indicates inverse FFT)
    call transform              # Call the 'transform' function (performs IFFT)
    
    li t0, 0                    # i = 0
    fcvt.s.w ft2, a2            # N for division in float

    forloopifft:
    bge t0, a2, endforloopifft
    
    # Load Real/Imag Pair
    flw ft0, 0(a0)
    flw ft1, 0(a1)

    # Divide by N
    fdiv.s ft0, ft0, ft2 
    fdiv.s ft1, ft1, ft2 

    # Save back to memory
    fsw ft0, 0(a0)
    fsw ft1, 0(a1)
    
    # Increment address by word size
    addi a0, a0, 4  
    addi a1, a1, 4

    addi t0, t0, 1
    j forloopifft
    endforloopifft:

    lw ra, 0(sp)
    lw a0, 4(sp)
    lw a1, 8(sp)
    lw a3, 12(sp)
    addi sp, sp, 16

    jr ra
    


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
