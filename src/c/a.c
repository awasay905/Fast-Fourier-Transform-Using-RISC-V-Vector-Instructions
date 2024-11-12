#include <stdio.h>
#include <stdint.h>

// Helper function to print a number in binary
void print_binary(uint32_t num, int bits) {
     
    for (int i = bits - 1; i >= 0; i--) {
        printf("%c", (num & (1U << i)) ? '1' : '0');
    }
    printf("\n");
}

// Function to increment a number in reverse using __builtin_clz
uint32_t increment_reversed(uint32_t num, int bits) {

    // Step 1: Create a mask where the first `bits` are 0 and the rest are 1 maskold =~((1U << bits) - 1);
    uint32_t mask = -1 << bits;
    // Step 2: OR the number with the mask
    num = num | mask;

    // Step 4: Count leading zeros in the negated number
    int leading_zeros = __builtin_clz( ~num);

    // Step 5: Calculate the position of the first 0-bit from the left
    int first_zero_index = 32 - leading_zeros;

    // Step 7: Set the bit at the first_zero_index to 1
    num |= (1U << ( first_zero_index - 1));

    // Step 8: Clear all bits to the left of the first_zero_index
    num &= ~((-1U << ( first_zero_index)) );

    return num;
}

int main() {
    int bits = 6;               // Number of bits being reversed

    // Call the function
    int num = 0;
    for (int i = 0; i < 1<<bits; i++){
        print_binary(num, bits);
   num = increment_reversed(num, bits);


    }



    return 0;
}
