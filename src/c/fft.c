#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265358979323846f
#define NEG_PI (-PI)
#define HALF_PI (PI / 2.0f)
#define NEG_HALF_PI (-HALF_PI)
#define TWO_PI (PI * 2.0f)
#define NEG_TWO_PI (-TWO_PI)
#define TERMS 12
#define MAX 200

int logint(int N) // Calculates the log2 of number
{
  int k = N, i = 0;
  while (k) {
    k >>= 1;
    i++;
  }
  return i - 1;
}

int reverse(int N, int n) // bit wise reverses the number
{
  int j, p = 0;
  for (j = 1; j <= logint(N); j++) {
    if (n & (1 << (logint(N) - j)))
      p |= 1 << (j - 1);
  }
  return p;
}

float mySin(float x) {
   // Reduce to [0, π/2] for better accuracy
    if (x < -HALF_PI) {
        x = -PI - x;
    } else if (x > HALF_PI) {
        x = PI - x;
    }
    
    float x2 = x * x;
    float term = x;
    float sum = x;
    float factorial = 1.0f;
    
    for (int i = 1; i <= TERMS; i++) {
        factorial *= (2*i) * (2*i + 1);
        term *= -x2;
        float next_term = term / factorial;
        sum += next_term;
    } 
    
    return sum;
}

float myCos(float x) {
  int sign = 1;
    // Reduce to [0, π/2] for better accuracy
    if (x < NEG_HALF_PI) {
        x = -PI - x;  // 
        sign = -1;
    } else if (x > HALF_PI) {
        x = PI - x;
        sign = -1;
    }
    float x2 = x * x;
    float term = 1.0f;  // First term is 1 for cosine
    float sum = 1.0f;   // Sum starts at 1 for cosine
    float factorial = 1.0f;
    
    for (int i = 1; i <= TERMS; i++) {
        factorial *= (2*i - 1) * (2*i);
        term *= -x2;
        float next_term = term / factorial;
        sum += next_term;
    } 
    
    return sign*sum;
}

float mySinOld(float x) {
    float term = x; // The first term is x
    float sum = x; // Initialize sum of series
    int sign = -1; // Alternating sign for each term

    for (int i = 3; i <= 2 * TERMS + 1; i += 2) {
        term *= x * x / ((i - 1) * i); // Calculate the next term in the series
        sum += sign * term; // Add the term to the sum
        sign = -sign; // Alternate the sign
    }

    return sum;
}

float myCosOld(float x) {
    float term = 1; // The first term is 1
    float sum = 1; // Initialize sum of series
    int sign = -1; // Alternating sign for each term

    for (int i = 2; i <= 2 * TERMS; i += 2) {
        term *= x * x / ((i - 1) * i); // Calculate the next term in the series
        sum += sign * term; // Add the term to the sum
        sign = -sign; // Alternate the sign
    }

    return sum;
}

// now i have to change complex to 2 array of float
void ordina(float* real, float* imag, int N) // using the reverse order in the array
{
  float real_temp[MAX], imag_temp[MAX];
  for (int i = 0; i < N; i++) {
    int rev_index = reverse(N, i);
    real_temp[i] = real[rev_index];
    imag_temp[i] = imag[rev_index];
  }
  for (int j = 0; j < N; j++) {
    real[j] = real_temp[j];
    imag[j] = imag_temp[j];
  }
}

void transform(float* real, float* imag, int N, bool inverse) //
{
  ordina(real, imag, N);    // first: reverse order
  float* W_real = (float*)malloc(N / 2 * sizeof(float));
  float* W_imag = (float*)malloc(N / 2 * sizeof(float));
   for (int i = 0; i < N / 2; i++) {
    float mulValue = -2.0 * PI * i / N;
    if (inverse) mulValue *= -1;
    W_real[i] = myCos(mulValue);
    W_imag[i] = mySin(mulValue);
  }
    
    
  int n = 1;
  int a = N / 2;
  for (int j = 0; j < logint(N); j++) {
    for (int i = 0; i < N; i++) {
      if (!(i & n)) {
        float temp_real = real[i];
        float temp_imag = imag[i];
        
        int k = (i * a) % (n * a);
        float W_real_k = W_real[k];
        float W_imag_k = W_imag[k];
        
        float temp1_real = W_real_k * real[i + n] - W_imag_k * imag[i + n];
        float temp1_imag = W_real_k * imag[i + n] + W_imag_k * real[i + n];
        
        real[i] = temp_real + temp1_real;
        imag[i] = temp_imag + temp1_imag;
        real[i + n] = temp_real - temp1_real;
        imag[i + n] = temp_imag - temp1_imag;
      }
    }
    n *= 2;
    a = a / 2;
  }
  free(W_real);
  free(W_imag);
}

void FFT(float* real, float* imag, int N, float d)
{
  transform(real, imag, N, false);
  for (int i = 0; i < N; i++) {
    real[i] *= d; // multiplying by step
    imag[i] *= d;
  }
}

void IFFT(float* real, float* imag, int N, float d)
{
  transform(real, imag, N, true);
  for (int i = 0; i < N; i++) {
    real[i] /= N; // multiplying by step
    imag[i] /= N;
  }
}

void testSineCosine(int N) {
    float max_error_new_sin = 0.0f, max_error_old_sin = 0.0f;
    float max_error_new_cos = 0.0f, max_error_old_cos = 0.0f;

    // Loop to test each i in the range [0, N/2)
    for (int i = 0; i < N / 2; i++) {
        float mulValue = -1.0f * TWO_PI * i / N;

        // Get values from your implementations
        float new_cos = myCos(mulValue);
        float new_sin = mySin(mulValue);
        float old_cos = myCosOld(mulValue);
        float old_sin = mySinOld(mulValue);

        // Get values from C library's cos and sin
        float c_cos = cosf(mulValue);
        float c_sin = sinf(mulValue);

        // Calculate errors for both sine and cosine
        float error_new_sin = fabsf(new_sin - c_sin);
        float error_old_sin = fabsf(old_sin - c_sin);
        float error_new_cos = fabsf(new_cos - c_cos);
        float error_old_cos = fabsf(old_cos - c_cos);


        // Update max errors
        if (error_new_sin > max_error_new_sin) max_error_new_sin = error_new_sin;
        if (error_old_sin > max_error_old_sin) max_error_old_sin = error_old_sin;
        if (error_new_cos > max_error_new_cos) max_error_new_cos = error_new_cos;
        if (error_old_cos > max_error_old_cos) max_error_old_cos = error_old_cos;
    }

    // Print the results
    printf("Results for N = %d:\n", N);
    printf("  Max Error (New Sin): %1.18f\n", max_error_new_sin);
    printf("  Max Error (Old Sin): %1.18f\n", max_error_old_sin);
    printf("  Max Error (New Cos): %1.18f\n", max_error_new_cos);
    printf("  Max Error (Old Cos): %1.18f\n", max_error_old_cos);

    // Print the differences between new and old implementations
    printf("  Max Error Difference (New vs Old Sin): %1.18f\n", fabsf(max_error_new_sin - max_error_old_sin));
    printf("  Max Error Difference (New vs Old Cos): %1.18f\n", fabsf(max_error_new_cos - max_error_old_cos));
    printf("\n");
}
// Function to compute sin(x) using Taylor series
float sin_approx(float a) {
  
  const float half_pi_hi =  1.57079637e+0f; //  0x1.921fb6p+0
    const float half_pi_lo = -4.37113883e-8f; // -0x1.777a5cp-25
    float c, j, r, s, sa, t;
    int i;

    /* subtract closest multiple of pi/2 giving reduced argument and quadrant */
    j = fmaf (a, 6.36619747e-1f, 12582912.f) - 12582912.f; // 2/pi, 1.5 * 2**23
    a = fmaf (j, -half_pi_hi, a);
    a = fmaf (j, -half_pi_lo, a);

    /* phase shift of pi/2 (one quadrant) for cosine */
    i = (int)j;
    //i = i + 1;

    sa = a * a;
    /* select sine approximation or cosine approximation based on quadrant */
    if (i & 1){ 
    /* Approximate cosine on [-PI/4,+PI/4] with maximum error of 0.87444 ulp */
    c =               2.44677067e-5f;  //  0x1.9a8000p-16
    c = fmaf (c, sa, -1.38877297e-3f); // -0x1.6c0efap-10
    c = fmaf (c, sa,  4.16666567e-2f); //  0x1.555550p-5
    c = fmaf (c, sa, -5.00000000e-1f); // -0x1.000000p-1
    r= fmaf (c, sa,  1.00000000e+0f); //  1.00000000p+0

    } else {
    /* Approximate sine on [-PI/4,+PI/4] with maximum error of 0.64196 ulp */
    s =               2.86567956e-6f;  //  0x1.80a000p-19
    s = fmaf (s, sa, -1.98559923e-4f); // -0x1.a0690cp-13
    s = fmaf (s, sa,  8.33338592e-3f); //  0x1.111182p-7
    s = fmaf (s, sa, -1.66666672e-1f); // -0x1.555556p-3
    t = a * sa;
    r = fmaf (s, t, a);
    }
    /* adjust sign based on quadrant */
    r = (i & 2) ? (0.0f - r) : r;

    return r;
}

// Function to compute cos(x) using Taylor series
float cos_approx(float a) {
  
  const float half_pi_hi =  1.57079637e+0f; //  0x1.921fb6p+0
    const float half_pi_lo = -4.37113883e-8f; // -0x1.777a5cp-25
    float c, j, r, s, sa, t;
    int i;

    /* subtract closest multiple of pi/2 giving reduced argument and quadrant */
    j = fmaf (a, 6.36619747e-1f, 12582912.f) - 12582912.f; // 2/pi, 1.5 * 2**23
    a = fmaf (j, -half_pi_hi, a);
    a = fmaf (j, -half_pi_lo, a);

    /* phase shift of pi/2 (one quadrant) for cosine */
    i = (int)j;
    i = i + 1;

    sa = a * a;
    /* select sine approximation or cosine approximation based on quadrant */

     if (i & 1){ 
    /* Approximate cosine on [-PI/4,+PI/4] with maximum error of 0.87444 ulp */
    c =               2.44677067e-5f;  //  0x1.9a8000p-16
    c = fmaf (c, sa, -1.38877297e-3f); // -0x1.6c0efap-10
    c = fmaf (c, sa,  4.16666567e-2f); //  0x1.555550p-5
    c = fmaf (c, sa, -5.00000000e-1f); // -0x1.000000p-1
    r= fmaf (c, sa,  1.00000000e+0f); //  1.00000000p+0

    } else {
    /* Approximate sine on [-PI/4,+PI/4] with maximum error of 0.64196 ulp */
    s =               2.86567956e-6f;  //  0x1.80a000p-19
    s = fmaf (s, sa, -1.98559923e-4f); // -0x1.a0690cp-13
    s = fmaf (s, sa,  8.33338592e-3f); //  0x1.111182p-7
    s = fmaf (s, sa, -1.66666672e-1f); // -0x1.555556p-3
    t = a * sa;
    r = fmaf (s, t, a);
    }
    /* adjust sign based on quadrant */
    r = (i & 2) ? (0.0f - r) : r;

    return r;
}

float* sin_cos_approx(float a) {
    static float result[2];
  
    const float half_pi_hi =  1.57079637e+0f; //  0x1.921fb6p+0
    const float half_pi_lo = -4.37113883e-8f; // -0x1.777a5cp-25
    float c, j, rc, rs, s, sa, t;
    int i, ic; //i for sin, ic for cos

    /* subtract closest multiple of pi/2 giving reduced argument and quadrant */
    j = fmaf (a, 6.36619747e-1f, 12582912.f) - 12582912.f; // 2/pi, 1.5 * 2**23
    a = fmaf (j, -half_pi_hi, a);
    a = fmaf (j, -half_pi_lo, a);

    /* phase shift of pi/2 (one quadrant) for cosine */
    i = (int)j;
    ic = i + 1;

    sa = a * a;
    /* select sine approximation or cosine approximation based on quadrant */

    
    /* Approximate cosine on [-PI/4,+PI/4] with maximum error of 0.87444 ulp */
    c =               2.44677067e-5f;  //  0x1.9a8000p-16
    c = fmaf (c, sa, -1.38877297e-3f); // -0x1.6c0efap-10
    c = fmaf (c, sa,  4.16666567e-2f); //  0x1.555550p-5
    c = fmaf (c, sa, -5.00000000e-1f); // -0x1.000000p-1
    c= fmaf (c, sa,  1.00000000e+0f); //  1.00000000p+0

    
    /* Approximate sine on [-PI/4,+PI/4] with maximum error of 0.64196 ulp */
    s =               2.86567956e-6f;  //  0x1.80a000p-19
    s = fmaf (s, sa, -1.98559923e-4f); // -0x1.a0690cp-13
    s = fmaf (s, sa,  8.33338592e-3f); //  0x1.111182p-7
    s = fmaf (s, sa, -1.66666672e-1f); // -0x1.555556p-3
    t = a * sa;
    s = fmaf (s, t, a);

    /* select sine approximation or cosine approximation based on quadrant */
    // if (i & 1) is tru then (ic & 1) is false because ic = i + 1

    if (i & 1) {
      rs = c;
      rc = s;
    } else {
      rs = s;
      rc = c;
    }
  

    /* adjust sign based on quadrant */
    rs = (i & 2) ? (0.0f - rs) : rs;
    rc = (ic & 2) ? (0.0f - rc) : rc;

    result[0] = rs;
    result[1] = rc;

    return result;
}


int main()
{
  int inverse = 0;  // Set to 1 for inverse
  float err = 0;
  int N = 1<<25;
    for (int i = 0; i < N/2; i++) {
        // Calculate mulValue
        float mulValue = -2.0 * PI * i / N;
        if (inverse) {
            mulValue *= -1;
        }

        // Calculate sin and cos using Taylor series
        float* r = sin_cos_approx(mulValue);
        float sin_taylor = r[0];
        float cos_taylor = r[1];

        // Calculate sin and cos using standard library functions
        float sin_lib = sin(mulValue);
        float cos_lib = cos(mulValue);

        // Compare results
        err += fabs(cos_taylor - cos_lib);
    }
    printf("Summed error: %.16f\n", err );
    return 0;
  int n = 8; // array size
    float d = 1;  // step size = 1
    float real[MAX] = {1, 2, 3, 4, 5, 6, 7, 8};
    float imag[MAX] = {0,0,0,0,1,1,1,1};
  
  FFT(real, imag, n, d);
  //printf("After FFT Value: \n");
  // Now the matrix has the FFT
  for (int j = 0; j < n; j++)
    printf("%f + %fi\n", real[j], imag[j]);
    
    IFFT(real, imag, n, d);
    printf("After Invese r FFT Value: \n");
    for (int j = 0; j < n; j++)
    printf("%f + %fi\n", real[j], imag[j]);
  return 0;
}