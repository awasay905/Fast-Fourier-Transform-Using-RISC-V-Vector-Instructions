#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265358979323846
#define TWO_PI 6.28318530718 
#define TERMS 60
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
      // Convert x to positive and preserve the sign
    int sign = 1;
    if (x < 0) {
        x = -x;
        sign = -1;
    }

    // Reduce x to the range [0, 2π) efficiently
    x = x - ((int)(x / TWO_PI)) * TWO_PI;
    float term = x; // The first term is x
    float sum = x; // Initialize sum of series
    int alt_sign = -1; // Alternating sign for each term

    for (int i = 3; i <= 2 * TERMS + 1; i += 2) {
        term *= x * x / ((i - 1) * i); // Calculate the next term in the series
        sum += alt_sign * term; // Add the term to the sum
        alt_sign = -alt_sign; // Alternate the sign
    }

    return sign * sum;
}

float myCos(float x) {
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

int main()
{
  int n = 8; // array size
    float d = 1;  // step size = 1
    float real[MAX] = {1, 2, 3, 4, 5, 6, 7, 8};
    float imag[MAX] = {0,0,0,0,1,1,1,1};

printf("Original Value: \n");
 for (int j = 0; j < n; j++)
    printf("%f + %fi\n", real[j], imag[j]);
  
  FFT(real, imag, n, d);
  printf("After FFT Value: \n");
  // Now the matrix has the FFT
  for (int j = 0; j < n; j++)
    printf("%f + %fi\n", real[j], imag[j]);
    
    IFFT(real, imag, n, d);
    printf("After Invese r FFT Value: \n");
    for (int j = 0; j < n; j++)
    printf("%f + %fi\n", real[j], imag[j]);
  return 0;
}