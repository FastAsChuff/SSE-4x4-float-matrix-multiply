void mmult8x8_ps_naive(const float **A, const float **B, float **C) {
  int i,j,k;
  float sum;
  for (i=0; i<8; i++) {
    for (j=0; j<8; j++) {
      sum = 0.0f;
      for (k=0; k<8; k++) {
        sum += A[i][k]*B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

#ifdef __AVX__
#ifndef __FMA__
static inline __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c) {
  __m256 temp = _mm256_mul_ps(a,b);
  return _mm256_add_ps(temp, c);
}
#endif
void mmult8x8_ps(const float **A, const float **B, float **C) {
  int i,j;
  __m256 a, b[8], c[8];
  for (i=0; i<8; i++) {
    b[i] = _mm256_loadu_ps(B[i]);
  }
  for (i=0; i<8; i++) {
    c[i] = _mm256_set1_ps(0);
    for (j=0; j<8; j++) {
      a = _mm256_set1_ps(A[i][j]);
      c[i] = _mm256_fmadd_ps(a, b[j], c[i]);
    }
  }
  for (i=0; i<8; i++) {
    _mm256_storeu_ps(C[i], c[i]);
  }
}

void mmult8x8_ps2(const float **A, const float **B, float **C) {
  // Bit faster but requires that C does not overlap A
  int i,j;
  __m256 a, b[8], c;
  for (i=0; i<8; i++) {
    b[i] = _mm256_loadu_ps(B[i]);
  }
  for (i=0; i<8; i++) {
    c = _mm256_set1_ps(0);
    for (j=0; j<8; j++) {
      a = _mm256_set1_ps(A[i][j]);
      c = _mm256_fmadd_ps(a, b[j], c);
    }
    _mm256_storeu_ps(C[i], c);
  }
}
#else
//Using the function pointers instead makes it much slower, so just copy the naive function.
void mmult8x8_ps(const float **A, const float **B, float **C) {
  int i,j,k;
  float sum;
  for (i=0; i<8; i++) {
    for (j=0; j<8; j++) {
      sum = 0.0f;
      for (k=0; k<8; k++) {
        sum += A[i][k]*B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

void mmult8x8_ps2(const float **A, const float **B, float **C) {
  int i,j,k;
  float sum;
  for (i=0; i<8; i++) {
    for (j=0; j<8; j++) {
      sum = 0.0f;
      for (k=0; k<8; k++) {
        sum += A[i][k]*B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

#endif
