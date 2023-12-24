void mmult4x4_ps_naive(const float **A, const float **B, float **C) {
  int i,j,k;
  float sum;
  for (i=0; i<4; i++) {
    for (j=0; j<4; j++) {
      sum = 0.0f;
      for (k=0; k<4; k++) {
        sum += A[i][k]*B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

#ifdef __SSE__
#ifndef __FMA__
static inline __m128 _mm_fmadd_ps(__m128 a, __m128 b, __m128 c) {
  __m128 temp = _mm_mul_ps(a,b);
  return _mm_add_ps(temp, c);
}
#endif
void mmult4x4_ps(const float **A, const float **B, float **C) {
  int i,j;
  __m128 a, b[4], c[4];
  for (i=0; i<4; i++) {
    b[i] = _mm_loadu_ps(B[i]);
  }
  for (i=0; i<4; i++) {
    c[i] = _mm_set1_ps(0);
    for (j=0; j<4; j++) {
      a = _mm_set1_ps(A[i][j]);
      c[i] = _mm_fmadd_ps(a, b[j], c[i]);
    }
  }
  for (i=0; i<4; i++) {
    _mm_storeu_ps(C[i], c[i]);
  }
}

void mmult4x4_ps2(const float **A, const float **B, float **C) {
  // Bit faster but requires that C does not overlap A
  int i,j;
  __m128 a, b[4], c;
  for (i=0; i<4; i++) {
    b[i] = _mm_loadu_ps(B[i]);
  }
  for (i=0; i<4; i++) {
    c = _mm_set1_ps(0);
    for (j=0; j<4; j++) {
      a = _mm_set1_ps(A[i][j]);
      c = _mm_fmadd_ps(a, b[j], c);
    }
    _mm_storeu_ps(C[i], c);
  }
}
#else
//Using the function pointers instead makes it much slower, so just copy the naive function.
//void (*mmult4x4_ps)(const float **A, const float **B, float **C) = &mmult4x4_ps_naive;
//void (*mmult4x4_ps2)(const float **A, const float **B, float **C) = &mmult4x4_ps_naive;
void mmult4x4_ps(const float **A, const float **B, float **C) {
  int i,j,k;
  float sum;
  for (i=0; i<4; i++) {
    for (j=0; j<4; j++) {
      sum = 0.0f;
      for (k=0; k<4; k++) {
        sum += A[i][k]*B[k][j];
      }
      C[i][j] = sum;
    }
  }
}
void mmult4x4_ps2(const float **A, const float **B, float **C) {
  int i,j,k;
  float sum;
  for (i=0; i<4; i++) {
    for (j=0; j<4; j++) {
      sum = 0.0f;
      for (k=0; k<4; k++) {
        sum += A[i][k]*B[k][j];
      }
      C[i][j] = sum;
    }
  }
}
#endif
