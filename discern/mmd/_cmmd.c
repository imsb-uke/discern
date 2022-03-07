
#include <float.h>
#include <math.h>

#include "_cmmd.h"

#ifdef USE_FLOAT
#define EXP(X) expf((X))
#else
#define EXP(X) exp((X))
#endif

#define MAX(a, b)                                                              \
  ({                                                                           \
    __typeof__(a) _a = (a);                                                    \
    __typeof__(b) _b = (b);                                                    \
    _a > _b ? _a : _b;                                                         \
  })
#define MIN(a, b)                                                              \
  ({                                                                           \
    __typeof__(a) _a = (a);                                                    \
    __typeof__(b) _b = (b);                                                    \
    _a < _b ? _a : _b;                                                         \
  })

double exponentiation_symmetric_matrix(_DFLOAT *array, const _DFLOAT scale,
                                       const unsigned long array_size) {
  unsigned long j;
  double array_sum = 0.;
  _DFLOAT value;
  unsigned long index;
  for (unsigned long i = 0; i < array_size; ++i) {
    for (j = i + 1; j < array_size; ++j) {
      index = i * array_size + j;
      value = -array[index] / scale;
      array_sum += (double)EXP(value);
    }
  }
  return array_sum;
}
double exponentiation_non_symmetric_matrix(_DFLOAT *array, const _DFLOAT scale,
                                           const unsigned long array_dim1,
                                           const unsigned long array_dim2) {
  _DFLOAT value;
  double array_sum = 0.;
  for (unsigned long i = 0; i < array_dim1 * array_dim2; ++i) {
    value = -array[i] / scale;
    array_sum += (double)EXP(value);
  }
  return array_sum;
}

double _loop(const _DFLOAT *scales, const unsigned long scale_size,
             _DFLOAT *dist_xx, _DFLOAT *dist_yy, _DFLOAT *dist_xy,
             const _DFLOAT sigma, const unsigned long dist_xx_size,
             const unsigned long dist_yy_size) {
  _DFLOAT val;
  double k_sum, k_xxnd, k_yynd, res1, res2;
  double stat = -INFINITY;

  for (unsigned long i = 0; i < scale_size; ++i) {
    val = scales[i] * sigma;
    k_sum = exponentiation_symmetric_matrix(dist_xx, 2.0 * val, dist_xx_size);
    k_xxnd = k_sum / ((dist_xx_size * dist_xx_size - dist_xx_size) / 2.0);

    k_sum = exponentiation_symmetric_matrix(dist_yy, 2.0 * val, dist_yy_size);
    k_yynd = k_sum / ((dist_yy_size * dist_yy_size - dist_yy_size) / 2.0);

    res1 = k_xxnd + k_yynd;

    res2 = exponentiation_non_symmetric_matrix(dist_xy, 2.0 * val, dist_xx_size,
                                               dist_yy_size);
    res2 /= ((dist_xx_size * dist_yy_size) / 2.0);

    stat = MAX(res1 - res2, stat);
  }
  return stat;
}

#undef EXP
#undef MIN
#undef MAX
