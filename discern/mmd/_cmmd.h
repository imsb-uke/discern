
#ifndef CMMD_H
#define CMMD_H

#define USE_FLOAT

#ifdef USE_FLOAT
typedef float _DFLOAT;
#else
typedef double _DFLOAT;
#endif

double _loop(const _DFLOAT *scales, const unsigned long scale_size, _DFLOAT *dist_xx,
              _DFLOAT *dist_yy, _DFLOAT *dist_xy, _DFLOAT sigma,
              const unsigned long dist_xx_size, const unsigned long dist_yy_size);

#endif
