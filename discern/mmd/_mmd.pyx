# cython: profile=False
# cython: linetrace=False
# cython: language_level=3, boundscheck=False
# cython: wraparound=False, nonecheck=False, cdivision=True

cdef extern from "_cmmd.h" nogil:
    ctypedef float _DFLOAT
    double _loop(const _DFLOAT* scales, const unsigned long scale_size,
                   _DFLOAT* dist_xx, _DFLOAT* dist_yy, _DFLOAT* dist_xy,
                   _DFLOAT sigma, const unsigned long dist_xx_size,
                   const unsigned long dist_yy_size)

def _mmd_loop(_DFLOAT[:,:] dist_xy, _DFLOAT[:,:] dist_xx, _DFLOAT[:,:] dist_yy, const _DFLOAT[:] scales, _DFLOAT sigma):
    cdef long dist_xx_size = dist_xx.shape[0]
    cdef long dist_yy_size = dist_yy.shape[0]
    cdef long scale_size = scales.shape[0]
    return _loop(&scales[0],scale_size,&dist_xx[0,0],&dist_yy[0,0],&dist_xy[0,0], sigma, dist_xx_size, dist_yy_size)
