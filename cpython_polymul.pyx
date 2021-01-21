import numpy as np

cimport numpy as np

DTYPE = np.int
ctypedef np.int_t DTYPE_t

def polymul(np.ndarray[DTYPE_t, ndim=1] p1, np.ndarray[DTYPE_t, ndim=1] p2):
    cdef int len_p3 = len(p1) + len(p2) - 1
    cdef np.ndarray[DTYPE_t, ndim=1] p3 = np.zeros(len_p3, dtype=DTYPE)
    cdef unsigned int i1, i2
    for i1 in range(len(p1)):
        for i2 in range(len(p2)):
            p3[i1+i2] += p1[i1] * p2[i2]
    return p3

