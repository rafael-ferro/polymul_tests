import time
import numpy as np
from numba import jit, prange
from timeit import timeit

import cpython_polymul as cp
import fortran_polymul as ft


def naive_polymul(p1, p2):
    p3 = np.zeros(len(p1) + len(p2) - 1)

    for i1, c1 in enumerate(p1):
        for i2, c2 in enumerate(p2):
            p3[i1+i2] += c1 * c2

    return p3


def ndenumerate_polymul(p1, p2):
    p3 = np.zeros(len(p1) + len(p2) - 1)

    for i1, c1 in np.ndenumerate(p1):
        for i2, c2 in np.ndenumerate(p2):
            p3[i1[0]+i2[0]] += c1 * c2

    return p3


@jit(nopython=True, parallel=True)
def numba_polymul(p1, p2):
    p3 = np.zeros(len(p1) + len(p2) - 1)
    for i1 in prange(len(p1)):
        for i2 in prange(len(p2)):
            p3[i1+i2] += p1[i1] * p2[i2]

    return p3

if __name__=='__main__':
    np.random.seed(42)

    # Test arrays
    p1 = np.random.rand(1000)
    p2 = np.random.rand(1000)

    # Using numpy.polymul to check the results
    np_check = np.polymul(p1, p2)
    print("numpy result:")
    print(np_check)

#-----------------------------------------------------------------------------
# USING THE ENUMERATE FUNCTION IN THE FOR LOOPS
#-----------------------------------------------------------------------------
    print("\n\nUsing function naive_polymul(p1, p2)\n")

    start = time.time()
    p3 = naive_polymul(p1, p2)
    end = time.time()

    print("Elapsed time = %s\n" % (end - start))
    print(p3)
    print("numpy.polymul check:", np.isclose(p3, np_check))


#-----------------------------------------------------------------------------
# USING NUMPY NDENUMERATE
#-----------------------------------------------------------------------------
    print("\n\nUsing function ndenumerate_polymul(p1, p2)\n")

    start = time.time()
    p3 = ndenumerate_polymul(p1, p2)
    end = time.time()

    print("Elapsed time = %s\n" % (end - start))
    print(p3)
    print("numpy.polymul check:", np.isclose(p3, np_check))


#-----------------------------------------------------------------------------
# USING NUMBA
#-----------------------------------------------------------------------------
    print("\n\nUsing function numba_polymul(p1, p2)\n")
    # Compilation time is included in the execution time
    start = time.time()
    p3 = numba_polymul(p1, p2)
    end = time.time()
    print("Elapsed time (with compilation) = %s\n" % (end - start))
    # Now that the function is compiled,
    # re-time it executing from the cache
    start = time.time()
    p3 = numba_polymul(p1, p2)
    end = time.time()
    print("Elapsed time (after compilation) = %s\n" % (end - start))
    print(p3)
    print("numpy.polymul check:", np.isclose(p3, np_check))


#-----------------------------------------------------------------------------
# CUSTOM CPYTHON MODULE
#-----------------------------------------------------------------------------
    print("\n\nUsing function cpython_polymul with custom cpython code\n")

    start = time.time()
    p3 = cp.polymul(p1, p2)
    end = time.time()

    print("Elapsed time = %s\n" % (end - start))
    print(p3)
    print("numpy.polymul check:", np.isclose(p3, np_check))


#-----------------------------------------------------------------------------
# CUSTOM FORTRAN MODULE
#-----------------------------------------------------------------------------
    print("\n\nUsing function fortran_polymul with custom fortran code\n")

    start = time.time()
    p3 = ft.polymul(p1, p2, len(p1), len(p2))
    end = time.time()

    print("Elapsed time = %s\n" % (end - start))
    print(p3)
    print("numpy.polymul check:", np.isclose(p3, np_check))

