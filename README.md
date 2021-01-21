# Testing different approaches to polynomial multiplication in Python

A polynomial _p(x) = a\_n x^n + a\_{n-1} x^{n-1} + ... + a\_1 x + a\_0_ can be represented by a 1D array of its coefficients (from highest to lowest degree, in this case).

## Objective

Given two arrays, _p1_ and _p2_, containing the coefficients of two polynomials, calculate the product of the two.

## Solution

Apply the distributive property by iterating over the input arrays, multiplying the coefficients and adding those with equivalent exponent (corresponding to the index of the final array).

The order of a polynomial _p_ is given by _len(p)-1_, since the
corresponding array contains the constant term (_x^0_).
The resulting polynomial is thus of order _(len(p1)-1 + len(p2)-1)_, and the array to store its coefficients must have a length of _(len(p1)-1 + len(p2)-1) + 1_, in order to account for the new constant term.

## Methodology

In order to try to optimize the nested for loops inherent to the algorithm, I've tested and compared 5 different approaches:

- using the function _enumerate_ to obtain both the index and the values of the arrays;

- using the numpy function [ndenumerate](https://numpy.org/doc/stable/reference/generated/numpy.ndenumerate.html), which is an index iterator;

- using [numba](https://numba.readthedocs.io/en/stable/user/parallel.html#numba-parallel), as a method for automatically parallelizing the code;

- using a custom [Cython](https://docs.cython.org/en/latest/src/userguide/numpy_tutorial.html) module;

- and using a custom Fortran module, compiled with [f2py](https://numpy.org/doc/stable/f2py/).

The functions and test codes are defined in the file [polymul_tests.py](https://github.com/rafael-ferro/polymul_tests/blob/main/polymul_tests.py), the Cython module is in the file [cpython_polymul.pyx](https://github.com/rafael-ferro/polymul_tests/blob/main/cpython_polymul.pyx), and the Fortran module is in the file [fortran_polymul.f90](https://github.com/rafael-ferro/polymul_tests/blob/main/fortran_polymul.f90).

Below are described the steps to run this code and the results I obtained in my system.
In order to check the calculations, I compared my results with those from the function `numpy.polymul`.

## Steps to reproduce

The Python modules required for this program are listed in [requirements.txt](https://github.com/rafael-ferro/polymul_tests/blob/main/requirements.txt). Besides, it is necessary to have a Fortran compiler installed. In Ubuntu-based systems, the _gfortran_ compiler can be easily installed with the command `sudo apt-get install gfortran`.

- Clone this repository: `git clone https://github.com/rafael-ferro/polymul_tests.git`
- Create a virtual environment and install the Python modules: `pip install -r requirements.txt`
- Compile the Cython code: `python setup.py build_ext --inplace`
- Compile the Fortran code: `python -m numpy.f2py -c fortran_polymul.f90 -m fortran_polymul`
- Run: `python polymul_tests.py`

## Results

These are the results I obtained in my system with 1000 elements in each array:

    numpy result:
    [0.06933971 0.37897217 0.9776624  ... 0.12357965 0.29359143 0.12585706]


    Using function naive_polymul(p1, p2)

    Elapsed time = 0.41352176666259766

    [0.06933971 0.37897217 0.9776624  ... 0.12357965 0.29359143 0.12585706]
    numpy.polymul check: [ True  True  True ...  True  True  True]


    Using function ndenumerate_polymul(p1, p2)

    Elapsed time = 0.6266388893127441

    [0.06933971 0.37897217 0.9776624  ... 0.12357965 0.29359143 0.12585706]
    numpy.polymul check: [ True  True  True ...  True  True  True]


    Using function numba_polymul(p1, p2)

    Elapsed time (with compilation) = 0.4556293487548828

    Elapsed time (after compilation) = 0.0006182193756103516

    [0.06933971 0.37897217 0.9776624  ... 0.12357965 0.29359143 0.12585706]
    numpy.polymul check: [ True  True  True ...  True  True  True]


    Using function cpython_polymul with custom cpython code

    Elapsed time = 0.0015795230865478516

    [0.06933971 0.37897217 0.9776624  ... 0.12357965 0.29359143 0.12585706]
    numpy.polymul check: [ True  True  True ...  True  True  True]


    Using function fortran_polymul with custom fortran code

    Elapsed time = 0.00023603439331054688

    [0.06933971 0.37897217 0.9776624  ... 0.12357965 0.29359143 0.12585706]
    numpy.polymul check: [ True  True  True ...  True  True  True]
