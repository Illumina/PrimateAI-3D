#!python
#cython: language_level=3

from libc.math cimport exp

from scipy.special.cython_special cimport betaln

def sf_old(int k, int tot, int good, int N):
    ''' survival function for hypergeometric distribution
    '''
    cdef int start, end, bad, _k
    low_half = k < (N // 2)

    # if k is sufficiently low, just use 1 - cdf for speed
    start, end = k + 1, N + 1
    if low_half:
        start, end = 0, k + 1

    bad = tot - good
    # some parts do not change between iterations, so only compute once
    cdef double a = betaln(good + 1, 1)
    cdef double b = betaln(bad + 1, 1)
    cdef double c = betaln(tot - N + 1, N + 1)
    cdef double d = betaln(tot + 1, 1)

    cdef double res = 0
    for _k in range(start, end):
        res += exp(a + b + c - betaln(_k + 1, good - _k + 1) - betaln(N - _k + 1, bad - N + _k + 1) - d)

    if low_half:
        res = 1 - res

    # clip value between 0 and 1
    return min(max(res, 0.0), 1.0)


def sf(int k, int tot, int good, int N):
    ''' survival function for hypergeometric distribution
    '''
    cdef int start = k + 1
    cdef int end = N + 1
    cdef int bad = tot - good

    # some parts do not change between iterations, so only compute once
    cdef double constant = betaln(good + 1, 1) + betaln(bad + 1, 1) + betaln(tot - N + 1, N + 1) - betaln(tot + 1, 1)

    cdef double peak = constant - betaln(start + 1, good - start + 1) - betaln(N - start + 1, bad - N + start + 1)
    cdef double current
    cdef double res = 0
    for _k in range(start, end):
        current = constant - betaln(_k + 1, good - _k + 1) - betaln(N - _k + 1, bad - N + _k + 1)
        res += exp(current)
        peak = max(peak, current)
        if (peak - current) > 50:
            # stop if the current value is 22 -log10 units from the peak,
            # since later iterations won't affect the final result
            break

    # clip value between 0 and 1
    return min(max(res, 0.0), 1.0)


