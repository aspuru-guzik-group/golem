#cython: language_level=3

cdef class BaseDist(object):
    cdef double x
    cdef double loc
    cpdef double pdf(self, x, loc)
    cpdef double cdf(self, double x, double loc)