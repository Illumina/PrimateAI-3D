# cython: language_level=3, boundscheck=False, emit_linenums=True

from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t, uint64_t, int32_t, int64_t

cdef extern from 'variant.h' namespace 'rvPRS':
    cdef cppclass CppVariant:
        # declare class constructor and methods
        CppVariant(string varid, string rsid, string chrom, uint32_t pos, string ref,
                string alt, string symbol, string consequence, 
                double missense_pathogenicity, double spliceai, double af,
                uint64_t ac, uint64_t an, double gnomad_af, double topmed_af,
                string _missing, string _homs, string _hets) except +
        CppVariant() except +
        vector[int32_t] missing() except +
        vector[int32_t] homs() except +
        vector[int32_t] hets() except +
        vector[int32_t] all_samples() except +
        void set_sample_idx(vector[uint64_t] &indices) except +
        void flip_alleles(vector[int32_t] &samples) except +
        
        # declare public attributes
        string varid, rsid, chrom, ref, alt, symbol, consequence
        uint32_t pos
        uint64_t ac, an
        double missense_pathogenicity, spliceai, af, gnomad_af, topmed_af
        string _missing, _homs, _hets
        vector[uint64_t] sample_idx
        int64_t _hash

cdef class Variant:
    cdef CppVariant *thisptr
    cdef int _hash
