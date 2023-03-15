# cython: language_level=3, boundscheck=False, emit_linenums=True

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.math cimport isnan
from libc.stdint cimport uint32_t, uint64_t, int32_t, int64_t

import numpy as np

from rvPRS.rare.id_decoder import to_bytes

cdef class Variant:
    def __cinit__(self, varid, rsid, chrom, uint32_t pos, ref, alt, symbol, 
                 consequence, missense_pathogenicity, spliceai, af, ac, an, 
                 gnomad_af, topmed_af, _missing, _homs, _hets):
        varid = b'' if varid is None else varid
        rsid = b'' if rsid is None else rsid
        chrom = b'' if chrom is None else chrom
        pos = 0 if pos is None else pos
        symbol = b'' if symbol is None else symbol
        af = 0.0 if af is None else af
        ac = 0 if ac is None else ac
        an = 0 if an is None else an
        
        missense_pathogenicity = float('nan') if missense_pathogenicity is None else missense_pathogenicity
        spliceai = float('nan') if spliceai is None else spliceai
        gnomad_af = 0 if gnomad_af is None else gnomad_af
        topmed_af = 0 if topmed_af is None else topmed_af
        _missing = b'' if _missing is None else _missing
        _hets = b'' if _hets is None else _hets
        _homs = b'' if _homs is None else _homs
        self.thisptr = new CppVariant(varid.encode('utf8'), rsid.encode('utf8'), 
                                  chrom.encode('utf8'), pos, ref.encode('utf8'), 
                                  alt.encode('utf8'), symbol.encode('utf8'),
                                  consequence.encode('utf8'), 
                                  missense_pathogenicity, spliceai,
                                  af, ac, an, gnomad_af, topmed_af, _missing,
                                  _homs, _hets)
    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, value):
        setattr(self, key, value)
    @property
    def varid(self): 
        return self.thisptr.varid.decode('utf8')
    @property
    def rsid(self):
        return self.thisptr.rsid.decode('utf8')
    @property
    def chrom(self):
        return self.thisptr.chrom.decode('utf8')
    @property
    def pos(self): return self.thisptr.pos
    @property
    def ref(self):
        return self.thisptr.ref.decode('utf8')
    @property
    def alt(self):
        return self.thisptr.alt.decode('utf8')
    @property
    def symbol(self):
        return self.thisptr.symbol.decode('utf8')
    @property
    def consequence(self):
        return self.thisptr.consequence.decode('utf8')
    @property
    def primateai(self):
        tmp = self.thisptr.missense_pathogenicity
        return None if isnan(tmp) else float(tmp)
    @primateai.setter
    def primateai(self, value):
        self.thisptr.missense_pathogenicity = value
    @property
    def spliceai(self):
        tmp = self.thisptr.spliceai
        return None if isnan(tmp) else float(tmp)
    @property
    def af(self): return float(self.thisptr.af)
    @af.setter
    def af(self, value): self.thisptr.af = value
    @property
    def ac(self): return int(self.thisptr.ac)
    @ac.setter
    def ac(self, value): self.thisptr.ac = value
    @property
    def an(self): return int(self.thisptr.an)
    @property
    def gnomad_af(self): return self.thisptr.gnomad_af
    @property
    def topmed_af(self): return self.thisptr.topmed_af
    @property
    def missing(self): 
        return self.thisptr.missing()
    @property
    def hets(self): 
        return self.thisptr.hets()
    @property
    def homs(self):
        return self.thisptr.homs()
    @homs.setter
    def homs(self, values):
        self.thisptr._homs = to_bytes(values)
    @property
    def all_samples(self):
        cdef uint64_t length = (self.thisptr._hets.size() + self.thisptr._homs.size()) // 3
        if length == 0:
            return []
        return self.thisptr.all_samples()
    @property
    def sample_idx(self):
        cdef uint64_t length = self.thisptr.sample_idx.size()
        if length == 0:
            return []
        return self.thisptr.sample_idx
    @sample_idx.setter
    def sample_idx(self, vector[uint64_t] x):
        self.thisptr.set_sample_idx(x)
    def flip_alleles(self, vector[int32_t] samples):
        self.thisptr.flip_alleles(samples)
    def __gt__(self, Variant other):
        # compare via cpp, much quicker than python comparison
        cdef CppVariant * thisptr = self.thisptr
        cdef CppVariant * otherptr = other.thisptr
        cdef bool ret = thisptr.missense_pathogenicity > otherptr.missense_pathogenicity
        return ret
    def __lt__(self, Variant other):
        # compare via cpp, much quicker than python comparison
        cdef CppVariant * thisptr = self.thisptr
        cdef CppVariant * otherptr = other.thisptr
        cdef bool ret = thisptr.missense_pathogenicity < otherptr.missense_pathogenicity
        return ret
    def __eq__(self, other):
        return self.__hash__()
    def __hash__(self):
        return self.thisptr._hash
    def __dealloc__(self):
        if self.thisptr is not NULL:
            del self.thisptr
