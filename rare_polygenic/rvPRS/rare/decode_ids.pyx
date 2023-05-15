# cython: language_level=3, boundscheck=False, emit_linenums=True

from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdint cimport int32_t

cdef extern from 'convert_ids.h' namespace 'rvPRS':
    vector[int32_t] from_bytes_cpp(string& byte_ids)
    string to_bytes_cpp(vector[int32_t]& ids)

def to_bytes(nums, length=3):
    ''' convert numbers to sequence of 3-byte encoded values
    
    Converts ints to byte sequences  e.g. [1000, 2000] ->  b'\xe8\x03\x00\xd0\x07\x00'
    where \xe8\x03\x00 is 1000 encoded in 3-bytes and \xd0\x07\x00 is 2000
    encoded in 3-bytes. This is requires less storage, particularly compared to
    storing the ints as comma-separated string of int values e.g. 
    '1000000,2000000' takes 15 bytes as string, but only 6 bytes when converted 
    to byte sequence. The sample ID lists are the biggest portion of the exome
    database, and converting them to bytes shrank the db by ~2.6X. This made 
    subsequent loading from the db much quicker, particularly when the db was 
    being queried by many processes from a single node.
    
    The byte-sequences can be decoded to lists of ints with from_bytes().
    '''
    if length == 3:
        return to_bytes_cpp(nums)
    return b''.join(x.to_bytes(length, 'little', signed=True) for x in nums)

def from_bytes(byte_ids, bit_len=3):
    ''' convert byte sequence to list of ints from 24-bit encoded values
    
    Decode byte-sequences to int lists e.g. b'\xe8\x03\x00\xd0\x07\x00' -> [1000, 2000]
    \xe8\x03\x00 is 3-byte encoded 1000, and \xd0\x07\x00 is 3-byte encoded 2000
    '''
    if bit_len == 3:
        return from_bytes_cpp(byte_ids)
    
    length = len(byte_ids)
    return [int.from_bytes(byte_ids[i:i+bit_len], byteorder='little', signed=True) for i in range(0, length, bit_len)]
