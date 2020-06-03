import struct

def int_to_float(n):
    return struct.unpack('@f', struct.pack('@I', n))[0]


def float_to_int(f):
    return struct.unpack('@I', struct.pack('@f', f))[0]


def float_to_bin(x):
    return bin(float_to_int(x))


def hamming_weight(f):
    return float_to_bin(f).count('1')