import struct
import numpy as np
import pandas as pd


def int_to_float(n):
    """
    convert from an 4-byte integer to 4-byte floating point number
    :param n: 4-byte integer
    :return: 4-byte floating point number
    """
    return struct.unpack('@f', struct.pack('@I', n))[0]


def float_to_int(f):
    """
    convert from an 4-byte floating point number to 4-byte integer
    :param f: 4-byte floating point
    :return: 4-byte integer
    """
    return struct.unpack('@I', struct.pack('@f', f))[0]


def float_to_bin(x):
    """
    convert from an 4-byte floating point number to binary string
    :param x: 4-byte floating point number
    :return: binary string
    """
    return bin(float_to_int(x))


def hamming_weight(f):
    """
    compute the Hamming weight og a 4-byte floating point number
    :param f: 4-byte floating point number
    :return: the Hamming weight
    """
    return float_to_bin(f).count('1')
