import numpy as np
import pandas as pd
import struct

max_mantissa_nbits = 16
two_powers = 1 / np.asarray([np.power(2.0, i) for i in np.arange(0, max_mantissa_nbits)])

def _compute_mantisa(n):
    retval = 0.0
    s = "{0:b}".format(n)
    l = len(s)
    for idx in range(l):
        if s[l-idx-1] == '1':
            retval = retval + two_powers[idx]
    return retval

def build_guess_values(value_type='mantissa', numbers=None, mantissa_nbits=10, guess_range=None):
    if value_type == 'mantissa':
        if mantissa_nbits > max_mantissa_nbits:
            raise ValueError('the mantissa_nbits value is not supported')
        guess_numbers = np.vectorize(_compute_mantisa)(np.arange(0, np.power(2, mantissa_nbits)))
    elif value_type == 'exponent':
        exponent_nbits = 8
        exponents = np.arange(0, np.power(2.0, exponent_nbits)) - 127
        two_exponents = np.vectorize(lambda x: np.power(2.0, x) if x >= 0 else 1.0/np.power(2.0, -x))(exponents)
        x = (np.asarray(numbers) * two_exponents.reshape(-1, 1)).reshape(-1)
        guess_numbers = x[(guess_range[0] <= x) & (x <= guess_range[1])]
    elif value_type == 'sign':
        y = np.concatenate((np.asarray(numbers), -np.asarray(numbers)))
        guess_numbers = y[(guess_range[0] <= y) & (y <= guess_range[1])]
    else:
        raise ValueError('the value_type is not supported')
    return guess_numbers

def compute_corr_numbers(secret_hw, known_inputs, guess_numbers):
    # define Hamming Weight lambda function
    hamming_weight = lambda x: bin(struct.unpack('@I', struct.pack('@f', x))[0]).count('1')

    hw = pd.DataFrame(columns=guess_numbers,
                        data=np.vectorize(hamming_weight)(known_inputs.reshape(-1, 1) * guess_numbers))
    return hw.corrwith(pd.Series(secret_hw), method='pearson')