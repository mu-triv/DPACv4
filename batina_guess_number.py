import numpy as np
import pandas as pd
import struct
from utils import *

import numpy as np
import pandas as pd

max_mantissa_nbits = 23


def build_guess_values(value_type='mantissa', numbers=None, mantissa_nbits=10, guess_range=None):
    if value_type == 'mantissa':
        # set the exponent 1
        e = (0x7f << max_mantissa_nbits)
        guess_numbers = np.vectorize(int_to_float)(
            np.left_shift(np.arange(0, 1 << mantissa_nbits), max_mantissa_nbits - mantissa_nbits) | e)
    elif value_type == 'exponent':
        # remove the exponent bits in mantissa
        m = np.vectorize(lambda x: x & ~(0xff << max_mantissa_nbits))(np.vectorize(float_to_int)(numbers))
        # set all possible exponent bits
        e = np.left_shift(np.arange(0, 1 << 8), max_mantissa_nbits)
        y = np.vectorize(int_to_float)(m | e[:, np.newaxis]).reshape(-1)
        # because we do not take in account the sign, the value is in positive
        # so we need to change the low and high of the guess range
        if None is guess_range:
            guess_numbers = y
        else:
            hi_range = max(np.abs(guess_range))
            lo_range = max(np.min(guess_range), 0.0)
            guess_numbers = y[(lo_range <= y) & (y <= hi_range)]
    elif value_type == 'sign':
        y = np.concatenate((np.asarray(numbers), -np.asarray(numbers)))
        if None is guess_range:
            guess_numbers = y
        else:
            guess_numbers = y[(guess_range[0] <= y) & (y <= guess_range[1])]
    else:
        raise ValueError('the value_type is not supported')
    return guess_numbers


def compute_corr_numbers(weight_hw, known_inputs, guess_numbers, processing_func=None):
    if None is processing_func:
        processing_func = lambda x: float_to_bin(x).count('1')
    hw = pd.DataFrame(columns=guess_numbers,
                      data=np.vectorize(processing_func)(np.asarray(known_inputs).reshape(-1, 1) * guess_numbers))
    return hw.corrwith(pd.Series(weight_hw), method='pearson')


def batina_recover_weight(secret_number, guess_range, mantissa_nbits=10, max_number_of_best_candidates=10):
    # step 1: guess the mantissa 10 bits
    total_mantissa_nbits = 23
    known_inputs = np.vectorize(int_to_float)(
        np.left_shift(np.arange(0, 2 << mantissa_nbits), total_mantissa_nbits - mantissa_nbits))
    weight_hw = np.vectorize(hamming_weight)(known_inputs * secret_number)
    guess_numbers = build_guess_values(value_type='mantissa', mantissa_nbits=mantissa_nbits, guess_range=guess_range)
    mantissa_processing_func = lambda x: bin(float_to_int(x) & 0x7fffff).count('1')
    mantisa_corr = compute_corr_numbers(weight_hw, known_inputs, guess_numbers,
                                        processing_func=mantissa_processing_func)
    # step 2: guess the exponent 8 bits
    known_inputs = np.vectorize(int_to_float)(np.left_shift(np.arange(0, 2 << 8), total_mantissa_nbits))
    weight_hw = np.vectorize(hamming_weight)(known_inputs * secret_number)
    guess_numbers = build_guess_values(value_type='exponent', numbers=mantisa_corr.sort_values(ascending=False).index[
                                                                      :max_number_of_best_candidates],
                                       guess_range=guess_range)
    mantisa_exp_corr = compute_corr_numbers(weight_hw, known_inputs, guess_numbers)
    # step 3: guess the sign 1 bit
    guess_numbers = build_guess_values(value_type='sign', numbers=mantisa_exp_corr.sort_values(ascending=False).index[
                                                                  :max_number_of_best_candidates],
                                       guess_range=guess_range)
    known_inputs = np.random.uniform(-1.0, 1.0, 100)
    weight_hw = np.vectorize(hamming_weight)(known_inputs * secret_number)
    full_number_corr = compute_corr_numbers(weight_hw, known_inputs, guess_numbers)
    return full_number_corr.sort_values(ascending=False).iloc[:10]
