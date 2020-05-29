import numpy as np
import pandas as pd
import struct

import numpy as np
import pandas as pd
import struct

int_to_float = lambda n: struct.unpack('@f', struct.pack('@I', n))[0]
float_to_int = lambda f: struct.unpack('@I', struct.pack('@f', f))[0]
float_to_bin = lambda x: bin(float_to_int(x))

# Note: From my view, this is the right thing to do for generating the guess numbers.
# Anyhow this does not work properly, since the conversion int_to_float and float_to_int
# have an error. This causes the guess wrong.
#
# For this issue, please check https://stackoverflow.com/questions/62067301/python-struct-to-convert-int-to-ieee-float
#
# def build_guess_values(value_type='mantissa', numbers=None, mantissa_nbits=10, guess_range=None):
#     max_mantissa_nbits = 23
#     if value_type == 'mantissa':
#         guess_numbers = np.vectorize(int_to_float)(np.left_shift(np.arange(0, np.power(2, mantissa_nbits)), max_mantissa_nbits-mantissa_nbits))
#     elif value_type == 'exponent':
#         m = np.vectorize(float_to_int)(numbers)
#         e = np.left_shift(np.arange(1, np.power(2, 8)), max_mantissa_nbits)
#         y = np.vectorize(int_to_float)(m | e[:, np.newaxis]).reshape(-1)
#         # because we do not take in account the sign, the value is in positive
#         # so we need to change the low and high of the guess range
#         if None is guess_range:
#             guess_numbers = y
#         else:
#             hi_range = np.max(np.abs(guess_range))
#             lo_range = np.min(np.abs(guess_range))
#             guess_numbers = y[(lo_range <= y) & (y <= hi_range)]
#     elif value_type == 'sign':
#         y = np.concatenate((np.asarray(numbers), -np.asarray(numbers)))
#         if None is guess_range:
#             guess_numbers = y
#         else:
#             guess_numbers = y[(guess_range[0] <= y) & (y <= guess_range[1])]
#     else:
#         raise ValueError('the value_type is not supported')
#     return guess_numbers


max_mantissa_nbits = 16
two_powers = 1 / np.asarray([np.power(2.0, i) for i in np.arange(0, max_mantissa_nbits)])


def _compute_mantisa(n):
    """
    get the floating number from n.
    :param n:
    :return:
    """
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
        y = (np.asarray(numbers) * two_exponents.reshape(-1, 1)).reshape(-1)
        # because we do not take in account the sign, the value is in positive
        # so we need to change the low and high of the guess range
        hi_range = max(np.abs(guess_range))
        lo_range = max(np.min(guess_range), 0.0)
        print('lo_range = %f, hi_range=%f' % (lo_range, hi_range))
        guess_numbers = y[(lo_range <= y) & (y <= hi_range)]
    elif value_type == 'sign':
        y = np.concatenate((np.asarray(numbers), -np.asarray(numbers)))
        guess_numbers = y[(guess_range[0] <= y) & (y <= guess_range[1])]
    else:
        raise ValueError('the value_type is not supported')
    return guess_numbers


def compute_corr_numbers(weight_hw, known_inputs, guess_numbers):
    # define Hamming Weight lambda function
    hamming_weight = lambda x: float_to_bin(x).count('1')

    hw = pd.DataFrame(columns=guess_numbers,
                      data=np.vectorize(hamming_weight)(np.asarray(known_inputs).reshape(-1, 1) * guess_numbers))
    return hw.corrwith(pd.Series(weight_hw), method='pearson')


def batina_recover_weight(weight_hw, known_inputs, guess_range):
    max_number_of_best_cadidate = 10
    # step 1: guess the mantissa 10 bits
    guess_numbers = build_guess_values(value_type='mantissa', mantissa_nbits=10, guess_range=guess_range)
    mantisa_corr = compute_corr_numbers(weight_hw, known_inputs, guess_numbers)
    # step 2: guess the exponent 8 bits
    guess_numbers = build_guess_values(value_type='exponent', numbers=mantisa_corr.sort_values(ascending=False).index[:max_number_of_best_cadidate], guess_range=guess_range)
    mantisa_exp_corr = compute_corr_numbers(weight_hw, known_inputs, guess_numbers)
    # step 3: guess the sign 1 bit
    guess_numbers = build_guess_values(value_type='sign', numbers=mantisa_exp_corr.sort_values(ascending=False).index[:max_number_of_best_cadidate], guess_range=guess_range)
    full_number_corr = compute_corr_numbers(weight_hw, known_inputs, guess_numbers)
    return full_number_corr.sort_values(ascending=False).iloc[:10]

