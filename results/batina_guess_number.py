import numpy as np
import pandas as pd
import struct
from utils import *

import numpy as np
import pandas as pd

max_mantissa_nbits = 23


def build_guess_values(component='mantissa', numbers=None, mantissa_nbits=10, guess_range=None):
    """
    build the list of guess values which is used to evaluate the weight value
    :param component: IEEE 754 component name, it must be "mentissa", "exponent", "sign"
    :param numbers: the list of numbers of the previous state
    :param mantissa_nbits: number of mantissa bits to be recovered
    :param guess_range: the guess range
    :return: the list of guess numbers
    """
    if component == 'mantissa':
        # set the exponent 1
        e = (0x7f << max_mantissa_nbits)
        guess_numbers = np.vectorize(int_to_float)(
            np.left_shift(np.arange(0, 1 << mantissa_nbits), max_mantissa_nbits - mantissa_nbits) | e)
    elif component == 'exponent':
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
    elif component == 'sign':
        y = np.concatenate((np.asarray(numbers), -np.asarray(numbers)))
        if None is guess_range:
            guess_numbers = y
        else:
            guess_numbers = y[(guess_range[0] <= y) & (y <= guess_range[1])]
    else:
        raise ValueError('the component is not supported')
    return guess_numbers


def build_input_values(component='mantissa',  mantissa_nbits=10):
    """
    build the list of input values which is used to evaluate the weight value
    :param component: IEEE 754 component name, it must be "mentissa", "exponent", "sign"
    :param mantissa_nbits: number of mantissa bits to be recovered
    :return: the list of input numbers
    """
    if component == 'mantissa':
        retval = np.vectorize(int_to_float)(
            np.left_shift(np.arange(0, 2 << mantissa_nbits), max_mantissa_nbits - mantissa_nbits))
    elif component == 'exponent':
        retval = np.vectorize(int_to_float)(np.left_shift(np.arange(0, 2 << 8), max_mantissa_nbits))
    elif component == 'sign':
        retval = np.random.uniform(-1.0, 1.0, 1000)
    else:
        raise ValueError('the component is not supported')
    return retval


def compute_corr_numbers(weight_hw, known_inputs, guess_numbers):
    """
    compute the HW correlations of the weight_hw and the HW of the results of the multiplication
    between known_inputs and guess_numbers.
    :param weight_hw: hamming weight of the known_inputs with the secret value
    :param known_inputs: known input values
    :param guess_numbers: guess numbers
    :return: Pearson correlation of the hamming weights
    """
    hw = pd.DataFrame(columns=guess_numbers,
                      data=np.vectorize(hamming_weight)(known_inputs.reshape(-1, 1) * guess_numbers))
    return hw.corrwith(pd.Series(weight_hw), method='pearson')


def batina_recover_weight(secret_number, guess_range, mantissa_nbits=10, max_number_of_best_candidates=10, noise=None,):
    """
    recover the weight value (secret_number)
    :param secret_number:
    :param guess_range:
    :param mantissa_nbits:
    :param max_number_of_best_candidates:
    :param noise: a tuple of (add_noise_function, signal_to_noise, frequency). The prototype of add_noise_function is,  func(signal, signal_to_noise, frequency)
    :return: the 10 values which have highest HW correlations
    """
    if noise is not None:
        add_noise_function, signal_to_noise, frequency = noise

    # step 1: guess the mantissa 10 bits
    known_inputs = build_input_values(mantissa_nbits=mantissa_nbits, component='mantissa')
    weight_hw = np.vectorize(hamming_weight)(known_inputs * secret_number)
    if noise is not None:
        weight_hw = add_noise_function(weight_hw, signal_to_noise, frequency)
    guess_numbers = build_guess_values(component='mantissa', mantissa_nbits=mantissa_nbits, guess_range=guess_range)
    mantisa_corr = compute_corr_numbers(weight_hw, known_inputs, guess_numbers)

    # step 2: guess the exponent 8 bits
    known_inputs = build_input_values(component='exponent')
    weight_hw = np.vectorize(hamming_weight)(known_inputs * secret_number)
    if noise is not None:
        weight_hw = add_noise_function(weight_hw, signal_to_noise, frequency)
    guess_numbers = build_guess_values(component='exponent',
                                       numbers=mantisa_corr
                                       .sort_values(ascending=False)
                                       .index[:max_number_of_best_candidates],
                                       guess_range=guess_range)
    mantisa_exp_corr = compute_corr_numbers(weight_hw, known_inputs, guess_numbers)

    # step 3: guess the sign 1 bit
    guess_numbers = build_guess_values(component='sign', numbers=mantisa_exp_corr.sort_values(ascending=False).index[
                                                                  :max_number_of_best_candidates],
                                       guess_range=guess_range)
    known_inputs = build_input_values(component='sign')
    weight_hw = np.vectorize(hamming_weight)(known_inputs * secret_number)
    if noise is not None:
        weight_hw = add_noise_function(weight_hw, signal_to_noise, frequency)
    full_number_corr = compute_corr_numbers(weight_hw, known_inputs, guess_numbers)
    return full_number_corr.sort_values(ascending=False).iloc[:max_number_of_best_candidates]
