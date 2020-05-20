import struct
import numpy as np
import pandas as pd

# define Hamming Weight lambda function
hamming_weight = lambda x: bin(struct.unpack('@I', struct.pack('@f', x))[0]).count('1')

def compute_corr(secret_hw, guess_range, number_tests, known_inputs):
    """
    The function computes the Pearson correlations of secret_hw and the Hamming weights of the multiplication
    of the numbers in the guess_range with the known_inputs.
    :param secret_hw: the result of the the multiplication of the secret number with the known_inputs.
    :param guess_range: the range where it searches the secret number
    :param number_tests: the number of tests that are realized
    :param known_inputs: the known input values which is the random numbers
    :return: the pandas series of the Pearson correlations
    """
    low, high = guess_range
    guess_val = np.arange(low, high, (high - low) / (number_tests - 1.0 + 1e-5))
    hw = pd.DataFrame(columns=guess_val,
                        data=np.vectorize(hamming_weight)(known_inputs.reshape(-1, 1) * guess_val))
    return hw.corrwith(secret_hw, method='pearson')


def guess_number_range(secret_number, guess_range, prescision, number_tests, known_inputs):
    """
    The function computes the range in which the secret_number could be in
    :param secret_number: the secret number
    :param guess_range: the range where it searches the secret number
    :param prescision: the precision of the guess range
    :param number_tests: the number of tests that are realized
    :param known_inputs: the known input values which is the random numbers
    :return: the range of the value
    """
    secret_hw = pd.Series(np.vectorize(hamming_weight)(known_inputs * secret_number), name='secret_hw')
    best_corr = None
    low, high = guess_range
    while (high - low) > prescision:
        range_middle_value = (high + low) / 2.0
        print('***guess_range =', (low, high), '(%f)' % range_middle_value)
        low_sub_range, high_sub_range = ((low, range_middle_value), (range_middle_value, high))
        low_corr = compute_corr(secret_hw, low_sub_range, number_tests, known_inputs)
        high_corr = compute_corr(secret_hw, high_sub_range, number_tests, known_inputs)
        if None is not best_corr:
            low_corr = pd.concat([low_corr, best_corr[best_corr.index <= range_middle_value]])
            high_corr = pd.concat([high_corr, best_corr[best_corr.index >= range_middle_value]])

        print('low_corr =', low_corr.max(), 'high_corr =', high_corr.max())
        if low_corr.max() > high_corr.max():
            low, high = low_sub_range
            best_corr = low_corr[low_corr == low_corr.max()]
        else:
            low, high = high_sub_range
            best_corr = high_corr[high_corr == high_corr.max()]

    print('final guess_range =', (low, high))
    return (low, high)
