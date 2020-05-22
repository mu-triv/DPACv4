
import struct
import numpy as np
import pandas as pd

# define Hamming Weight lambda function
hamming_weight = lambda x: bin(struct.unpack('@I', struct.pack('@f', x))[0]).count('1')

def compute_corr(secret_hw, guess_range, precision, known_inputs):
    """
    The function computes the Pearson correlations of secret_hw and the Hamming weights of the multiplication
    of the numbers in the guess_range with the known_inputs.
    :param secret_hw: the result of the the multiplication of the secret number with the known_inputs.
    :param guess_range: the range where it searches the secret number
    :param precision: prescision
    :param known_inputs: the known input values which is the random numbers
    :return: the pandas series of the Pearson correlations
    """
    low, high = guess_range
    guess_val_size = int(max(min((high - low) / precision * 1e2, 200), 1))
    guess_val = np.arange(low, high, (high - low) / (guess_val_size - 1e-5))
    # print('guess_val size =', len(guess_val))
    hw = pd.DataFrame(columns=guess_val,
                        data=np.vectorize(hamming_weight)(known_inputs.reshape(-1, 1) * guess_val))
    return hw.corrwith(secret_hw, method='pearson')


def guess_number_range(secret_hw, guess_range, prescision, known_inputs):
    """
    The function computes the range in which the secret_number could be in
    :param secret_hw: the result of the the multiplication of the secret number with the known_inputs.
    :param guess_range: the range where it searches the secret number
    :param prescision: the precision of the guess range
    :param known_inputs: the known input values which is the random numbers
    :return: the range of the value
    """
    best_corr = None
    low, high = guess_range
    while (high - low) > prescision:
        range_middle_value = (high + low) / 2.0
        # print('***guess_range =', (low, high), '(%f)' % range_middle_value)

        #
        # split up the guess_range into 2 subranges, then calculate the correlations of
        # hamming weights for each subrange.
        low_sub_range, high_sub_range = ((low, range_middle_value), (range_middle_value, high))
        low_corr = compute_corr(secret_hw, low_sub_range, prescision, known_inputs)
        high_corr = compute_corr(secret_hw, high_sub_range, prescision, known_inputs)
        #
        # concatenate best_corr which was computed in the previous loop
        if None is not best_corr:
            low_corr = pd.concat([low_corr, best_corr[best_corr.index <= range_middle_value]])
            high_corr = pd.concat([high_corr, best_corr[best_corr.index >= range_middle_value]])

        #
        # compare the highest scores of the subranges correlations
        # print('low_corr =', low_corr.max(), 'high_corr =', high_corr.max())
        if low_corr.max() > high_corr.max():
            low, high = low_sub_range
            best_corr = low_corr[low_corr == low_corr.max()]
        else:
            low, high = high_sub_range
            best_corr = high_corr[high_corr == high_corr.max()]

    # print('final guess_range =', (low, high))
    return low, high, best_corr.max()

LOW_VALUE = 'low value'
HIGH_VALUE = 'high value'
CORRELATION = 'correlation'

def guess_number_range_multiple_inputs(secret_number, guess_range, prescision, known_input_set):
    index = [LOW_VALUE, HIGH_VALUE, CORRELATION]
    results = pd.DataFrame(index=index)
    for knowm_input_idx in known_input_set.index:
        known_inputs = np.asarray(known_input_set.loc[knowm_input_idx])
        secret_hw = pd.Series(np.vectorize(hamming_weight)(known_inputs * secret_number), name='secret_hw')
        low, high, corr = guess_number_range(secret_hw, guess_range, prescision, known_inputs)
        results = pd.concat([results, pd.Series([low, high, corr], index=index, name=knowm_input_idx)], axis=1)
    return results.T

