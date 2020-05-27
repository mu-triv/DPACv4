import struct
import numpy as np
import pandas as pd

# define Hamming Weight lambda function
hamming_weight = lambda x: bin(struct.unpack('@I', struct.pack('@f', x))[0]).count('1')

def generate_known_inputs(exps=(-1, 3), nsize=3000):
    """
    generate input uniform random numbers. The number are split up
    in different exponential buckets.
    :param exps: exponent range
    :param nsize: total numbers of input numbers
    :return: input uniform random numbers
    """
    low, high = exps[0], (exps[1] + 1)
    subsize = int(nsize / (high - low))
    return pd.DataFrame(
        data=[np.random.uniform(-10.0 ** i, 10.0 ** i, subsize) for i in range(low, high)],
        index=range(low, high))

def compute_corr(secret_hw, guess_range, known_inputs):
    """
    The function computes the Pearson correlations of secret_hw and the Hamming weights of the multiplication
    of the numbers in the guess_range with the known_inputs.
    :param secret_hw: the result of the the multiplication of the secret number with the known_inputs.
    :param guess_range: the range where it searches the secret number
    :param known_inputs: the known input values which is the random numbers
    :return: the pandas series of the Pearson correlations
    """
    assert(len(secret_hw) == len(known_inputs))
    low, high = guess_range
    if low >= high:
        raise ValueError('range value invalid %s' % str(guess_range))
    guess_val = np.arange(low, high, (high - low) / 200)
    hw = pd.DataFrame(columns=guess_val, data=np.vectorize(hamming_weight)(np.asarray(known_inputs).reshape(-1, 1) * guess_val))
    return hw.corrwith(pd.Series(secret_hw), method='pearson')


def guess_number_range(secret_hw, guess_range, precision, known_inputs):
    """
    The function computes the range in which the secret_number could be in
    :param secret_hw: the result of the the multiplication of the secret number with the known_inputs.
    :param guess_range: the range where it searches the secret number
    :param precision: the precision of the guess range
    :param known_inputs: the known input values which is the random numbers
    :return: the range of the value
    """
    best_corr = None
    low, high = guess_range
    if (high - low) < precision:
        raise ValueError('range is too small %s' % str(guess_range))

    while (high - low) >= precision:
        range_middle_value = (high + low) / 2.0
        # print('***guess_range =', (low, high), '(%f)' % range_middle_value)

        #
        # split up the guess_range into 2 subranges, then calculate the correlations of
        # hamming weights for each subrange.
        low_sub_range, high_sub_range = ((low, range_middle_value), (range_middle_value, high))
        low_corr = compute_corr(secret_hw, low_sub_range, known_inputs)
        high_corr = compute_corr(secret_hw, high_sub_range, known_inputs)
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
INPUT_ID = 'input id'

def guess_number_positive_and_negative_range(secret_number, guess_range, precision, known_input_size=1000):
    index = [INPUT_ID, LOW_VALUE, HIGH_VALUE, CORRELATION]
    retval = pd.DataFrame()
    range_low, range_high = guess_range
    if 0 < range_high:
        exp = int(np.round(-np.log10(range_high), decimals=0))
        for i in [exp-1, exp, exp+1]:
            known_inputs = np.random.uniform(-10 ** i, 10 ** i, known_input_size)
            secret_hw = np.vectorize(hamming_weight)(known_inputs * secret_number)
            l, h, c = guess_number_range(secret_hw, (max(0.0, range_low), range_high), precision, known_inputs)
            retval = pd.concat([retval, pd.Series([i, l, h, c], index=index)], axis=1)
    if 0 > range_low:
        exp = int(np.round(-np.log10(np.abs(range_low)), decimals=0)) + 1
        for i in [exp-1, exp, exp+1]:
            known_inputs = np.random.uniform(-10 ** i, 10 ** i, known_input_size)
            secret_hw = np.vectorize(hamming_weight)(known_inputs * secret_number)
            l, h, c = guess_number_range(secret_hw, (range_low, min(0.0, range_high)), precision, known_inputs)
            retval = pd.concat([retval, pd.Series([i, l, h, c], index=index)], axis=1)
    return retval.T


def guess_number_range_subranges(secret_number, guess_range, precision, known_input_size=1000):
    results = pd.DataFrame()
    range_low, range_high = guess_range
    known_input_set = pd.DataFrame()
    try:
        while True:
            print('searching in the range (%f, %f)' % (range_low, range_high))
            results = pd.concat([results,
                                guess_number_positive_and_negative_range(secret_number, (range_low, range_high), precision, known_input_size)],
                                axis=0, ignore_index=True)
            range_low = max(range_low, range_low / 10.0)
            range_high = min(range_high, range_high / 10.0)
    except ValueError:
        # this is the range exception
        pass
    return results
