import struct
import numpy as np
import pandas as pd

import sys
sys.path.append("NN-Recovery")

# define Hamming Weight lambda function
from pybatina.utils import *


LOW_VALUE = 'low value'
HIGH_VALUE = 'high value'
CORRELATION = 'correlation'
INPUT_ID = 'input id'
HIGH_RANGE = 'high search range'
LOW_RANGE = 'low search range'


def generate_known_inputs(exps=[-2, 0, 2], nsize=5000):
    """
    generate input uniform random numbers. The number are split up
    in different exponential buckets.
    :param exps: exponent range
    :param nsize: total numbers of input numbers
    :return: input uniform random numbers
    """
    try:
        return pd.DataFrame(data=[np.random.uniform(-10.0 ** i, 10.0 ** i, nsize) for i in exps], index=exps)
    except TypeError:
        return pd.Series(np.random.uniform(-10.0 ** exps, 10.0 ** exps, nsize))


def compute_corr(secret_hw, guess_range, known_inputs, number_values=200):
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
    guess_val = np.random.uniform(low, high, number_values)
    hw = pd.DataFrame(columns=guess_val, data=np.vectorize(hamming_weight)(np.asarray(known_inputs).reshape(-1, 1) * guess_val))
    return hw.corrwith(pd.Series(secret_hw), method='pearson')


def guess_number_range(secret_hw, guess_range, precision, known_inputs, number_values=200):
    """
    The function computes the range in which the secret_number could be in
    :param secret_hw: the result of the the multiplication of the secret number with the known_inputs.
    :param guess_range: the range where it searches the secret number
    :param precision: the precision of the guess range
    :param known_inputs: the known input values which is the random numbers
    :return: the range of the value
    """
    print('guess_number_range')
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
        low_corr = compute_corr(secret_hw, low_sub_range, known_inputs, number_values)
        high_corr = compute_corr(secret_hw, high_sub_range, known_inputs, number_values)
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
            best_corr = low_corr
        else:
            low, high = high_sub_range
            best_corr = high_corr

    # print('final guess_range =', (low, high))
    return low, high, best_corr.max()


def filter_duplicated_ranges(candidates, precision):
    """
    :param candidates:
    :param precision:
    :return:
    """
    print('filter_duplicated_ranges')
    s = pd.Series((candidates.loc[LOW_VALUE] + candidates.loc[HIGH_VALUE])/2)
    s = (s + precision*5e-1).round(decimals=int(-np.log10(precision)))
    return [candidates[s[s == v].index].loc[CORRELATION].idxmax() for v in s.unique()]


def get_range_candidates(secret_hws, guess_range, known_input_set, precision=1e-6, number_values=200):
    """
    The function is an extension of the function guess_number_range with the known_input_set.
    :param secret_hws: the secret hamming weights in DataFrame
    :param guess_range: the guess range
    :param known_input_set: a set of known inputs
    :param precision: precision
    :param number_values: the number of guess values
    :return: the Dataframe which contains all information...
    """
    print('get_range_candidates %s' % str(guess_range))
    low_range, high_range = guess_range
    assert(low_range < high_range)
    results = pd.DataFrame()
    if (high_range <= 0.0) or (low_range >= 0.0):
        for inputs_idx in known_input_set.index:
            print('inputs_idx=%d' % inputs_idx)
            known_inputs = known_input_set.loc[inputs_idx]
            l, h, c = guess_number_range(secret_hw=secret_hws.loc[inputs_idx],
                                         guess_range=(low_range, high_range),
                                         precision=precision,
                                         known_inputs=known_inputs,
                                         number_values=number_values)
            s = pd.Series(data=[l, h, c, inputs_idx], index=[LOW_VALUE, HIGH_VALUE, CORRELATION, INPUT_ID])
            results = pd.concat([results, s], axis=1, ignore_index=True)
    else:
        neg_df = get_range_candidates(secret_hws=secret_hws, guess_range=(low_range, 0.0),
                                      known_input_set=known_input_set, precision=precision,
                                      number_values=number_values)
        pos_df = get_range_candidates(secret_hws=secret_hws,
                                      guess_range=(0.0, high_range),
                                      known_input_set=known_input_set,
                                      precision=precision,
                                      number_values=number_values)
        results = pd.concat([results, neg_df, pos_df], axis=1, ignore_index=True)
    # get rid of overlap candidates
    non_duplicated_index = filter_duplicated_ranges(candidates=results, precision=precision)
    return results[non_duplicated_index].T.reset_index(drop=True).T


def compute_ranges_corr(secret_hws, range_df, known_input_set, number_values):
    corr_df = pd.DataFrame(columns=known_input_set.index, index=range_df.columns)
    for input_idx in known_input_set.index:
        corrs = pd.Series(index=range_df.columns)
        for col in range_df.columns:
            c = range_df[col]
            corr = compute_corr(secret_hw=secret_hws.iloc[input_idx],
                                guess_range=(c[LOW_VALUE], c[HIGH_VALUE]),
                                known_inputs=known_input_set.iloc[input_idx],
                                number_values=number_values)
            corrs[col] = corr.max()
        corr_df[input_idx] = corrs
    return corr_df


def dismiss_bad_ranges(secret_hws, range_df, known_input_set, number_values):
    best_corr_df = compute_ranges_corr(secret_hws, range_df, known_input_set, number_values=number_values)
    series_max = best_corr_df.apply(lambda x: x >= x.max(), axis=0).apply(lambda x: x.all(), axis=1)
    retval = None
    if series_max.any():
        retval = range_df.T[series_max].iloc[0]
    return retval


def get_subranges(guess_range, precision):
    """
    First, separate the guess range in two parts: positive range and negative range
    extract the guess range in smaller and smaller guess ranges.
    For example: when guess_range=[]
    :param guess_range:
    :param precision:
    :return:
    """
    low_range, high_range = guess_range
    assert(low_range < high_range)
    if low_range < 0 and high_range > 0:
        low_subranges = get_subranges(guess_range=(low_range, 0.0), precision=precision)
        high_subranges = get_subranges(guess_range=(0.0, high_range), precision=precision)
        retval = low_subranges + high_subranges
    elif high_range <= 0:
        neg_subrange = get_subranges(guess_range=(-high_range, -low_range), precision=precision)
        retval = [(-h, -l) for (l, h) in neg_subrange]
    else:
        assert(low_range >= 0)
        retval = []
        while ((high_range - low_range) > precision) and (low_range < high_range ):
            retval.append((low_range, high_range))
            high_range = 10 ** np.ceil(np.log10(high_range / 10.0))
    return retval


def advanced_guess_number_range(secret_number, guess_range, precision, known_input_size=1000, number_values=200):
    results = pd.DataFrame()
    for (low_range, high_range) in get_subranges(guess_range, precision):
        exponent = np.floor(-np.log10(np.max(np.abs(np.asarray([low_range, high_range])))))
        for e in [exponent-1,exponent,exponent+1]:
            known_inputs = np.random.uniform(-10 ** e, 10 ** e, known_input_size)
            secret_hw = np.vectorize(hamming_weight)(known_inputs * secret_number)
            l, h, c = guess_number_range(secret_hw=secret_hw, known_inputs=known_inputs,
                                      guess_range=(low_range, high_range),
                                         number_values=number_values,
                                      precision=min(precision, (precision * (10 ** (-exponent)))))
            s = pd.Series(data=[l, h, c, e, low_range, high_range],
                           index=[LOW_VALUE, HIGH_VALUE, CORRELATION, INPUT_ID, LOW_RANGE, HIGH_RANGE])
            print(s)
            results = pd.concat([results, s], axis=1, ignore_index=True)
    return results.T.reset_index(drop=True)

