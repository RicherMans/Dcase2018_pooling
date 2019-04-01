import numpy as np
from dcase_util.data import DecisionEncoder


def activity_detection(x, high_thres, low_thres, n_connect=1):
    high_locations = np.where(x > high_thres)[0]
    locations = x > low_thres
    dec_enc = DecisionEncoder()
    encoded_pairs = dec_enc.find_contiguous_regions(locations)

    filtered_list = list(filter(lambda pair: ((pair[0] <= high_locations) &
                                              (high_locations <= pair[1])).any(), encoded_pairs))

    filtered_list = connect_(filtered_list, n_connect)
    return filtered_list


def connect_(pairs, n=1):
    if len(pairs) == 0:
        return []
    start_, end_ = pairs[0]
    new_pairs = []
    for i, (next_item, cur_item) in enumerate(zip(pairs[1:], pairs[0:])):
        end_ = next_item[1]
        if next_item[0] - cur_item[1] <= n:
            pass
        else:
            new_pairs.append((start_, cur_item[1]))
            start_ = next_item[0]
    new_pairs.append((start_, end_))
    return new_pairs
