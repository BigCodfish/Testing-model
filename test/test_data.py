import numpy as np

from datautils.data_transformer import mask_data, SpanInfo


def _test_data_mask():
    data = np.ones([10, 10])
    info_list = [[SpanInfo(dim=2, activation_fn=''), SpanInfo(dim=1, activation_fn='')],
                 [SpanInfo(dim=3, activation_fn=''), SpanInfo(dim=4, activation_fn='')]]
    masked = mask_data(data, info_list, mask_rate=0.5)
    print(data)
    print(masked)
    t = data-masked
    print(np.count_nonzero(t == 0) / 100)

_test_data_mask()