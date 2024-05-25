import numpy as np


class DataSampler:
    def __init__(self, data, output_info_list):
        self.n_data = len(data)

        def is_discrete(info):
            return len(info) == 1 and info[0].activation_fn == 'softmax'

        # 数据中离散数据列的数量
        self.n_discrete = sum(
            [1 for info in output_info_list if is_discrete(info)])
        # 所有离散列中类型的总和
        self.n_categories = sum([
            info[0].dim
            for info in output_info_list
            if is_discrete(info)
        ])
        max_category = max([
            info[0].dim
            for info in output_info_list
            if is_discrete(info)
        ], default=0)

        self._discrete_column_cond_st = np.zeros(self.n_discrete, dtype='int32')
        self._discrete_column_category_prob = np.zeros((self.n_discrete, max_category))

        st = 0
        current_id = 0
        current_cond_st = 0
        for column_info in output_info_list:
            if is_discrete(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim
                category_freq = np.sum(data[:, st:ed], axis=0)
                # if log_frequency:
                #     category_freq = np.log(category_freq + 1)
                category_prob = category_freq / np.sum(category_freq)
                self._discrete_column_category_prob[current_id, :span_info.dim] = category_prob
                self._discrete_column_cond_st[current_id] = current_cond_st
                # self._discrete_column_n_category[current_id] = span_info.dim
                current_cond_st += span_info.dim
                current_id += 1
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])

    def sample_cond(self, batch, data_mask=None):
        """Generate the conditional vector for training.
            condvec每次只选择一个离散列中的一个类别
        Returns:
            cond (batch x #categories):
                The conditional vector.
            mask (batch x #discrete columns):
                A one-hot vector indicating the selected discrete column.
            discrete column id (batch):
                Integer representation of mask.
            category_id_in_col (batch):
                Selected category in the selected discrete column.
        """
        if self.n_discrete == 0:
            return None
        # n_discrete_columns: 数据中数据类型为离散的列的数量
        # chosen_column_id: 随机选择的列的id
        chosen_column_id = np.random.choice(
            np.arange(self.n_discrete), batch)

        mask = np.zeros((batch, self.n_discrete), dtype='float32')
        mask[np.arange(batch), chosen_column_id] = 1
        cond = np.zeros((batch, self.n_categories), dtype='float32')
        category_id_in_col = self._random_choice_prob_index(chosen_column_id)
        category_id = (self._discrete_column_cond_st[chosen_column_id] + category_id_in_col)
        cond[np.arange(batch), category_id] = 1

        return cond, mask, chosen_column_id, category_id_in_col

    def _random_choice_prob_index(self, discrete_column_id):
        probs = self._discrete_column_category_prob[discrete_column_id]
        r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
        return (probs.cumsum(axis=1) > r).argmax(axis=1)