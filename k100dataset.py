import torch.utils.data.dataset as dataset
import os
import pandas as pd


class K100Dataset(dataset.Dataset):
    def __init__(self, prediction_horizon, data_source='Train data CT03') -> None:
        super().__init__()

        self.src = data_source

        self._init_data_src()

        self.prediction_horizon = prediction_horizon
        self.ny = 1

    def _init_data_src(self):
        def _feather_to_df(path):
            return pd.read_feather(path)

        def _df_to_data(df):
            y_cols = ['o2pp']
            u_cols = ['o2duty', 'apduty']

            # y = list(zip(*(df[c] for c in y_cols)))
            # u = list(zip(*(df[c] for c in u_cols)))

            z = list(zip(*(df[c] for c in y_cols + u_cols)))

            x, target = z[:-self.prediction_horizon], z[-self.prediction_horizon:]
            target = list(zip(*target))[0]

            return z[:], target

        # def _p_to_d(path):
        #     return _df_to_data(_feather_to_tensor(path))

        dfs = [_feather_to_df(f'{self.src}/{fn}')
               for fn in os.listdir(self.src) if '.feather' in fn]
        self.data = [_df_to_data(df) for df in dfs if len(df) >= 2*self.prediction_horizon]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # return super().__getitem__(index)

        return self.data[index],
