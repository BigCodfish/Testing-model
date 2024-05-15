from datautils.data_sampler import DataSampler
from model import train
from datautils import data_transformer as dt
from model.layers import NetConfig, LayerConfig
from utils import painter, utils

def run_train():
    data, discrete_column = dt.read_csv('adult.csv', 'adult.json')
    data, output_info_list, output_dim = dt.transform(data, discrete_column)
    # data = dt.mask_data(raw_data=data, info_list=output_info_list, mask_rate=0.7)
    data_sampler = DataSampler(data, output_info_list)
    cond_dim = data_sampler.n_categories
    print(f"模型输入维数：{output_dim * 2}")
    layers = [LayerConfig(output_dim * 2, output_dim, 'Linear', 'relu'),
              LayerConfig(output_dim, output_dim, 'Linear', 'relu'),
              LayerConfig(output_dim, output_dim, 'Linear', 'sigmoid')]

    # 逐层分段激活
    # config_g = NetConfig(type='MixDataNet', layers=layers, optim='Adam', loss='')
    # config_d = NetConfig(type='MixDataNet', layers=layers, optim='Adam', loss='wasserstein')

    # 不逐层分段激活
    config_g = NetConfig(type='MLP', layers=layers, optim='Adam', loss='GAIN')
    config_d = NetConfig(type='MLP', layers=layers, optim='Adam', loss='wasserstein')

    loss_g, loss_d, w_distance, loss_test, acc = train.train(data, config_g, config_d, output_info_list, data_sampler,
                                                     batch_size=128, num_epochs=500)
    combined = list(map(list, zip(loss_g, loss_d, loss_test, w_distance, acc)))
    utils.save_result('1', combined)
    # painter.draw_sub([loss_g, loss_d, w_distance, loss_test, acc])

# run_train()
