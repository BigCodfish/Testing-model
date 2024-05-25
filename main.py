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
    print(f"模型输入维数：{output_dim * 2 + cond_dim}")
    layers = [LayerConfig(output_dim * 2 + cond_dim, output_dim, 'Linear', 'relu'),
              LayerConfig(output_dim, output_dim, 'Linear', 'relu'),
              LayerConfig(output_dim, output_dim, 'Linear', '')]
    layers_VAE = [LayerConfig(output_dim * 2 + cond_dim, output_dim, '', ''),
                  LayerConfig(100, output_dim, '', '')]

    # VAE
    config_g = NetConfig(type='VAE', layers=layers_VAE, optim='Adam', loss='VAE')
    config_d = NetConfig(type='MixDataNet', layers=layers, optim='Adam', loss='WD')

    # 逐层分段激活
    # config_g = NetConfig(type='MixDataNet', layers=layers, optim='Adam', loss='CTGAN')
    # config_d = NetConfig(type='MixDataNet', layers=layers, optim='Adam', loss='WD')

    # 不逐层分段激活
    # config_g = NetConfig(type='MLP', layers=layers, optim='Adam', loss='CTGAN')
    # config_d = NetConfig(type='MLP', layers=layers, optim='Adam', loss='WD')

    loss_g, loss_d, w_distance, loss_test, acc = train.train(data, config_g, config_d, output_info_list, data_sampler,
                                                     batch_size=128, num_epochs=1000, use_cond=True)
    # combined = list(map(list, zip(loss_g, loss_d, loss_test, w_distance, acc)))
    # utils.save_result('8', combined)
    painter.draw_sub([loss_g, loss_d, w_distance, loss_test, acc])

run_train()



def comparison_1_2():
    res_1 = utils.read_result('1')
    res_2 = utils.read_result('2')
    painter.draw([res_1[0], res_1[1]], ylabel='loss', ytags=['loss_g', 'loss_d'],title='loss in training 1')
    painter.draw([res_2[0], res_2[1]], ylabel='loss', ytags=['loss_g', 'loss_d'], title='loss in training 2')
    painter.draw([res_1[2], res_2[2]], ylabel='w_distance', ytags=['GAIN loss', 'WDist loss'], title='w_distance')
    painter.draw([res_1[3], res_2[3]], ylabel='loss test', ytags=['GAIN loss', 'WDist loss'], title='loss test')
    painter.draw([res_1[4], res_2[4]], ylabel='test acc', ytags=['GAIN loss', 'WDist loss'], title='test acc')

def comparison_1_3():
    res_2 = utils.read_result('1')
    res_3 = utils.read_result('3')
    painter.draw([res_2[0], res_2[1]], ylabel='loss', ytags=['loss_g', 'loss_d'], title='loss in training 1')
    painter.draw([res_3[0], res_3[1]], ylabel='loss', ytags=['loss_g', 'loss_d'], title='loss in training 3')
    painter.draw([res_2[2], res_3[2]], ylabel='w_distance', ytags=['no cond', 'with cond'], title='w_distance')
    painter.draw([res_2[3], res_3[3]], ylabel='loss test', ytags=['no cond', 'with cond'], title='loss test')
    painter.draw([res_2[4], res_3[4]], ylabel='test acc', ytags=['no cond', 'with cond'], title='test acc')

def comparison_3_4():
    res_3 = utils.read_result('3')
    res_4 = utils.read_result('4')
    painter.draw([res_3[0], res_3[1]], ylabel='loss', ytags=['loss_g', 'loss_d'], title='loss in training 3')
    painter.draw([res_4[0], res_4[1]], ylabel='loss', ytags=['loss_g', 'loss_d'], title='loss in training 4')
    painter.draw([res_3[2], res_4[2]], ylabel='w_distance', ytags=['GAIN loss_g', 'CTGAN loss_g'], title='w_distance')
    painter.draw([res_3[3], res_4[3]], ylabel='loss test', ytags=['GAIN loss_g', 'CTGAN loss_g'], title='loss test')
    painter.draw([res_3[4], res_4[4]], ylabel='test acc', ytags=['GAIN loss_g', 'CTGAN loss_g'], title='test acc')

def comparison_4_5_6():
    res_4 = utils.read_result('4')
    res_5 = utils.read_result('5')
    res_6 = utils.read_result('6')
    painter.draw([res_4[0], res_4[1]], ylabel='loss', ytags=['loss_g', 'loss_d'], title='loss in training 4')
    painter.draw([res_5[0], res_5[1]], ylabel='loss', ytags=['loss_g', 'loss_d'], title='loss in training 5')
    painter.draw([res_6[0], res_6[1]], ylabel='loss', ytags=['loss_g', 'loss_d'], title='loss in training 6')
    painter.draw([res_4[2], res_5[2], res_6[2]], ylabel='w_distance', ytags=['r-r-sigmoid', 'r-r-span', 'all span'], title='w_distance')
    painter.draw([res_4[3], res_5[3], res_6[3]], ylabel='loss test', ytags=['r-r-sigmoid', 'r-r-span', 'all span'], title='loss test')
    painter.draw([res_4[4], res_5[4], res_6[4]], ylabel='test acc', ytags=['r-r-sigmoid', 'r-r-span', 'all span'], title='test acc')

def comparison_6_7_8():
    res_6 = utils.read_result('6')
    res_7 = utils.read_result('7')
    res_8 = utils.read_result('8')
    painter.draw([res_6[0], res_6[1]], ylabel='loss', ytags=['loss_g', 'loss_d'], title='loss in training 6')
    painter.draw([res_7[0], res_7[1]], ylabel='loss', ytags=['loss_g', 'loss_d'], title='loss in training 7')
    painter.draw([res_8[0], res_8[1]], ylabel='loss', ytags=['loss_g', 'loss_d'], title='loss in training 8')
    painter.draw([res_6[2], res_7[2], res_8[2]], ylabel='w_distance', ytags=['CTGAN loss_g', 'Inform loss_g', 'GAIN loss_g'], title='w_distance')
    painter.draw([res_6[3], res_7[3], res_8[3]], ylabel='loss test', ytags=['CTGAN loss_g', 'Inform loss_g', 'GAIN loss_g'], title='loss test')
    painter.draw([res_6[4], res_7[4], res_8[4]], ylabel='test acc', ytags=['CTGAN loss_g', 'Inform loss_g', 'GAIN loss_g'], title='test acc')
