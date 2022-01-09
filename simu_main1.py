# pylint --max-line-length=400
from copy import deepcopy
import argparse
import time

from Model import Net
import torch
from torchvision import datasets
from sklearn.preprocessing import minmax_scale
from alg_katyusha import KATYUSHA
import numpy as np
import matplotlib.pyplot as plt


class ALG_Katyusha(object):
    pass


if __name__ == '__main__':
    tic = time.time()
    is_gpu = torch.cuda.is_available()
    idx_device = torch.device('cuda:0' if is_gpu else "cpu")


    ## Parameter setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, help='Learning Rate')
    parser.add_argument('--num_sample', default=int(1e4), help='数据样本数')
    parser.add_argument('--num_epoch', default=int(10), help='训练轮次')
    parser.add_argument('--path_saga', default='./save/SAGA', help='SAGA参数配置')
    parser.add_argument('--path_svrg', default='./save/SVRG', help='SVRG参数配置')
    parser.add_argument('--path_katyusha', default='./save/Katyusha', help='Katyusha参数配置')
    simu_para = vars( parser.parse_args() )
    simu_para['device'] = idx_device
    simu_para['rseed'] = 0
    print(simu_para)
    torch.manual_seed(simu_para['rseed'])  ##使用固定的随机数种子，有利于我们复现实验结果

    ## dataloader
    train_data = datasets.MNIST(root='./data', train=True, download=True)
    img, lbl = train_data.train_data, train_data.train_labels
    img = img.reshape(img.shape[0], -1)
    img = torch.from_numpy( minmax_scale(img) )
    dim_img = img.shape[1]
    dim_lbl = lbl.unique().shape[0]

    img = img[ range(simu_para['num_sample']) ].type(torch.float32)
    lbl = lbl[ range(simu_para['num_sample']) ]
    img = img.to(device=idx_device)  ## move to device (gpu or cpu)
    lbl = lbl.to(device=idx_device)  ## move to device (gpu or cpu)
    print(f"Feature info: {img.shape}, label info: {lbl.shape}")

    # 模型
    net = Net(dim_img, dim_lbl)
    net_saga = deepcopy(net).to(device=idx_device)
    net_katyusha = deepcopy(net).to(device=idx_device)
    net_svrg = deepcopy(net).to(device=idx_device)

    func_loss = torch.nn.CrossEntropyLoss().to(device=idx_device)
    func_loss_init = torch.nn.CrossEntropyLoss(reduction='none').to(device=idx_device)
    epoch = simu_para['num_epoch']
    lr = simu_para['lr']

    ###### KATYUSHA ALGORITHM #################################################
    print("Start KATYUSHA optimization")
    solver_katyusha = KATYUSHA(0, 0)
    saga_train_loss = solver_katyusha.train()

    # ###### SAGA ALGORITHM #################################################
    # print("Start SAGA optimization")
    # solver_saga = SAGA(img, lbl, net_saga, func_loss, func_loss_init, simu_para)
    # solver_saga.init_grad()
    # saga_train_loss = solver_saga.train()
    #
    #
    # ###### SARAH ALGORITHM #################################################
    # print("Start SARAH optimization")
    # solver_saga = SARAH(img, lbl, net_saga, func_loss, func_loss_init, simu_para)
    # solver_saga.init_grad()
    # saga_train_loss = solver_saga.train()

    # ###### SVRG ALGORITHM #############################################
    # print("Start SVRG optimization")
    # svrg_train_loss = ALG_SVRG.SVRG(net_svrg, func_loss, img, lbl, epoch, lr)
    #
    # ###### KATYUSHA ALGORITHM #############################################
    print("Start Katyusha optimization")
    sigma = 0.6
    L = 1000
    option = 1
    katyusha_train_loss = ALG_Katyusha.Katyusha(epoch, sigma, L, net_katyusha, func_loss, img, lbl, option)

    # # optimization result
    epoch = np.arange(epoch + 1)
    plt.plot(epoch, saga_train_loss, label='SAGA')
    plt.plot(epoch, katyusha_train_loss, label='Katyusha')
    #plt.plot(epoch, svrg_train_loss, label='SVRG')
    plt.title('Optimization Algorithm')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.legend()
    plt.show()

    toc = time.time()
    print('Consumed duration: {:.5f}....'.format(toc - tic))  ## 输出运行时间
