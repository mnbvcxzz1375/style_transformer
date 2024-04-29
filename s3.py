import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
from PIL import Image
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import shutil
from glob import glob

from tensorboardX import SummaryWriter

import numpy as np
import multiprocessing

import copy
from tqdm import tqdm
from collections import defaultdict

# import horovod.torch as hvd
import torch.utils.data.distributed

from utils import *
from models import *
import time
from user_input import get_user_input
from pprint import pprint
display = pprint

def reTrain(L):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    is_hvd = False

    tag = 'nohvd'
    list = L

    base = list['base']
    style_weight = list['style_weight']
    content_weight = list['content_weight']
    tv_weight = list['tv_weight']
    epochs = list['epochs']

    batch_size = list['batch_size']
    width = list['width']

    verbose_hist_batch = list['verbose_hist_batch']
    verbose_image_batch = list['verbose_image_batch']

    model_name = f'metanet_base{base}_style{style_weight}_tv{tv_weight}_tag{tag}'
    print(f'model_name: {model_name}')

    print(base,style_weight)


    def rmrf(path):
        try:
            shutil.rmtree(path)
        except:
            pass

    for f in glob('runs/*/.AppleDouble'):
        rmrf(f)

    rmrf('runs/' + model_name)

    vgg16 = models.vgg16(pretrained=True)
    vgg16 = VGG(vgg16.features[:23]).to(device).eval()

    transform_net = TransformNet(base).to(device)
    transform_net.get_param_dict()

    metanet = MetaNet(transform_net.get_param_dict()).to(device)

    import torchvision
    from torchvision import transforms
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(width, scale=(256/480, 1), ratio=(1, 1)),
        transforms.ToTensor(),
        tensor_normalizer
    ])

    style_dataset = torchvision.datasets.ImageFolder(r'D:\Anaconda3Project\styletransfer\dataset\hasu', transform=data_transform)
    content_dataset = torchvision.datasets.ImageFolder(r'D:\Anaconda3Project\styletransfer\dataset\train\train', transform=data_transform)


    content_data_loader = torch.utils.data.DataLoader(content_dataset, batch_size=batch_size,
        shuffle=True)
    # , num_workers=multiprocessing.cpu_count()

    print(style_dataset)
    print('-'*20)
    print(content_dataset)

    metanet.eval()
    transform_net.eval()

    rands = torch.rand(4, 3, 256, 256).to(device)
    features = vgg16(rands);
    weights = metanet(mean_std(features));
    transform_net.set_weights(weights)
    transformed_images = transform_net(torch.rand(4, 3, 256, 256).to(device));

    if not is_hvd or hvd.rank() == 0:
        print('features:')
        display([x.shape for x in features])

        print('weights:')
        display([x.shape for x in weights.values()])

        print('transformed_images:')
        display(transformed_images.shape)

    visualization_style_image = random.choice(style_dataset)[0].unsqueeze(0).to(device)

    visualization_content_images = torch.stack([random.choice(content_dataset)[0] for i in range(4)]).to(device)


    if not is_hvd or hvd.rank() == 0:
        for f in glob('runs/*/.AppleDouble'):
            rmrf(f)

        rmrf('runs/' + model_name)
        writer = SummaryWriter('runs/'+model_name)
    else:
        writer = SummaryWriter('./tmp/'+model_name)  # 修改为相对路径，或者修改为适合你的存储路径

    visualization_style_image = random.choice(style_dataset)[0].unsqueeze(0).to(device)

    # # 删除变量
    # del rands, features, weights, transformed_images# 设置模型为评估模式，防止参数更新
    # 设置模型为评估模式，防止参数更新
    transform_net.eval()
    for param in transform_net.parameters():
        param.requires_grad_(False)

    # 使用 torchvision.utils.make_grid 创建图像网格
    content_grid = torchvision.utils.make_grid(visualization_content_images, nrow=2, normalize=True, scale_each=True)

    # 添加内容图像到 TensorBoard
    writer.add_image('D:\\Anaconda3Project\\styletransfer\\dataset\\train\\train', content_grid, 0)



    # 恢复模型为训练模式
    transform_net.train()
    for param in transform_net.parameters():
        param.requires_grad_(True)

    # 删除变量
    del rands, features, weights, transformed_images



    trainable_params = {}
    trainable_param_shapes = {}
    for model in [vgg16, transform_net, metanet]:
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params[name] = param
                trainable_param_shapes[name] = param.shape



    optimizer = optim.Adam(trainable_params.values(), 1e-3)

    if is_hvd:
        optimizer = hvd.DistributedOptimizer(optimizer,
                                             named_parameters=trainable_params.items())
        params = transform_net.state_dict()
        params.update(metanet.state_dict())
        hvd.broadcast_parameters(params, root_rank=0)

    import os
    n_batch = len(content_data_loader)
    metanet.train()
    transform_net.train()

    for epoch in range(epochs):
        smoother = defaultdict(Smooth)
        with tqdm(enumerate(content_data_loader), total=n_batch) as pbar:
            for batch, (content_images, _) in pbar:
                n_iter = epoch * n_batch + batch

                # 每 20 个 batch 随机挑选一张新的风格图像，计算其特征
                if batch % 20 == 0:
                    style_image = random.choice(style_dataset)[0].unsqueeze(0).to(device)
                    style_features = vgg16(style_image)
                    style_mean_std = mean_std(style_features)

                # 检查纯色
                x = content_images.cpu().numpy()
                if (x.min(-1).min(-1) == x.max(-1).max(-1)).any():
                    continue

                optimizer.zero_grad()

                # 使用风格图像生成风格模型
                weights = metanet(mean_std(style_features))
                transform_net.set_weights(weights, 0)

                # 使用风格模型预测风格迁移图像
                content_images = content_images.to(device)
                transformed_images = transform_net(content_images)

                # 使用 vgg16 计算特征
                content_features = vgg16(content_images)
                transformed_features = vgg16(transformed_images)
                transformed_mean_std = mean_std(transformed_features)

                # content loss
                content_loss = content_weight * F.mse_loss(transformed_features[2], content_features[2])

                # style loss
                style_loss = style_weight * F.mse_loss(transformed_mean_std,
                                                       style_mean_std.expand_as(transformed_mean_std))

                # total variation loss
                y = transformed_images
                tv_loss = tv_weight * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
                                       torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

                # 求和
                loss = content_loss + style_loss + tv_loss

                loss.backward()
                optimizer.step()

                smoother['content_loss'] += content_loss.item()
                smoother['style_loss'] += style_loss.item()
                smoother['tv_loss'] += tv_loss.item()
                smoother['loss'] += loss.item()

                max_value = max([x.max().item() for x in weights.values()])

                writer.add_scalar('loss/loss', loss, n_iter)
                writer.add_scalar('loss/content_loss', content_loss, n_iter)
                writer.add_scalar('loss/style_loss', style_loss, n_iter)
                writer.add_scalar('loss/total_variation', tv_loss, n_iter)
                writer.add_scalar('loss/max', max_value, n_iter)

                s = 'Epoch: {} '.format(epoch + 1)
                s += 'Content: {:.2f} '.format(smoother['content_loss'])
                s += 'Style: {:.1f} '.format(smoother['style_loss'])
                s += 'Loss: {:.2f} '.format(smoother['loss'])
                s += 'Max: {:.2f}'.format(max_value)

                if (batch + 1) % verbose_image_batch == 0:
                    transform_net.eval()
                    visualization_transformed_images = transform_net(visualization_content_images)
                    transform_net.train()
                    # visualization_transformed_images = torch.cat([style_image, visualization_transformed_images])
                    # writer.add_image('debug', recover_tensor(visualization_transformed_images), n_iter)
                    visualization_transformed_images = torch.cat([style_image, visualization_transformed_images])
                    visualization_transformed_images = torch.transpose(visualization_transformed_images, 1, 2).to(
                        'cpu').detach().numpy()
                    # writer.add_image('debug', visualization_transformed_images, n_iter)

                    del visualization_transformed_images

                if (batch + 1) % verbose_hist_batch == 0:
                    for name, param in weights.items():
                        writer.add_histogram('transform_net.' + name, param.clone().cpu().data.numpy(),
                                             n_iter, bins='auto')

                    for name, param in transform_net.named_parameters():
                        writer.add_histogram('transform_net.' + name, param.clone().cpu().data.numpy(),
                                             n_iter, bins='auto')

                    for name, param in metanet.named_parameters():
                        l = name.split('.')
                        l.remove(l[-1])
                        writer.add_histogram('metanet.' + '.'.join(l), param.clone().cpu().data.numpy(),
                                             n_iter, bins='auto')

                pbar.set_description(s)

                del transformed_images, weights

        # if not is_hvd or hvd.rank() == 0:

        #     # 检查并创建目录
        #     os.makedirs('D:\\Anaconda3Project\\styletransfer\\model', exist_ok=True)

        #     # 修改为你想要保存模型的路径
        #     model_save_path = 'D:\\Anaconda3Project\\styletransfer\\model\\{}_{}.pth'.format(model_name, epoch+1)
        #     transform_net_save_path = 'D:\\Anaconda3Project\\styletransfer\\model\\{}_transform_net_{}.pth'.format(model_name, epoch+1)

        #     # 修改为你想要保存模型的路径
        #     torch.save(metanet.state_dict(), model_save_path)
        #     torch.save(transform_net.state_dict(), transform_net_save_path)

        #     # 修改为你想要加载模型的路径
        #     loaded_metanet_path = 'D:\\Anaconda3Project\\styletransfer\\models\\{}.pth'.format(model_name)
        #     loaded_transform_net_path = 'D:\\Anaconda3Project\\styletransfer\\models\\{}_transform_net.pth'.format(model_name)

        #     # 修改为加载模型的路径
        #     torch.save(metanet.load_state_dict(torch.load(loaded_metanet_path)))
        #     torch.save(transform_net.load_state_dict(torch.load(loaded_transform_net_path)))
        if not is_hvd or hvd.rank() == 0:
            # 检查并创建目录
            os.makedirs('D:\\Anaconda3Project\\styletransfer\\model', exist_ok=True)

            # 修改为你想要保存模型的路径
            model_save_path = 'D:\\Anaconda3Project\\styletransfer\\model\\{}_{}.pth'.format(model_name, epoch + 1)
            transform_net_save_path = 'D:\\Anaconda3Project\\styletransfer\\model\\{}_transform_net_{}.pth'.format(
                model_name, epoch + 1)

            # 保存模型
            torch.save(metanet.state_dict(), model_save_path)
            torch.save(transform_net.state_dict(), transform_net_save_path)

    return True


