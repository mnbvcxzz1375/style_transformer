import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from tensorboardX import SummaryWriter
import random
import shutil
from glob import glob
from tqdm import tqdm
from utils import *
from models import *

# % matplotlib
# inline
# % config
# InlineBackend.figure_format = 'retina'

def make_image(style_image1):
    sty_img=style_image1
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(sty_img)
    def rmrf(path):
        try:
            shutil.rmtree(path)
        except:
            pass

    for f in glob('runs/*/.AppleDouble'):
        rmrf(f)

    rmrf('runs/metanet')
    rmrf('runs/transform_net')

    vgg16 = models.vgg16(pretrained=True)
    vgg16 = VGG(vgg16.features[:23]).to(device).eval()

    base = 16
    transform_net = TransformNet(base).to(device)
    transform_net.get_param_dict()
    class MetaNet(nn.Module):
        def __init__(self, param_dict):
            super(MetaNet, self).__init__()
            self.param_num = len(param_dict)
            self.hidden = nn.Linear(1920, 128 * self.param_num)
            self.fc_dict = {}
            for i, (name, params) in enumerate(param_dict.items()):
                self.fc_dict[name] = i
                setattr(self, 'fc{}'.format(i + 1), nn.Linear(128, params))

        # ONNX 要求输出 tensor 或者 list，不能是 dict
        def forward(self, mean_std_features):
            hidden = F.relu(self.hidden(mean_std_features))
            filters = {}
            for name, i in self.fc_dict.items():
                fc = getattr(self, 'fc{}'.format(i + 1))
                filters[name] = fc(hidden[:, i * 128:(i + 1) * 128])
            return list(filters.values())

        def forward2(self, mean_std_features):
            hidden = F.relu(self.hidden(mean_std_features))
            filters = {}
            for name, i in self.fc_dict.items():
                fc = getattr(self, 'fc{}'.format(i + 1))
                filters[name] = fc(hidden[:, i * 128:(i + 1) * 128])
            return filters


    metanet = MetaNet(transform_net.get_param_dict()).to(device)

    width = 256

    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(width, scale=(256 / 480, 1), ratio=(1, 1)),
        transforms.ToTensor(),
        tensor_normalizer
    ])

    content_dataset = torchvision.datasets.ImageFolder(r'flasktest\static\original\1',
                                                       transform=data_transform)
    


    style_weight = 50
    content_weight = 1
    tv_weight = 1e-6
    batch_size = 8

    trainable_params = {}
    trainable_param_shapes = {}
    for model in [vgg16, transform_net, metanet]:
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params[name] = param
                trainable_param_shapes[name] = param.shape

    optimizer = optim.Adam(trainable_params.values(), 1e-3)
    content_data_loader = torch.utils.data.DataLoader(content_dataset, batch_size=batch_size, shuffle=True)
    style_image = read_image(sty_img,
                             target_width=256).to(device)
    # style_image = read_image(f'static\\style\\xingkong.jpg',
    #                          target_width=256).to(device)
    style_features = vgg16(style_image)
    style_mean_std = mean_std(style_features)

    metanet.load_state_dict(
        torch.load('flasktest\\sourse\\model\\metanet_base16_style50_tv1e-06_tagnohvd_5.pth'))
    transform_net.load_state_dict(torch.load(
        'flasktest\\sourse\\model\\metanet_base16_style50_tv1e-06_tagnohvd_transform_net_5.pth'))

    n_batch = 20
    with tqdm(enumerate(content_data_loader), total=n_batch) as pbar:
        for batch, (content_images, _) in pbar:
            x = content_images.cpu().numpy()
            if (x.min(-1).min(-1) == x.max(-1).max(-1)).any():
                continue

            optimizer.zero_grad()

            # 使用风格图像生成风格模型
            weights = metanet.forward2(mean_std(style_features))
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

            if batch > n_batch:
                break






    ''''''

    import os
    from PIL import Image  # Make sure to install the Pillow library for image processing

    def count_images_in_directory(directory):
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']

        # Get all files in the directory
        files = os.listdir(directory)

        # Count the number of files with image extensions
        image_count = sum(1 for file in files if os.path.splitext(file)[1].lower() in image_extensions)
        for file in files:
            print(file)
        return image_count

    # Specify the directory path
    directory_path = 'flasktest/static/original/1/a/'

    # Count the number of images in the directory
    num_images = count_images_in_directory(directory_path)

    print(f"The directory '{directory_path}' contains {num_images} image(s).")

    import os
    from torchvision.utils import save_image

    output_dir = 'flasktest/static/inputs/'
    os.makedirs(output_dir, exist_ok=True)
    
    content_images = torch.stack([content_dataset[i][0] for i in range(num_images)]).to(device)
    # content_images = torch.stack([random.choice(content_dataset)[0] for i in range(num_images)]).to(device)
    transformed_images = transform_net(content_images)

    # transformed_images_vis = torch.cat([x for x in transformed_images], dim=-1)
    # content_images_vis = torch.cat([x for x in content_images], dim=-1)

    for i in range(num_images):
        plt.figure(figsize=(10, 6))

        # plt.subplot(2, 1, 1)
        # imshow(content_images[i])
        # plt.title("Original Image")

        # plt.subplot(2, 1, 2)

        imshow(transformed_images[i])
        plt.axis('off')
        # plt.savefig('output.png')
        plt.savefig(os.path.join(output_dir, f'transformed_image_{i + 1}.png'), bbox_inches='tight')
        plt.title("Transformed Image")



    print(f"Transformed images saved to {output_dir}")

    # save_directory = 'static/inputs/'

    #
    # imshow(style_image)
    # plt.axis('off')
    # plt.savefig(os.path.join(save_directory, 'style_image.png'))
    # plt.clf()
    #
    # imshow(content_images_vis)
    # plt.axis('off')
    # plt.savefig(os.path.join(save_directory, 'content_images_vis.png'))
    # plt.clf()
    #
    # imshow(transformed_images_vis)
    # plt.axis('off')
    # plt.savefig(os.path.join(save_directory, 'transformed_images_vis.png'))
    # plt.clf()


# style_image=r'flasktest\static\style\style3\style3.jpg'
# make_image(style_image)