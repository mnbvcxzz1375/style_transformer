import torch
import multiprocessing

def get_user_input():
    base = int(input("请输入 base（默认为16）: ") or 16)
    style_weight = float(input("请输入 style_weight（默认为50）: ") or 50)
    content_weight = float(input("请输入 content_weight（默认为1）: ") or 1)
    tv_weight = float(input("请输入 tv_weight（默认为1e-6）: ") or 1e-6)
    epochs = int(input("请输入 epochs（默认为22）: ") or 22)
    batch_size = int(input("请输入 batch_size（默认为2）: ") or 2)
    width = int(input("请输入 width（默认为256）: ") or 256)
    verbose_hist_batch = int(input("请输入 verbose_hist_batch（默认为100）: ") or 100)
    verbose_image_batch = int(input("请输入 verbose_image_batch（默认为800）: ") or 800)

    list = {
        'base': base,
        'style_weight': style_weight,
        'content_weight': content_weight,
        'tv_weight': tv_weight,
        'epochs': epochs,
        'batch_size': batch_size,
        'width': width,
        'verbose_hist_batch': verbose_hist_batch,
        'verbose_image_batch': verbose_image_batch
    }

    return list