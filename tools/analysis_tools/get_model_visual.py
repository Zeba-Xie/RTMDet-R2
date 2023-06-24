# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import numpy as np
import torch
from mmengine.config import Config, DictAction

from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


from tensorboardX import SummaryWriter 


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1024, 1024],
        help='input image size')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--size-divisor',
        type=int,
        default=32,
        help='Pad the input image, the minimum size that is divisible '
        'by size_divisor, -1 means do not pad the image.')
    args = parser.parse_args()
    return args


def main():
    register_all_modules()
    args = parse_args()

    if len(args.shape) == 1:
        h = w = args.shape[0]
    elif len(args.shape) == 2:
        h, w = args.shape
    else:
        raise ValueError('invalid input shape')
    ori_shape = (3, h, w)
    divisor = args.size_divisor
    if divisor > 0:
        h = int(np.ceil(h / divisor)) * divisor
        w = int(np.ceil(w / divisor)) * divisor

    input_shape = (3, h, w)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model = MODELS.build(cfg.model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    data_in = torch.rand((1,) + input_shape)
    if torch.cuda.is_available():
        data_in =  data_in.cuda()

    file_name = args.config
    file_name = file_name.split('/')[-1].split('.')[0]

    with SummaryWriter(f'./events/{file_name}', comment='sample_model_visualization') as sw:
        sw.add_graph(model, data_in)
    print('-'*100)
    print(f'Saved events to ./events/{file_name}')
    print(f'Run <tensorboard --logdir=./events/{file_name} --host=127.0.0.1 --port=9009>')
    print('-'*100)

    model_trace = torch.jit.trace(model, data_in) 
    torch.onnx.export(model_trace, data_in, f'./events/{file_name}/{file_name}_trace.onnx') 
    print('-'*100)
    print(f'ONNX trace model is saved in ./events/{file_name}')
    print('-'*100)

if __name__ == '__main__':
    main()
