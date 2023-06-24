
import cv2  
import numpy as np  
import mmcv
import os, sys
import random
import math

'''
从trianval(无纯背景图片)最中随机抽取30%的样本
分为三部分 做噪声处理后保存为_noisy
将_nosiy和trianval合并为为最终数据集_add_nosiy

'''

SET_PATH = 'data/hrsc/ImageSets/'
IMG_PATH = 'data/hrsc/FullDataSet/AllImages/'
ANN_PATH = 'data/hrsc/FullDataSet/Annotations/'

LIST_PATH = 'tools/data/hrsc/random_nosiy_imgs_list.txt'

SAVE_IMG_PATH = 'data/hrsc/FullDataSet_add_noisy/images_noisy/'
SAVE_ANN_PATH = 'data/hrsc/FullDataSet_add_noisy/annotations_noisy/'

'''

'''
def get_random_file(folder_path=SET_PATH, prob=0.3):

    file_list = read_list(folder_path+'trainval.txt')

    n_samples = int(len(file_list) * prob)
    sample_paths = random.sample(file_list, n_samples)

    print('Get random imgs:', len(sample_paths), len(sample_paths) / len(file_list))
    print('Save to', LIST_PATH)

    save_list(sample_paths, LIST_PATH)
    # save_list(file_list+sample_paths, folder_path+'trainval_add_noisy.txt')

    return sample_paths

def noisy_transform(img_name, mode='gauss'):
    '''
    img_name: the img's file name
    mode: different mode in transforming
    '''
    img = mmcv.imread(IMG_PATH + img_name)

    if mode == 'gauss':
        mean = 0
        var = 100
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, img.shape)
        gauss = gauss.reshape(img.shape)
        trans_img = img + gauss

    elif mode == 'pepper':
        # 生成一个与原图像尺寸相同的噪声矩阵
        noise = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.randu(noise, 0, 255)

        # 将噪声矩阵二值化，将较小的值设为黑色，较大的值设为白色
        threshold = 245
        noise[noise < threshold] = 0
        noise[noise >= threshold] = 255

        # 将噪声矩阵扩展为三通道，与原图像进行融合
        noise = cv2.merge([noise, noise, noise])
        trans_img = cv2.addWeighted(img, 0.8, noise, 0.2, 0)

    elif mode == 'poisson':
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # 将 L 通道分离出来
        L, A, B = cv2.split(img_lab)

        # 将 L 通道转换为浮点数类型
        L = L.astype(np.float64)

        # 添加泊松噪声
        noise = np.random.poisson(L)

        # 将 L 通道转换回无符号 8 位整数类型
        L = L.astype(np.uint8)

        # 合并 L, A, B 通道
        img_lab = cv2.merge((L, A, B))

        # 将 LAB 颜色空间转换回 BGR 颜色空间
        trans_img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

    if os.path.isdir(SAVE_IMG_PATH):
        print(f'{SAVE_IMG_PATH} already exist')
    else:
        print(f'Mkdir {SAVE_IMG_PATH}')
        os.mkdir(SAVE_IMG_PATH)

    mmcv.imwrite(trans_img, SAVE_IMG_PATH + img_name.split('.')[0] + '_noisy.bmp')

def save_list(list, path):
    with open(path, 'w') as file:
        for item in list:
            file.write(str(item) + '\n')

def read_list(path):
    with open(path, 'r') as file:
        list = [line.strip() for line in file]
    return list
    
def copy_ann_files():
    imgs_list = read_list(LIST_PATH)

    if os.path.isdir(SAVE_ANN_PATH):
        print(f'{SAVE_ANN_PATH} already exist')
    else:
        print(f'Mkdir {SAVE_ANN_PATH}')
        os.mkdir(SAVE_ANN_PATH)

    for img in imgs_list:
        cmd = 'cp ' + ANN_PATH + img.split('.')[0] + '.xml ' + SAVE_ANN_PATH + img.split('.')[0] + '_noisy.xml'
        os.system(cmd)

def gen_files():
    imgs_list = read_list(LIST_PATH)

    n = len(imgs_list)
    m = math.ceil(n / 3)
    random.shuffle(imgs_list)
    split_lists = [imgs_list[i:i+m] for i in range(0, n, m)]

    assert (len(split_lists[0]) + len(split_lists[1]) + len(split_lists[2])) == len(imgs_list)

    for img in split_lists[0]:
        noisy_transform(img+'.bmp', mode='gauss')
    
    for img in split_lists[1]:
        noisy_transform(img+'.bmp', mode='pepper')
    
    for img in split_lists[2]:
        noisy_transform(img+'.bmp', mode='poisson')


if __name__ == '__main__':
    # get_random_file()
    # gen_files()
    copy_ann_files()
