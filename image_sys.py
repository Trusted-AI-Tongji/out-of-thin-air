#  name: 图像裁剪,能够循环截取不同部位
#  @param
#  @return
#  

import os
import cv2
import numpy as np
import json
import copy
from PIL import Image

if __name__ == '__main__':
    
    clip_factor = 0  # 图像截取的调节参数
    x_move_factor = 200  # 横坐标噪声移动的调节参数
    y_move_factor = 0  # 纵坐标噪声移动的调节参数
    per_count = 0  # 对抗扰动命名
    epochs = 20
    
    '''读取原始样本的图像大小，必须有的'''
    orig = cv2.imread("output/adv_1.jpg")
    image_height, image_width, _ = orig.shape

    '''图像的预处理设置'''
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    '''读取原始图像'''
    image_path = "output/7.jpg"
    orig = cv2.imread(image_path)[..., ::-1]
    orig = cv2.resize(orig, (736, 736), interpolation=cv2.INTER_CUBIC)
    img = orig.copy().astype(np.float32)
    max_change_above = img + 50
    max_change_below = img - 50
    img /= 255.0
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)

    for i in range(0, epochs):
        '''读取对抗噪声'''
        imgs_perturbation_path = "output/perturbation_{}.jpg".format(per_count)
        imgs_perturbation = cv2.imread(imgs_perturbation_path)[..., ::-1]
        imgs_perturbation = imgs_perturbation.copy().astype(np.float32)
        imgs_perturbation /= 255.0
        imgs_perturbation = (imgs_perturbation - mean) / std
        imgs_perturbation = imgs_perturbation.transpose(2, 0, 1)
        # imgs_perturbation = imgs_perturbation + imgs_perturbation
        per_count += 1
        
        # '''PIL裁剪扰动'''
        adv = img
        print(adv.shape)
        exit()
        keypoints_x_min_second, keypoints_y_min_second, keypoints_x_max, keypoints_y_max = 168, 270, 259, 417
        for i_height in range(image_height):  # 遍历图片的所有像素
            for j_width in range(image_width):
                if j_width > (keypoints_x_min_second - clip_factor) and j_width < (keypoints_x_max + clip_factor) \
                and i_height > (keypoints_y_min_second - clip_factor) and i_height < (keypoints_y_max + clip_factor):
                    adv[0, i_height + y_move_factor, j_width + x_move_factor] = img[0, i_height + y_move_factor, j_width + x_move_factor] + imgs_perturbation[0, i_height, j_width]  # 将img1的图像该位置像素替换成img2
                    adv[1, i_height + y_move_factor, j_width + x_move_factor] = img[1, i_height + y_move_factor, j_width + x_move_factor] + imgs_perturbation[1, i_height, j_width]
                    adv[2, i_height + y_move_factor, j_width + x_move_factor] = img[2, i_height + y_move_factor, j_width + x_move_factor] + imgs_perturbation[2, i_height, j_width]
        '''
        对抗样本
        '''
        adv = adv.transpose(1, 2, 0)
        adv = (adv * std) + mean
        adv = adv * 255.0
        adv = np.clip(adv, max_change_below, max_change_above)
        adv = np.clip(adv, 0, 255).astype(np.uint8)

        im = Image.fromarray(adv)
        img_attacked_path = 'output/poseflow/adv_{}.jpg'.format(1)
        im.save(img_attacked_path)
        
        # 再次读取图片
        image_path = 'output/poseflow/adv_{}.jpg'.format(1)
        orig = cv2.imread(image_path)[..., ::-1]
        img = orig.copy().astype(np.float32)
        max_change_above = img + 30
        max_change_below = img - 30
        img /= 255.0
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)


        # adv = img + imgs_perturbation
        # '''
        # 对抗样本
        # '''
        # adv = adv.transpose(1, 2, 0)
        # adv = (adv * std) + mean
        # adv = adv * 255.0
        # adv = np.clip(adv, max_change_below, max_change_above)
        # adv = np.clip(adv, 0, 255).astype(np.uint8)

        # im = Image.fromarray(adv)
        # img_attacked_path = 'output/adv_{}.jpg'.format(8)
        # im.save(img_attacked_path)
                    
