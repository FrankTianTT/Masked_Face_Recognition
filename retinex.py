import numpy as np
import cv2


def singleScaleRetinex(img, sigma):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))

    return retinex


def multiScaleRetinex(img, sigma_list):
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)

    retinex = retinex / len(sigma_list)

    return retinex


def colorRestoration(img, alpha, beta):
    img_sum = np.sum(img, axis=2, keepdims=True)

    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))

    return color_restoration


def simplestColorBalance(img, low_clip, high_clip):
    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for u, c in zip(unique, counts):
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c

        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)

    return img


def MSRCR(img, sigma_list, G, b, alpha, beta, low_clip, high_clip):
    img = np.float64(img) + 1.0

    img_retinex = multiScaleRetinex(img, sigma_list)

    img_color = colorRestoration(img, alpha, beta)
    img_msrcr = G * (img_retinex * img_color + b)

    for i in range(img_msrcr.shape[2]):
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                             (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                             255

    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
    img_msrcr = simplestColorBalance(img_msrcr, low_clip, high_clip)

    return img_msrcr


def automatedMSRCR(img, sigma_list, image_name):
    img = np.float64(img) + 1.0

    img_retinex = multiScaleRetinex(img, sigma_list)

    for i in range(img_retinex.shape[2]):
        zero_count = None
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break

        if zero_count is None:
            print(image_name)
            return img

        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break

        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255

    img_retinex = np.uint8(img_retinex)

    return img_retinex


def MSRCP(img, sigma_list, low_clip, high_clip):
    img = np.float64(img) + 1.0

    intensity = np.sum(img, axis=2) / img.shape[2]

    retinex = multiScaleRetinex(intensity, sigma_list)

    intensity = np.expand_dims(intensity, 2)
    retinex = np.expand_dims(retinex, 2)

    intensity1 = simplestColorBalance(retinex, low_clip, high_clip)

    intensity1 = (intensity1 - np.min(intensity1)) / \
                 (np.max(intensity1) - np.min(intensity1)) * \
                 255.0 + 1.0

    img_msrcp = np.zeros_like(img)

    for y in range(img_msrcp.shape[0]):
        for x in range(img_msrcp.shape[1]):
            B = np.max(img[y, x])
            A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])
            img_msrcp[y, x, 0] = A * img[y, x, 0]
            img_msrcp[y, x, 1] = A * img[y, x, 1]
            img_msrcp[y, x, 2] = A * img[y, x, 2]

    img_msrcp = np.uint8(img_msrcp - 1.0)

    return img_msrcp


import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from multiprocessing import Pool
from multiprocessing import cpu_count


def f_train(img_path):
    sigma = [15, 80, 200]
    img = cv2.imread(img_path)
    (new_dataset_name, name), image_name = os.path.split(os.path.split(img_path)[0]), os.path.split(img_path)[1]

    img_en = automatedMSRCR(img, sigma, image_name)
    new_dataset_name = new_dataset_name + "_en"
    os.makedirs(os.path.join(new_dataset_name, name), exist_ok=True)
    cv2.imwrite(os.path.join(new_dataset_name, name, image_name), img_en)


def deal_train(dataset_name='test'):
    all_img_path = []
    for name in os.listdir(dataset_name):
        for img_name in os.listdir(os.path.join(dataset_name, name)):
            img_path = os.path.join(dataset_name, name, img_name)
            all_img_path.append(img_path)

    # process_map(f, all_img_path, max_workers=cpu_count())

    with Pool(cpu_count()) as p:
        p.map(f_train, all_img_path)


def f_assess(img_path):
    sigma = [15, 80, 200]
    img = cv2.imread(img_path)
    new_dataset_name, image_name = os.path.split(img_path)[0], os.path.split(img_path)[1]

    img_en = automatedMSRCR(img, sigma, image_name)
    new_dataset_name = new_dataset_name + "_en"
    os.makedirs(os.path.join(new_dataset_name), exist_ok=True)
    cv2.imwrite(os.path.join(new_dataset_name, image_name), img_en)


def deal_assess(dataset_name='assess'):
    all_img_path = []
    for img_name in os.listdir(os.path.join(dataset_name)):
        img_path = os.path.join(dataset_name, img_name)
        all_img_path.append(img_path)

    # process_map(f, all_img_path, max_workers=cpu_count())

    with Pool(cpu_count()) as p:
        p.map(f_assess, all_img_path)


if __name__ == '__main__':
    deal_assess()

# if __name__ == '__main__':
#     for name in tqdm(os.listdir(DATA_ROOT)):
#         for img_name in os.listdir(os.path.join(DATA_ROOT, name)):
#             img_path = os.path.join(DATA_ROOT, name, img_name)
#             sigma = [15, 80, 200]
#             img = cv2.imread(img_path)
#             img_en = automatedMSRCR(img, sigma)
#             cv2.imwrite(img_path, img_en)
