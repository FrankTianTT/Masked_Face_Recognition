import os

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


class customized_dataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, mode: str, label_to_samples=None):
        self.df = dataframe
        self.mode = mode
        transforms_list1 = [transforms.Resize((128, 128)),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])]
        transforms_list2 = [transforms.Resize((128, 128)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])]
        self.transforms_train = transforms.Compose(transforms_list1)
        self.transforms_test = transforms.Compose(transforms_list2)
        # self.label_to_samples = np.array(label_to_samples)
        self.label_to_samples = label_to_samples

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index: int):
        if self.mode == 'test':
            image_path = self.df.iloc[index]['path']
            pair_path = self.df.iloc[index]['pair_path']
            img = Image.open(image_path).convert("RGB")
            pair_img = Image.open(pair_path).convert("RGB")
            img = self.transforms_train(img)
            pair_img = self.transforms_test(pair_img)
            return {'image': img, 'pair_image': pair_img}

        # target label
        target = self.df.iloc[index]['target']
        image_path = self.df.iloc[index]['path']
        # original image
        img = Image.open(image_path).convert("RGB")
        if self.mode == 'train' or self.mode == 'valid':
            img = self.transforms_train(img)
            return {'image': img, 'target': target}
        else:
            img = self.transforms_test(img)
            pair_path = self.df.iloc[index]['pair_path']
            pair_target = self.df.iloc[index]['pair_target']
            pair_img = Image.open(pair_path).convert("RGB")
            pair_img = self.transforms_test(pair_img)
            return {'image': img, 'target': target, 'pair_image': pair_img, 'pair_target': pair_target}

    def is_exist(self, index):
        image_path = self.df.iloc[index]['path']
        if self.mode == 'train' or self.mode == 'valid':

            return os.path.exists(image_path)
        else:
            pair_path = self.df.iloc[index]['pair_path']
            return os.path.exists(image_path) and os.path.exists(pair_path)

    def detection(self):
        for i in tqdm(range(len(self))):
            if not self.is_exist(i):
                print(i)

    def show(self, index, pred, path):
        image_path = self.df.iloc[index]['path']
        pair_path = self.df.iloc[index]['pair_path']
        img = Image.open(image_path).convert("RGB")
        pair_img = Image.open(pair_path).convert("RGB")
        img = np.asarray(img) / 255
        pair_img = np.asarray(pair_img) / 255

        fig, axs = plt.subplots(1, 2)
        fig.suptitle("pred: {}".format(bool(pred)))
        axs[0].axis('off')
        axs[0].imshow(img)
        axs[1].axis('off')
        axs[1].imshow(pair_img)

        plt.savefig(path)
        plt.close()
