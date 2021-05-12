# encoding: utf-8

"""
Dataset Class for chest X ray image loading

"""

import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image
import os

class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None, percent=1):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                folder_name = items[0]
                image_name= items[1]
                label = items[2:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, folder_name, "images", image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

        if percent != 1:
          total = len(self.image_names)
          random_index = np.random.choice(total, int(total * percent), replace = False)
          self.image_names = [self.image_names[i] for i in random_index]
          self.labels = [self.labels[i] for i in random_index]
          print("length of images after sampling:", len(self.image_names), len(self.labels))

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)