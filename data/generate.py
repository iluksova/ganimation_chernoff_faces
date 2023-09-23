import torch
import os
from PIL import Image
import random
import numpy as np
import csv
import torchvision.transforms as transforms


class GenerateDataset(torch.utils.data.Dataset):
    """docstring for BaseDataset"""

    def __init__(self):
        super(GenerateDataset, self).__init__()



    def name(self):
        return os.path.basename(self.opt.data_root.strip('/'))

    def initialize(self, opt):
        self.opt = opt
        self.is_train = False

        # load input image path 
        self.imgs_name_file = self.opt.input_img_path
        self.output_dir = self.opt.output_path

        # load AUs dictionary 
        self.aus_to_generate = self.load_aus_csv(self.opt.aus_csv)

        self.img2tensor = self.img_transformer()

    def load_aus_csv(self, csv_path):
        aus_data = []
        with open(csv_path) as csvfile:
            aus_csv_reader = csv.reader(csvfile, delimiter=',')
            for row in aus_csv_reader:
                aus_data.append({'id': row[0], 'aus': row[1:18]})
        return aus_data

    def get_img_by_path(self, img_path):
        assert os.path.isfile(img_path), "Cannot find image file: %s" % img_path
        img_type = 'L' if self.opt.img_nc == 1 else 'RGB'
        return Image.open(img_path).convert(img_type)

    def img_transformer(self):
        transform_list = []
        if self.opt.resize_or_crop == 'resize_and_crop':
            transform_list.append(transforms.Resize([self.opt.load_size, self.opt.load_size], Image.BICUBIC))
            transform_list.append(transforms.RandomCrop(self.opt.final_size))
        elif self.opt.resize_or_crop == 'crop':
            transform_list.append(transforms.RandomCrop(self.opt.final_size))
        elif self.opt.resize_or_crop == 'none':
            transform_list.append(transforms.Lambda(lambda image: image))
        else:
            raise ValueError("--resize_or_crop %s is not a valid option." % self.opt.resize_or_crop)

        if self.is_train and not self.opt.no_flip:
            transform_list.append(transforms.RandomHorizontalFlip())

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        img2tensor = transforms.Compose(transform_list)

        return img2tensor

    def __len__(self):
        return len(self.aus_to_generate)

    def __getitem__(self, index):
        print("Getting item " + str(index))

        img_path = self.imgs_name_file

        # load source image
        src_img = self.get_img_by_path(img_path)
        src_img_tensor = self.img2tensor(src_img)

        # load target image
        print(self.aus_to_generate[index]['aus'])

        tar_aus = np.array(self.aus_to_generate[index]['aus']).astype(float) / 5.0
        tar_img_path = os.path.join(self.output_dir, self.aus_to_generate[index]['id'] + ".jpg")

        # record paths for debug and test usage
        data_dict = {'src_img': src_img_tensor, 'tar_aus': tar_aus,
                     'src_path': img_path, 'tar_path': tar_img_path}

        return data_dict
