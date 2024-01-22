import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

class OdometerTypeDataset(Dataset):
    def __init__(self, file, image_dir, image_size, batch_size=4, transform=None):
        """
        
        :param file: the path of txt file.
        :param image_dir: the path of directory with all images.
        :param transform: Optional transform to be applied.
        """

        with open(file) as f:
            self.data = f.readlines()
        self.image_dir = image_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.transform = transform

    def __len__(self):
        return len(self.data) // self.batch_size
    
    def __getitem__(self, index):
        i = index * self.batch_size
        batch_data = self.data[i: i + self.batch_size]
        
        x = torch.zeros((self.batch_size, ) + (3, ) + self.image_size, dtype=torch.float32)
        y = torch.zeros((self.batch_size, 1), dtype=torch.float32)

        for j, data in enumerate(batch_data):
            split = data.split(',')
            image_path = os.path.join(self.image_dir, split[0])
            target = float(split[1][0])

            image = read_image(image_path)
            if self.transform is not None:
                image = self.transform(image)

            x[j] = image / 255.0
            y[j] = target

        return x, y
