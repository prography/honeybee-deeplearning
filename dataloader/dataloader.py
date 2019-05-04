import torch.utils.data
import os
import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import random_split


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size, style):
        self.data_dir = data_dir
        self.image_size = image_size
        self.style = style

        if not os.path.exists(self.data_dir):
            raise Exception(" [!] {}  not exists.".format(self.data_dir))

        self.images = []
        self.image_dir = os.path.join(self.data_dir, self.style)

        for name in os.listdir(self.image_dir):
            for path in glob.glob(os.path.join(self.image_dir, name, '*')):
                self.images.append((path, self.style))

    def __getitem__(self, item):
        path, label = self.images[item]

        image = Image.open(path).convert('RGB')

        transform = transforms.Compose([
            transforms.CenterCrop(min(image.size[0], image.size[1])),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        return transform(image), label

    def __len__(self):
        return len(self.images)


def get_loader(data_dir, image_size, batch_size):
    dataset = Dataset(data_dir, image_size)

    train_length = int(0.9 * len(dataset))
    test_length = len(dataset) - train_length

    train_dataset, test_dataset = random_split(dataset, (train_length, test_length))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    return train_loader, test_loader