from util.util import load_image
import torch
from model.transfer_net import TransferNet
import os
from glob import glob
from torchvision import transforms
from torchvision.utils import save_image


class Tester:
    def __init__(self, config, data_loader):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_size = config.image_size
        self.checkpoint_dir = config.checkpoint_dir
        self.data_loader = data_loader
        self.epoch = config.epoch
        self.batch_size = config.sample_batch_size
        self.style_dir = config.style_dir
        self.style_image_name = config.style_image_name
        self.num_residual = config.num_residual
        self.sample_dir = config.sample_dir

        self.build_model()
        self.load_feature_style()

    def test(self):
        self.transfer_net.eval()
        for step in self.data_loader:
            transformed_image = self.transfer_net(self.style_image)
            save_image(transformed_image, os.path.join(self.sample_dir, f"{self.style_image_name}_{step}.png")
                       , normalize=False)

    def build_model(self):
        self.transfer_net = TransferNet(self.num_residual)
        self.transfer_net.to(self.device)
        self.load_model()

    def load_model(self):
        print(f"[*] Load model from {self.checkpoint_dir}")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.listdir(self.checkpoint_dir):
            raise Exception(f"[!] No checkpoint in {self.checkpoint_dir}")

        transfer_net = glob(os.path.join(self.checkpoint_dir, f'TransferNet_{self.epoch}.pth'))

        if not transfer_net:
            print(f"[!] No checkpoint in epoch {self.epoch}")
            return

        self.transfer_net.load_state_dict(torch.load(transfer_net[0]))

    def load_feature_style(self):
        image = load_image(os.path.join(self.style_dir, self.style_image_name), size=self.image_size)
        image = transforms.Compose([
            transforms.CenterCrop(min(image.size[0], image.size[1])),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])(image)
        image = image.repeat(self.batch_size, 1, 1, 1)
        self.style_image = image.to(self.device)

