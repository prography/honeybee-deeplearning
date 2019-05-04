import torch
import os
from model.transfer_net import TransferNet
from model.vgg import VGG16
from glob import glob
from torch.optim.adam import Adam
import torch.nn as nn
from torchvision import transforms
from util.util import load_image, gram_matrix


class Trainer:
    def __init__(self, config, data_loader):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_epoch = config.num_epoch
        self.epoch = config.epoch
        self.image_size = config.image_size
        self.data_loader = data_loader
        self.num_residual = config.num_residual
        self.checkpoint_dir = config.checkpoint_dir
        self.lr = config.lr
        self.content_weight = config.content_weight
        self.style_weight = config.style_weight
        self.style_image = config.style_image
        self.style_dir = config.style_dir
        self.batch_size = config.batch_size

        self.build_model()
        self.load_feature_style()

    def train(self):
        total_step = len(self.data_loader)
        optimizer = Adam(self.transfer_net.parameters(), lr=self.lr)
        loss = nn.MSELoss()
        self.transfer_net.train()

        for epoch in range(self.epoch, self.num_epoch):
            for step, image in enumerate(self.data_loader):
                image = image.to(self.device)
                transformed_image = self.transfer_net(image)

                image_feature = self.vgg(image)
                transformed_image_feature = self.vgg(transformed_image)

                content_loss = self.content_weight*loss(image_feature, transformed_image_feature)

                style_loss = 0
                for ft_y, gm_s in zip(transformed_image_feature, self.gram_style):
                    gm_y = gram_matrix(ft_y)
                    style_loss += load_image(gm_y, gm_s[:self.batch_size, :, :])
                style_loss *= self.style_weight

                total_loss = content_loss + style_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                if step % 10 == 0:
                    print(f"[Epoch {epoch}/{self.num_epoch}] [Batch {step}/{total_step}] "
                          f"[Style loss: {style_loss.item()}] [Content loss loss: {content_loss.item()}]")
            torch.save(self.transfer_net.state_dict(), os.path.join(self.checkpoint_dir, f"TransferNet_{epoch}.pth"))

    def build_model(self):
        self.transfer_net = TransferNet(self.num_residual)
        self.transfer_net.apply(self.weights_init)
        self.transfer_net.to(self.device)
        self.vgg = VGG16(requires_grad=True)
        self.vgg.apply(self.weights_init)
        self.vgg.to(self.device)
        self.load_model()

    def load_model(self):
        print(f"[*] Load model from {self.checkpoint_dir}")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.listdir(self.checkpoint_dir):
            print(f"[!] No checkpoint in {self.checkpoint_dir}")
            return

        transfer_net = glob(os.path.join(self.checkpoint_dir, f'TransferNet_{self.epoch}.pth'))
        vgg_net = glob(os.path.join(self.checkpoint_dir, f'VGG_{self.epoch}.pth'))

        if not (transfer_net or vgg_net):
            print(f"[!] No checkpoint in epoch {self.epoch}")
            return

        self.transfer_net.load_state_dict(torch.load(transfer_net[0]))
        self.vgg.load_state_dict(torch.load(vgg_net[0]))

    def load_feature_style(self):
        image = load_image(os.path.join(self.style_dir, self.style_image), size=self.image_size)
        iamge = transforms.Compose([
            transforms.CenterCrop(min(image.size[0], image.size[1])),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])(image)
        image = image.repeat(self.batch_size, 1, 1, 1)
        image = image.to(self.device)
        style_image = self.vgg(image)
        self.gram_style = [gram_matrix(y) for y in style_image]
        return image

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)

        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

