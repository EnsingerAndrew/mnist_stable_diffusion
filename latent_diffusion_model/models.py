import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import models, transforms


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total:,}")
    return total



class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.latent_channels = latent_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_in = nn.Conv2d(in_channels=in_channels, out_channels=latent_channels, kernel_size=5, stride=1, padding=2)
        self.conv_out = nn.Conv2d(in_channels=latent_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        self.l1_convlayers = nn.ModuleList([
            nn.Conv2d(in_channels=latent_channels, out_channels=latent_channels, kernel_size=5, stride=1, padding=2) for _ in range(num_layers)
        ])

        self.l2_convlayers = nn.ModuleList([
            nn.Conv2d(in_channels=latent_channels, out_channels=latent_channels, kernel_size=3, stride=1, padding=1) for _ in range(num_layers)
        ])


        self.down_block1 = nn.Sequential(
            nn.PixelUnshuffle(2), 
            nn.Conv2d(
                in_channels=4*latent_channels, 
                out_channels=latent_channels, 
                kernel_size=3, stride=1, padding=1 
            )
        )

        self.act = nn.SiLU()
    
    
    def forward(self, x):
        h1 = self.conv_in(x)
        for block in self.l1_convlayers: 
            h1 = self.act(block(h1))

        h2 = self.down_block1(h1)
        for block in self.l2_convlayers: 
            h2 = self.act(block(h2))

        y = self.conv_out(h2)
        return y
    


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.latent_channels = latent_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_in = nn.Conv2d(in_channels=in_channels, out_channels=latent_channels, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(in_channels=latent_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)

        self.l1_convlayers = nn.ModuleList([
            nn.Conv2d(in_channels=latent_channels, out_channels=latent_channels, kernel_size=5, stride=1, padding=2) for _ in range(num_layers)
        ])

        self.l2_convlayers = nn.ModuleList([
            nn.Conv2d(in_channels=latent_channels, out_channels=latent_channels, kernel_size=3, stride=1, padding=1) for _ in range(num_layers)
        ])

        self.act = nn.SiLU()

        self.up_block1 = nn.Sequential(
            nn.PixelShuffle(2), 
            nn.Conv2d(
                in_channels=int(latent_channels/4), 
                out_channels=latent_channels, 
                kernel_size=3, stride=1, padding=1 
            )
        )

    def forward(self, x):
        h1 = self.conv_in(x)
        for block in self.l1_convlayers: 
            h1 = self.act(block(h1))
    
        h2 = self.up_block1(h1)
        for block in self.l2_convlayers: 
            h2 = self.act(block(h2))

        y = self.conv_out(h2)
        return y



class VGGLoss(nn.Module):
    """Computes the VGG perceptual loss between two batches of images.
    The input and target must be 4D tensors with three channels
    ``(B, 3, H, W)`` and must have equivalent shapes. Pixel values should be
    normalized to the range 0-1.
    """
    def __init__(self, dev):
        super().__init__()
        self.layer = 8 
        self.models = {'vgg16': models.vgg16, 'vgg19': models.vgg19}
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.model = self.models['vgg16'](pretrained=True).features[:self.layer+1]
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(dev)

    def get_features(self, input):
        return self.model(self.normalize(input))

    def forward(self, input, target):
        sep = input.shape[0]
        batch = torch.cat([input, target])
        feats = self.get_features(batch)
        input_feats, target_feats = feats[:sep], feats[sep:]
        return F.mse_loss(input_feats, target_feats)


if __name__ == "__main__": 
    device = torch.device('cuda')
    myme = Encoder(3, 3, 64, 4).cuda()
    mymd = Decoder(3, 3, 64, 4).cuda()
    count_parameters(myme)
    count_parameters(mymd)

    B = 16
    x = torch.randn((B, 3, 28, 28)).cuda()

    z = myme(x)
    y = mymd(z)

    print(x.shape, z.shape, y.shape)

