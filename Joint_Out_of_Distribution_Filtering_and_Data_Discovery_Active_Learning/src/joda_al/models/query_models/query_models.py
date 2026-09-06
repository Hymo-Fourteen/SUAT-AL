import torch
import torch.nn as nn 
import torch.nn.init as init
import torch.nn.functional as F


class GenericLossNet(nn.Module):
    def __init__(self, feature_sizes=None, num_channels=None, interm_dim=128):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 128, 256, 512]
        if feature_sizes is None:
            feature_sizes = [32, 16, 8, 4]
        print(num_channels)
        print(feature_sizes)
        blocks=[]
        for f_size, n_channel in zip(feature_sizes,num_channels):
            blocks.append(nn.Sequential(*[nn.AvgPool2d(f_size),nn.Flatten(),nn.Linear(n_channel, interm_dim)]))
            self.add_module(f"Block-{len(blocks) - 1}", blocks[-1])
        self.blocks=blocks
        self.linear = nn.Linear(len(feature_sizes) * interm_dim, 1)

    def forward(self, features):
        out_features=[]

        # print([f.shape for f in features])
        # print(len(features))
        # print(len(self.blocks))
        for i, feature in enumerate(features):
            out_f=self.blocks[i](feature)
            #print(out_f.shape)
            out_features.append(out_f)
        out = self.linear(torch.cat(out_features, 1))
        return out


class LossNet(nn.Module):
    def __init__(self, feature_sizes=[32, 16, 8, 4], num_channels=[64, 128, 256, 512], interm_dim=128):
        super(LossNet, self).__init__()
        
        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)

        self.linear = nn.Linear(4 * interm_dim, 1)
    
    def forward(self, features):
        # print(features[0].shape)
        out1 = self.GAP1(features[0])
        #print(out1.shape)
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))
        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class VAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, z_dim=32, nc=3, f_filt=4,width=32,height=32,additonal_inputs=0):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc

        if type(f_filt) != tuple and type(f_filt) != list:
            f_filt = (f_filt, f_filt)
        self.f_filt = f_filt

        wi2 = width // 2 ** 3
        hi2 = height // 2 ** 3
        wi = (wi2 + 2 - f_filt[0]) // 2 +1
        hi = (hi2 + 2 - f_filt[1]) // 2 +1
        assert width % 2 ** 3 == 0 and height % 2 ** 3 == 0, "Image dimensions should be dividable by 8 and last filter even"
        assert (wi2 + 2 - f_filt[0]) % 2 == 0 and (hi2 + 2 - f_filt[1]) % 2 == 0, "Last filter should be even"
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32 ; 151 x 240
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16; 75 x 120
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8 ; 38 x 60
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, self.f_filt, 2, 1, bias=False),            # B, 1024,  4,  4 ; 19 x 30
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024*wi*hi)),                                 # B, 1024*4*4 //2,2
        )

        self.fc_mu = nn.Linear(1024*wi*hi, z_dim)                            # B, z_dim
        self.fc_logvar = nn.Linear(1024*wi*hi, z_dim)                            # B, z_dim
        self.decoder = nn.Sequential(
            nn.Linear(z_dim+additonal_inputs, 1024*wi2*hi2),                           # B, 1024*8*8
            View((-1, 1024, wi2, hi2)),                               # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),  # B,  512, 16, 16
            # nn.ConvTranspose2d(1024, 512, self.f_filt, 2, 1, bias=False),   # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 1),                       # B,   nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x):
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10, additonal_inputs=0):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim+additonal_inputs, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        return self.net(x)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class TAVAE(VAE):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, z_dim=32, nc=3, f_filt=4,width=32,height=32,additonal_inputs=1):
        super(TAVAE, self).__init__(z_dim=z_dim,nc=nc,f_filt=f_filt,width=width,height=height,additonal_inputs=additonal_inputs)

    def forward(self, r, x):
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        z = torch.cat([z,r],1)
        x_recon = self._decode(z)

        return x_recon, z, mu, logvar


class TADiscriminator(Discriminator):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10, additonal_inputs=1):
        super(TADiscriminator, self).__init__(z_dim=z_dim,additonal_inputs=additonal_inputs)

    def forward(self, r, z):
        z = torch.cat([z, r], 1)
        return self.net(z)


class QueryNet(nn.Module):
    def __init__(self, input_size=2, inter_dim=64):
        super().__init__()

        W1 = torch.rand(input_size, inter_dim, requires_grad=True) #ones
        W2 = torch.rand(inter_dim, 1, requires_grad=True) #ones
        b1 = torch.rand(inter_dim, requires_grad=True) #zeros
        b2 = torch.rand(1, requires_grad=True) #zeros

        self.W1 = torch.nn.parameter.Parameter(W1, requires_grad=True)
        self.W2 = torch.nn.parameter.Parameter(W2, requires_grad=True)
        self.b1 = torch.nn.parameter.Parameter(b1, requires_grad=True)
        self.b2 = torch.nn.parameter.Parameter(b2, requires_grad=True)

        #print(self.W2) # all 1

    def forward(self, X):
        out = torch.sigmoid(torch.matmul(X, torch.relu(self.W1)) + self.b1)
        out = torch.matmul(out, torch.relu(self.W2)) + self.b2
        return out

        

