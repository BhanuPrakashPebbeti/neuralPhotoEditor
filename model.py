import torch
import cv2
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from torchvision.models.resnet import ResNet, BasicBlock
from typing import Optional, Iterable
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

input_channels = 3
first_fmap_channels = 64
last_fmap_channels = 512
output_channels = 3
time_embedding = 256
learning_rate = 1e-3
min_lr = 1e-6
weight_decay = 0.0
n_timesteps = 500
beta_min = 1e-4
beta_max = 2e-2
beta_scheduler = 'linear'
batch_size = 8
image_size = (128, 128)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
class DiffusionUtils:
    def __init__(self, n_timesteps, beta_min, beta_max, device='cuda', scheduler='linear'):
        assert scheduler in ['linear', 'cosine'], 'scheduler must be linear or cosine'

        self.n_timesteps = n_timesteps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.device = device
        self.scheduler = scheduler
        
        self.betas = self.betaSamples()
        self.alphas = 1 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)
    
    
    def betaSamples(self):
        if self.scheduler == 'linear':
            return torch.linspace(start=self.beta_min, end=self.beta_max, steps=self.n_timesteps).to(self.device)

        elif self.scheduler == 'cosine':
            betas = []
            for i in reversed(range(self.n_timesteps)):
                T = self.n_timesteps - 1
                beta = self.beta_min + 0.5*(self.beta_max - self.beta_min) * (1 + np.cos((i/T) * np.pi))
                betas.append(beta)
                
            return torch.Tensor(betas).to(self.device)
    
    
    def sampleTimestep(self, size):
        return torch.randint(low=1, high=self.n_timesteps, size=(size, )).to(self.device)
    
    
    def noiseImage(self, x, t):
        assert len(x.shape) == 4, 'input must be 4 dimensions'
        alpha_hat_sqrts = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        one_mins_alpha_hat_sqrt = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x).to(self.device)
        return (alpha_hat_sqrts * x) + (one_mins_alpha_hat_sqrt * noise), noise
    
    def sample(self, x, model):
        assert len(x.shape) == 4, 'input must be 4 dimensions'
        model.eval()
        
        with torch.no_grad():
            iterations = range(1, self.n_timesteps)
            for i in tqdm(reversed(iterations)):
                #batch of timesteps t
                t = (torch.ones(x.shape[0]) * i).long().to(self.device)
                
                #params
                alpha = self.alphas[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                one_minus_alpha = 1 - alpha
                one_minus_alpha_hat = 1 - alpha_hat
                
                #predict noise pertaining for a given timestep
                predicted_noise = model(x, t)
                
                if i > 1:
                    noise = torch.randn_like(x).to(self.device)
                else:
                    noise = torch.zeros_like(x).to(self.device)
                x = 1/torch.sqrt(alpha) * (x - ((one_minus_alpha / torch.sqrt(one_minus_alpha_hat)) * predicted_noise))
                x = x + (torch.sqrt(beta) * noise)
            return x
        
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim_size, n:int=10000):
        assert dim_size % 2 == 0, 'dim_size should be an even number'
            
        super(SinusoidalEmbedding, self).__init__()
        
        self.dim_size = dim_size
        self.n = n
        
    def forward(self, x:torch.Tensor):
        N = len(x)
        output = torch.zeros(size=(N, self.dim_size)).to(x.device)
        
        for idx in range(0, N):
            for i in range(0, self.dim_size//2):
                emb = x[idx] / (self.n ** (2*i / self.dim_size))
                output[idx, 2*i] = torch.sin(emb)
                output[idx, (2*i) + 1] = torch.cos(emb)
        
        return output
    
class ImageSelfAttention(nn.Module):
    def __init__(self, input_channels:int, n_heads:int):
        super(ImageSelfAttention, self).__init__()
        
        self.input_channels = input_channels
        self.n_heads = n_heads
        self.layernorm = nn.LayerNorm(self.input_channels)
        self.attention = nn.MultiheadAttention(self.input_channels, self.n_heads, batch_first=True)
        
    def forward(self, x:torch.Tensor):
        # shape of x: (N, C, H, W)
        _, C, H, W = x.shape
        x = x.reshape(-1, C, H*W).permute(0, 2, 1)
        normalised_x = self.layernorm(x)
        attn_val, _ = self.attention(normalised_x, normalised_x, normalised_x)
        attn_val = attn_val + x
        attn_val = attn_val.permute(0, 2, 1).reshape(-1, C, H, W)
        return attn_val
    
class Encoder(ResNet):
    def __init__(
        self, input_channels:int, time_embedding:int, 
        block=BasicBlock, block_layers:list=[2, 2, 2, 2], n_heads:int=4):
      
        self.block = block
        self.block_layers = block_layers
        self.time_embedding = time_embedding
        self.input_channels = input_channels
        self.n_heads = n_heads
        
        super(Encoder, self).__init__(self.block, self.block_layers)
        
        #time embedding layer
        self.sinusiodal_embedding = SinusoidalEmbedding(self.time_embedding)
        
        fmap_channels = [64, 64, 128, 256, 512]
        #layers to project time embeddings unto feature maps
        self.time_projection_layers = self.make_time_projections(fmap_channels)
        #attention layers for each feature map
        self.attention_layers = self.make_attention_layers(fmap_channels)
        
        self.conv1 = nn.Conv2d(
            self.input_channels, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False)
        
        self.conv2 = nn.Conv2d(
            64, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3),
            bias=False)

        #delete unwanted layers
        del self.maxpool, self.fc, self.avgpool
        
        
    def forward(self, x:torch.Tensor, t:torch.Tensor):
        #embed time positions
        t = self.sinusiodal_embedding(t)
        
        #prepare fmap2
        fmap1 = self.conv1(x)
        t_emb = self.time_projection_layers[0](t)
        fmap1 = fmap1 + t_emb[:, :, None, None]
        fmap1 = self.attention_layers[0](fmap1)
        
        x = self.conv2(fmap1)
        x = self.bn1(x)
        x = self.relu(x)
        
        #prepare fmap2
        fmap2 = self.layer1(x)
        t_emb = self.time_projection_layers[1](t)
        fmap2 = fmap2 + t_emb[:, :, None, None]
        fmap2 = self.attention_layers[1](fmap2)
        
        #prepare fmap3
        fmap3 = self.layer2(fmap2)
        t_emb = self.time_projection_layers[2](t)
        fmap3 = fmap3 + t_emb[:, :, None, None]
        fmap3 = self.attention_layers[2](fmap3)
        
        #prepare fmap4
        fmap4 = self.layer3(fmap3)
        t_emb = self.time_projection_layers[3](t)
        fmap4 = fmap4 + t_emb[:, :, None, None]
        fmap4 = self.attention_layers[3](fmap4)
        
        #prepare fmap4
        fmap5 = self.layer4(fmap4)
        t_emb = self.time_projection_layers[4](t)
        fmap5 = fmap5 + t_emb[:, :, None, None]
        fmap5 = self.attention_layers[4](fmap5)
        
        return fmap1, fmap2, fmap3, fmap4, fmap5
    
    
    def make_time_projections(self, fmap_channels:Iterable[int]):
        layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.time_embedding, ch)
            ) for ch in fmap_channels ])
        
        return layers
    
    def make_attention_layers(self, fmap_channels:Iterable[int]):
        layers = nn.ModuleList([
            ImageSelfAttention(ch, self.n_heads) for ch in fmap_channels
        ])
        
        return layers
    
class DecoderBlock(nn.Module):
    def __init__(
        self, input_channels:int, output_channels:int, 
        time_embedding:int, upsample_scale:int=2, activation:nn.Module=nn.ReLU,
        compute_attn:bool=True, n_heads:int=4):
        super(DecoderBlock, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.upsample_scale = upsample_scale
        self.time_embedding = time_embedding
        self.compute_attn = compute_attn
        self.n_heads = n_heads
        
        #attention layer
        if self.compute_attn:
            self.attention = ImageSelfAttention(self.output_channels, self.n_heads)
        else:self.attention = nn.Identity()
        
        #time embedding layer
        self.sinusiodal_embedding = SinusoidalEmbedding(self.time_embedding)
        
        #time embedding projection layer
        self.time_projection_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.time_embedding, self.output_channels)
            )

        self.transpose = nn.ConvTranspose2d(
            self.input_channels, self.input_channels, 
            kernel_size=self.upsample_scale, stride=self.upsample_scale)
        
        self.instance_norm1 = nn.InstanceNorm2d(self.transpose.in_channels)

        self.conv = nn.Conv2d(
            self.transpose.out_channels, self.output_channels, kernel_size=3, stride=1, padding=1)
        
        self.instance_norm2 = nn.InstanceNorm2d(self.conv.out_channels)
        
        self.activation = activation()

    
    def forward(self, fmap:torch.Tensor, prev_fmap:Optional[torch.Tensor]=None, t:Optional[torch.Tensor]=None):
        output = self.transpose(fmap)
        output = self.instance_norm1(output)
        output = self.conv(output)
        output = self.instance_norm2(output)
        
        #apply residual connection with previous feature map
        if torch.is_tensor(prev_fmap):
            assert (prev_fmap.shape == output.shape), 'feature maps must be of same shape'
            output = output + prev_fmap
            
        #apply timestep embedding
        if torch.is_tensor(t):
            t = self.sinusiodal_embedding(t)
            t_emb = self.time_projection_layer(t)
            output = output + t_emb[:, :, None, None]
            
            output = self.attention(output)
            
        output = self.activation(output)
        return output
    
class Decoder(nn.Module):
    def __init__(
        self, last_fmap_channels:int, output_channels:int, 
        time_embedding:int, first_fmap_channels:int=64, n_heads:int=4):
        super(Decoder, self).__init__()
        
        self.last_fmap_channels = last_fmap_channels
        self.output_channels = output_channels
        self.time_embedding = time_embedding
        self.first_fmap_channels = first_fmap_channels
        self.n_heads = n_heads

        self.residual_layers = self.make_layers()

        self.final_layer = DecoderBlock(
            self.residual_layers[-1].input_channels, self.output_channels,
            time_embedding=self.time_embedding, activation=nn.Identity, 
            compute_attn=False, n_heads=self.n_heads)

        #set final layer second instance norm to identity
        self.final_layer.instance_norm2 = nn.Identity()


    def forward(self, *fmaps, t:Optional[torch.Tensor]=None):
        #fmaps(reversed): fmap5, fmap4, fmap3, fmap2, fmap1
        fmaps = [fmap for fmap in reversed(fmaps)]
        ouptut = None
        for idx, m in enumerate(self.residual_layers):
            if idx == 0:
                output = m(fmaps[idx], fmaps[idx+1], t)
                continue
            output = m(output, fmaps[idx+1], t)
        
        # no previous fmap is passed to the final decoder block
        # and no attention is computed
        output = self.final_layer(output)
        return output

      
    def make_layers(self, n:int=4):
        layers = []
        for i in range(n):
            if i == 0: in_ch = self.last_fmap_channels
            else: in_ch = layers[i-1].output_channels

            out_ch = in_ch // 2 if i != (n-1) else self.first_fmap_channels
            layer = DecoderBlock(
                in_ch, out_ch, 
                time_embedding=self.time_embedding,
                compute_attn=True, n_heads=self.n_heads)
            
            layers.append(layer)

        layers = nn.ModuleList(layers)
        return layers
    
class DiffusionNet(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder):
        
        super(DiffusionNet, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x:torch.Tensor, t:torch.Tensor):
        enc_fmaps = self.encoder(x, t=t)
        segmentation_mask = self.decoder(*enc_fmaps, t=t)
        return segmentation_mask
    
class Trainer:
    def __init__(self, model, diffusion_utils, 
                 device='cuda', pretrained = True):
        
        self.device = device
        self.model = model.to(self.device)
        self.diffusion_utils = diffusion_utils
        self.best_loss = None
        if pretrained:
            self.load()
            
    def load(self):
        print("=> Loading checkpoint")
        checkpoint = torch.load("/home/bhanu/Desktop/major project/npeApp/models/ddpm_model.pth",map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint["state_dict"])
    
    def addNoiseandMaskedDenoise(self,tback,image,mask,display=False):
        w,h,_ = image.shape
        mask = cv2.resize(mask,(128,128),interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(image,(128,128),interpolation=cv2.INTER_AREA)
        if display:
            plt.imshow(image)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.savefig("PaintedImage.png")
            plt.show()
        imgTransforms = transforms.Compose([
                transforms.Resize((128,128)), 
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        image = Image.fromarray(image)
        image = imgTransforms(image)
        images = torch.unsqueeze(image, 0)
        mask = torch.unsqueeze(torch.tensor(mask), 0).to(self.device)
        t = (torch.ones(images.shape[0])*tback).long().to(self.device)
        x_t, _ = self.diffusion_utils.noiseImage(images.to(self.device), t)
        noisyImages = x_t.cpu()
        noisyImages = (noisyImages.clamp(-1, 1) + 1) / 2
        n = images.shape[0]
        img = noisyImages[0].permute(1, 2, 0).numpy() * 255
        img = img.astype(np.uint8)
        if display:
            plt.imshow(img)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.savefig("NoisyImages.png")
            plt.show()
        self.model.eval()
        with torch.no_grad():
            iterations = range(1, tback)
            for i in tqdm(reversed(iterations)):
                #batch of timesteps t
                t = (torch.ones(x_t.shape[0]) * i).long().to(self.device)
                t_1 = (torch.ones(x_t.shape[0]) * (i-1)).long().to(self.device)
                #params
                alpha = self.diffusion_utils.alphas[t][:, None, None, None]
                beta = self.diffusion_utils.betas[t][:, None, None, None]
                alpha_hat = self.diffusion_utils.alpha_hat[t][:, None, None, None]
                one_minus_alpha = 1 - alpha
                one_minus_alpha_hat = 1 - alpha_hat
                
                #predict noise pertaining for a given timestep
                predicted_noise = self.model(x_t, t)
                
                if i > 1:
                    noise = torch.randn_like(x_t).to(self.device)
                else:
                    noise = torch.zeros_like(x_t).to(self.device)
                
                x_t = 1/torch.sqrt(alpha) * (x_t - ((one_minus_alpha / torch.sqrt(one_minus_alpha_hat)) * predicted_noise))
                x_t = x_t + (torch.sqrt(beta) * noise)
                if i > 10:
                    x_t = x_t*mask + (1-mask)*self.diffusion_utils.noiseImage(images.to(self.device), t_1)[0]
        x_t = x_t.cpu()
        x_t = (x_t.clamp(-1, 1) + 1) / 2

        img = x_t[0].permute(1, 2, 0).numpy() * 255
        img = img.astype(np.uint8)
        if display:
            plt.imshow(img)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.savefig("DeNoisedImages.png")
            plt.show()
        img = cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA)
        return img
    
    def addNoiseandDenoise(self,tback,file_path, display = False):
        imgTransforms = transforms.Compose([
                transforms.Resize((128,128)), 
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        image = Image.open(file_path).convert("RGB")
        image = imgTransforms(image)
        images = torch.unsqueeze(image, 0)
            
        t = (torch.ones(images.shape[0])*tback).long().to(self.device)
        x_t, _ = self.diffusion_utils.noiseImage(images.to(self.device), t)
        noisyImages = x_t.cpu()
        noisyImages = (noisyImages.clamp(-1, 1) + 1) / 2
        n = images.shape[0]
        img = noisyImages[0].permute(1, 2, 0).numpy() * 255
        img = img.astype(np.uint8)
        if display:
            plt.imshow(img)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.savefig("NoisyImages.png")
            plt.show()
        self.model.eval()
        with torch.no_grad():
            iterations = range(1, tback)
            for i in tqdm(reversed(iterations)):
                #batch of timesteps t
                t = (torch.ones(x_t.shape[0]) * i).long().to(self.device)
                
                #params
                alpha = self.diffusion_utils.alphas[t][:, None, None, None]
                beta = self.diffusion_utils.betas[t][:, None, None, None]
                alpha_hat = self.diffusion_utils.alpha_hat[t][:, None, None, None]
                one_minus_alpha = 1 - alpha
                one_minus_alpha_hat = 1 - alpha_hat
                
                #predict noise pertaining for a given timestep
                predicted_noise = self.model(x_t, t)
                
                if i > 1:
                    noise = torch.randn_like(x_t).to(self.device)
                else:
                    noise = torch.zeros_like(x_t).to(self.device)
                
                x_t = 1/torch.sqrt(alpha) * (x_t - ((one_minus_alpha / torch.sqrt(one_minus_alpha_hat)) * predicted_noise))
                x_t = x_t + (torch.sqrt(beta) * noise)
        x_t = x_t.cpu()
        x_t = (x_t.clamp(-1, 1) + 1) / 2

        img = x_t[0].permute(1, 2, 0).numpy() * 255
        img = img.astype(np.uint8)
        if display:
            plt.imshow(img)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.savefig("DeNoisedImages.png")
            plt.show()
        return img


# #encoder, decoder model initialisation
# encoder = Encoder(input_channels, time_embedding, block_layers=[2, 2, 2, 2])
# decoder = Decoder(last_fmap_channels, output_channels, time_embedding, first_fmap_channels)
# model = DiffusionNet(encoder, decoder)

# #diffusion utilities class initialisaion
# diffusion_utils = DiffusionUtils(n_timesteps, beta_min, beta_max, DEVICE, scheduler=beta_scheduler)
# trainer = Trainer(model, diffusion_utils, device=DEVICE, pretrained = True)

# trainer.addNoiseandDenoise(200,file_path="test.jpeg")