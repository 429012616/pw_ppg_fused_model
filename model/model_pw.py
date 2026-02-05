import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.ELU = nn.ELU(inplace=True)
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.ELU(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.ELU(out)
        return out

class ResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.ELU = nn.ELU(inplace=True)
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.ELU(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.ELU(out)
        return out

class AttentionGate1D(nn.Module):
    def __init__(self, in_channels_x, in_channels_g, inter_channels):
        super().__init__()
        self.W_g = nn.Conv1d(in_channels_g, inter_channels, 1)
        self.W_x = nn.Conv1d(in_channels_x, inter_channels, 1)
        self.psi = nn.Conv1d(inter_channels, 1, 1)
        self.ELU = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        psi = self.ELU(self.W_x(x) + self.W_g(g))
        psi = self.sigmoid(self.psi(psi))
        return x * psi


class AttentionGate2D(nn.Module):
    def __init__(self, in_channels_x, in_channels_g, inter_channels):
        super().__init__()
        self.W_g = nn.Conv2d(in_channels_g, inter_channels, 1)
        self.W_x = nn.Conv2d(in_channels_x, inter_channels, 1)
        self.psi = nn.Conv2d(inter_channels, 1, 1)
        self.ELU = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        psi = self.ELU(self.W_x(x) + self.W_g(g))
        psi = self.sigmoid(self.psi(psi))
        return x * psi

class UNet1D(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, input_length=2944):
        super().__init__()
        self.levels = 4
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = in_channels
        out_ch = base_channels
        for i in range(self.levels):
            block = nn.Sequential(
                ResBlock1D(channels, out_ch),
                ResBlock1D(out_ch, out_ch)
            )
            self.encoders.append(block)
            if i < self.levels - 1:
                self.downs.append(nn.Conv1d(out_ch, out_ch, 3, stride=2, padding=1))
            channels = out_ch
            out_ch *= 2

        self.attns = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        chs = [base_channels*(2**i) for i in range(self.levels)]
        prev_ch = chs[-1]
        for skip_ch in reversed(chs[:-1]):
            self.ups.append(nn.Conv1d(prev_ch, skip_ch, 1))
            self.attns.append(AttentionGate1D(skip_ch, skip_ch, skip_ch//2))
            self.decoders.append(nn.Sequential(
                ResBlock1D(skip_ch*2, skip_ch),
                ResBlock1D(skip_ch, skip_ch)
            ))
            prev_ch = skip_ch

    def forward(self, x):
        x = x.squeeze(-1)
        skips = []
        out = x
        for i, enc in enumerate(self.encoders):
            out = enc(out)
            skips.append(out)
            if i < self.levels - 1:
                out = self.downs[i](out)
        for i in range(self.levels-1):
            skip = skips[-2-i]
            out = F.interpolate(out, size=skip.shape[2:], mode='linear', align_corners=False)
            out = self.ups[i](out)
            skip = self.attns[i](skip, out)
            out = torch.cat([out, skip], dim=1)
            out = self.decoders[i](out)
        return out

class UNet2D(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, input_size=(62,92)):
        super().__init__()
        self.levels = 4
        h, w = input_size
        # 计算 padding
        pad_h = (2**self.levels - h % 2**self.levels) % 2**self.levels
        pad_w = (2**self.levels - w % 2**self.levels) % 2**self.levels
        self.pad = (0, pad_w, 0, pad_h)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = in_channels
        out_ch = base_channels
        for i in range(self.levels):
            block = nn.Sequential(
                ResBlock2D(channels, out_ch),
                ResBlock2D(out_ch, out_ch)
            )
            self.encoders.append(block)
            if i < self.levels-1:
                self.downs.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            channels = out_ch
            out_ch *= 2
        self.attns = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        chs = [base_channels*(2**i) for i in range(self.levels)]
        prev_ch = chs[-1]
        for skip_ch in reversed(chs[:-1]):
            self.ups.append(nn.Conv2d(prev_ch, skip_ch, 1))
            self.attns.append(AttentionGate2D(skip_ch, skip_ch, skip_ch//2))
            self.decoders.append(nn.Sequential(
                nn.Conv2d(skip_ch + skip_ch, skip_ch, 1),
                ResBlock2D(skip_ch, skip_ch),
                ResBlock2D(skip_ch, skip_ch)
            ))
            prev_ch = skip_ch

    def forward(self, x):
        x = F.pad(x, self.pad)
        skips = []
        out = x
        for i, enc in enumerate(self.encoders):
            out = enc(out)
            skips.append(out)
            if i < self.levels-1:
                out = self.downs[i](out)
        for i in range(self.levels-1):
            skip = skips[-2-i]
            out = F.interpolate(out, size=skip.shape[2:], mode='bilinear', align_corners=False)
            out = self.ups[i](out)
            skip = self.attns[i](skip, out)
            out = torch.cat([out, skip], dim=1)
            out = self.decoders[i](out)
        # 裁掉 padding
        out = out[:, :, :x.shape[2]-self.pad[3], :x.shape[3]-self.pad[1]]
        return out

class BPFusionAtten(nn.Module):
    def __init__(self, embed_dim=128, base_channels_pw=16, base_channels_stft = 16):
        super().__init__()
        self.unet1 = UNet1D(in_channels=1, base_channels=base_channels_pw, input_length=2944)
        self.unet2 = UNet1D(in_channels=1, base_channels=base_channels_pw, input_length=2944)
        self.unet3 = UNet2D(in_channels=1, base_channels=base_channels_stft, input_size=(62,92))
        self.unet4 = UNet2D(in_channels=1, base_channels=base_channels_stft, input_size=(62,92))
        self.fc1 = nn.Linear(base_channels_pw, embed_dim)
        self.fc2 = nn.Linear(base_channels_pw, embed_dim)
        self.fc3 = nn.Linear(base_channels_stft, embed_dim)
        self.fc4 = nn.Linear(base_channels_stft, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.mlp_fusion_1 = nn.Sequential(
            nn.Linear(4*embed_dim, 128),
            nn.ELU(),
            nn.Linear(128, embed_dim),
        )
        self.mlp_fusion_2 = nn.Sequential(
            nn.Linear(4*embed_dim, 128),
            nn.ELU(),
            nn.Linear(128, embed_dim),
        )
        self.fc_out_1 = nn.Linear(embed_dim, 1)
        self.fc_out_2 = nn.Linear(embed_dim, 1)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x1, x2, x3, x4):
        feat1 = self.unet1(x1)
        feat2 = self.unet2(x2)
        feat3 = self.unet3(x3)
        feat4 = self.unet4(x4)

        out1 = F.adaptive_avg_pool1d(feat1, 1).view(feat1.size(0), -1)
        out2 = F.adaptive_avg_pool1d(feat2, 1).view(feat2.size(0), -1)
        out3 = F.adaptive_avg_pool2d(feat3, (1,1)).view(feat3.size(0), -1)
        out4 = F.adaptive_avg_pool2d(feat4, (1,1)).view(feat4.size(0), -1)

        e1 = self.norm(self.fc1(out1))
        e2 = self.norm(self.fc2(out2))
        e3 = self.norm(self.fc3(out3))
        e4 = self.norm(self.fc4(out4))

        seq = torch.stack([e1, e2, e3, e4], dim=1)
        attn_out, _ = self.attention(seq, seq, seq)
        #a = attn_out.reshape(attn_out.size(0), -1)
        fused1 = self.mlp_fusion_1(attn_out.reshape(attn_out.size(0), -1))  # [B, embed_dim, 1]
        fused2 = self.mlp_fusion_2(attn_out.reshape(attn_out.size(0), -1))  # [B, embed_dim, 1]
        out1 = self.fc_out_1(fused1)
        out2 = self.fc_out_2(fused2)
       # a = torch.cat([out1,out2],dim = 1).permute([1,0])
        return torch.cat([out1,out2],dim = 1)

if __name__ == "__main__":
    model = BPFusionAtten()
    B = 1
    x1 = torch.randn(B, 1, 2944, 1)
    x2 = torch.randn(B, 1, 2944, 1)
    x3 = torch.randn(B, 1, 62, 91)
    x4 = torch.randn(B, 1, 62, 91)
    out = model(x1, x2, x3, x4)
    print("output shape:", out.shape) 
    summary(model.unet1, input_size=(1, 2944),device="cpu")