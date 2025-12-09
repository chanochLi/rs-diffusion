import math
import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    """
    使用正弦构成的时间t的embedding
    pE(pos, 2i) = sin(pos / 10000^{2i / d_{module}})
    pE(pos, 2i + 1) = cos(pos / 10000^{2i / d_{module}})
    
    Args:
        dim (int): embedding维度
        scale (float, optional): 对时间t应用的系数. Defaults to 1.0.
    """
    def __init__(self, dim: int, scale: float=1.0) -> None:
        super().__init__()
        assert dim % 2 == 0
        
        # 记录参数
        self.dim = dim
        self.scale = scale
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # 记录设备
        device = t.device
        
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # torch.outer()做外积/笛卡尔积，a: [m], b: [n] -> [m, n]，即out[i, j] = a[i] * a[j]
        emb = torch.outer(t * self.scale, emb)
        # 前一半sin，后一半cos
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb


class DownSample(nn.Module):
    """
    2倍下采样, 使用步长为2的卷积
    
    Args:
        in_channels (int): 下采样的输入维度
    """
    def __init__(self, in_channels: int) -> None:
        super().__init__()
    
        self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # 多余参数为了使用nn.Sequential
        return self.downsample(x)


class UpSample(nn.Module):
    """
    2倍上采样
    
    Args:
        in_channels (int): 上采样的输入维度
    """
    
    def __init__(
        self,
        in_channels: int,
        ) -> None:
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # 多余参数为了使用nn.Sequential
        return self.upsample(x)

class AttentionBlock(nn.Module):
    """
    针对于空间像素的attention, 把通道作为embedding dim, 使用conv完成
    
    Args:
        num_groups (int): 分组norm的组数
        in_channels (int): 输入通道
    """
    def __init__(
        self, 
        num_groups: int,
        in_channels: int
        ) -> None:
        super().__init__()

        self.in_channels = in_channels

        # 额外增加一个norm层
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)

        # 等价于使用nn.Linear(in_channels, in_channels)，只不过x要先做维度变换
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.out = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 记录参数
        b, c, h, w = x.shape
        
        qkv = self.to_qkv(self.norm(x))
        # [b, c, h, w]
        q, k, v = torch.split(qkv, self.in_channels, dim=1)
        
        # [b, h, w, c] -> [b, h*w, c]
        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        # [b, h, w, c] -> [b, c, h*w]
        k = k.view(b, c, h * w)
        # [b, h, w, c] -> [b, h*w, c]
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)
        
        # [b, h*w, c] [b, c, h*w] -> [b, h*w, h*w]
        attn_score = torch.softmax(q @ k // math.sqrt(c), dim=-1)
        
        # [b, h*w, h*w][b, h*w, c] -> [b, c, h, w]
        out = (attn_score @ v).view(b, w, h, c).permute(0, 3, 1, 2)
        out = self.out(out) + x
        
        return out
        

class ResBlock(nn.Module):
    """
    Unet中的残差链接层
    
    Args:
        in_channels (int): 输入通道
        out_channels (int): 输出通道
        dropout (float): dropout的概率
        act: 激活函数
        num_groups (int): 分组卷积和norm的组数
        time_emb_dim (int): 时间PE的维度
        num_classes (int): 类别数
        use_attention (bool): 是否在此残差层应用attention
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
        act = F.relu,
        num_groups: int = 32,
        time_emb_dim: int | None = None,
        num_classes: int | None = None,
        use_attention: bool = False,
        ) -> None:
        super().__init__()
        
        self.act = act
        
        # 两层残差链接
        self.norm_1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm_2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.conv_2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        # time PE，[B, T] -> [B, out_channel]
        self.time_bias = nn.Linear(time_emb_dim, out_channels) if time_emb_dim else None
        self.cls_bias = nn.Embedding(num_classes, out_channels) if num_classes else None
        
        # 残差连接
        self.res_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # attention
        self.attention = AttentionBlock(num_groups, out_channels) if use_attention else nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor | None = None, label: torch.Tensor | None = None) -> torch.Tensor:
        # [b, in_c, h, w]
        out = self.act(self.norm_1(x))
        # [b, in_c, h, w] -> [b, out_c, h, w]
        out = self.conv_1(out)
        
        # [b, t] -> [b, out_c] -> [b, out_c, 1, 1]
        if self.time_bias and time_emb is not None:
            out += self.time_bias(self.act(time_emb))[:, :, None, None]

        # [b, class] -> [b, out_c] -> [b, out_c, 1, 1]
        if self.cls_bias and label is not None:
            out += self.cls_bias(label)[:, :, None, None]
        
        # [b, out_c, h, w]
        out = self.act(self.norm_2(out))
        out = self.conv_2(out) + self.res_connection(x)
        out = self.attention(out)
        
        return out

class UNet(nn.Module):
    """
    ddpm implemented with unet
    
    Args:
    img_channels (int): 图像输入通道
    base_channels (int): 实际计算的输入输出通道, 会先用卷积网络映射
    dropout (float): dropout的概率
    act: 激活函数
    num_groups (int): 分组卷积和norm的组数
    attn_resolutions (tuple): 需要在ResBlock加入attention的层数
    num_res_blocks (int): 每一层ResBlock的层数
    channel_mults (tuple): 每层上下采样的倍数
    time_emb_dim (int): 时间PE的维度
    time_emb_scale (float): 时间PE的系数
    num_classes (int): 类别数
    init_pad (int): 需要pad的像素数, 主要应对w和h为非偶数情况
    """
    def __init__(
        self,
        img_channels: int,
        base_channels: int, 
        act = F.relu,
        dropout: float = 0.1,
        num_groups: int = 32,
        attn_resolutions: tuple = (),
        num_res_blocks: int = 2,
        channel_mults: tuple = (1, 2, 4, 8),
        time_emb_dim: int | None = None, 
        time_emb_scale: float = 1.0,
        num_classes: int | None = None,
        init_pad: int = 0,
        ) -> None:
        super().__init__()
        
        # 如果w或者h不为2的倍数
        self.init_pad = init_pad
        self.act = act
        
        # 时间位置编码
        if time_emb_dim:
            self.time_mlp = nn.Sequential(
                TimeEmbedding(base_channels, time_emb_scale),
                nn.Linear(base_channels, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim) 
            )
        else:
            self.time_mlp = None
            
        # 初始通道映射
        self.init_conv = nn.Conv2d(img_channels, base_channels, kernel_size=3, padding=1)
        
        # 记录每次下采样的通道数
        channels = [base_channels]
        now_channels = base_channels
        
        # 下采样层
        self.downs = nn.ModuleList()
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            
            # 残差层
            for _ in range(num_res_blocks):
                self.downs.append(ResBlock(
                    now_channels,
                    out_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    act=act,
                    num_groups=num_groups,
                    use_attention=i in attn_resolutions,
                ))
                now_channels = out_channels
                channels.append(now_channels)

            # 如果不是最后一层，就下采样
            if i != len(channel_mults) - 1:
                self.downs.append(DownSample(now_channels))
                channels.append(now_channels)
                
        # 映射层，前一层用attention，后一层不用
        self.mid = nn.ModuleList([
            ResBlock(
                now_channels,
                now_channels,
                dropout=dropout,
                act=act,
                num_groups=num_groups,
                num_classes=num_classes,
                use_attention=True,
                time_emb_dim=time_emb_dim
            ),
            ResBlock(
                now_channels,
                now_channels,
                dropout=dropout,
                act=act,
                num_groups=num_groups,
                num_classes=num_classes,
                use_attention=False,
                time_emb_dim=time_emb_dim
            )
        ])
        
        # 上采样层
        self.ups = []
        for i, mult in list(enumerate(channel_mults))[::-1]:
            out_channels = base_channels * mult
            
            
            for _ in range(num_res_blocks + 1):
                self.ups.append(ResBlock(
                    channels.pop() + now_channels,
                    out_channels,
                    dropout,
                    act=act,
                    num_groups=num_groups,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    use_attention=i in attn_resolutions,
                ))
                now_channels = out_channels
                
            if i != 0:
                self.ups.append(UpSample(now_channels))
                
        # 应该没有通道
        assert len(channels) == 0
        
        self.out_norm = nn.GroupNorm(num_groups=num_groups, num_channels=base_channels)
        self.out_conv = nn.Conv2d(base_channels, img_channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None, label: torch.Tensor | None = None) -> torch.Tensor:
        # x: [b, c, h, w]
        if self.init_pad != 0:
            # x: [b, c, h + 2 * init_pad, w + 2 * init_pad]
            x = F.pad(x, (self.init_pad, ) * 4)
        
        # 时间编码
        if self.time_mlp:
            time_emb = self.time_mlp(t)
        else:
            time_emb = None
            
        # 映射通道
        x = self.init_conv(x)
        
        # 跳层连接
        skips = [x]
        
        # 下采样层
        for layer in self.downs:
            x = layer(x, time_emb, label)
            skips.append(x)
        
        # 特征映射
        for layer in self.mid:
            x = layer(x, time_emb, label)
            
        # 上采样，每个block做残差链接（针对downs中ResBlock和DownSample），所以ups中多一个ResBlock
        for layer in self.ups:
            if isinstance(layer, ResBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x, time_emb, label)
            
        x = self.act(self.out_norm(x))
        x = self.out_conv(x)
        
        if self.init_pad != 0:
            return x[:, :, self.init_pad: - self.init_pad, self.init_pad: - self.init_pad]
        else:
            return x

