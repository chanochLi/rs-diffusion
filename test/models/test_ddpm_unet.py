import torch
import torch.nn.functional as F

from models.DDPM_UNet import UNet

def test_unet_once(
    img_size=32,
    img_channels=3,
    base_channels=64,
    channel_mults=(1, 2, 4),
    num_res_blocks=2,
    time_emb_dim=256,
    num_classes=None,
    attention_resolutions=(1,),
    batch_size=4,
    device="cpu",
):
    """
    对 UNet 做一次前向测试，检查形状是否正确。
    """

    # 1. 构造模型
    model = UNet(
        img_channels=img_channels,
        base_channels=base_channels,
        channel_mults=channel_mults,
        num_res_blocks=num_res_blocks,
        time_emb_dim=time_emb_dim,
        time_emb_scale=1.0,
        num_classes=num_classes,
        act=F.relu,
        dropout=0.1,
        attn_resolutions=attention_resolutions,
        num_groups=32,
        init_pad=0,
    ).to(device)

    model.eval()

    # 2. 构造输入
    x = torch.randn(batch_size, img_channels, img_size, img_size, device=device)

    # time: 假设扩散步数在 [0, 1000) 之间
    t = torch.randint(low=0, high=1000, size=(batch_size,), device=device).long().float()
    
    # 类别标签（如果使用类条件）
    if num_classes is not None:
        y = torch.randint(low=0, high=num_classes, size=(batch_size,), device=device)
    else:
        y = None

    # 3. 前向传播
    with torch.no_grad():
        out = model(x, t=t, label=y)

    # 4. 打印结果信息
    print("Input shape :", x.shape)
    print("Output shape:", out.shape)
    assert out.shape == x.shape, f"输出形状 {out.shape} 和输入 {x.shape} 不一致！"

    # 5. 简单数值检查
    print("Output mean:", out.mean().item())
    print("Output std :", out.std().item())
    print("UNet forward test passed.\n")


def main():
    # 选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1) 最基础：无时间、无类别条件（如果你 time_emb_dim=None 时允许）
    print("==== Test 1: time_emb_dim=None, no class conditioning ====")
    model_no_time = UNet(
        img_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        time_emb_dim=None,      # 不用时间嵌入
        num_classes=None,       # 不用类别嵌入
        act=F.relu,
        dropout=0.1,
        attn_resolutions=(),
        num_groups=32,
        init_pad=0,
    ).to(device)

    x = torch.randn(2, 3, 32, 32, device=device)
    with torch.no_grad():
        out = model_no_time(x, t=None, label=None)
    print("Input shape :", x.shape)
    print("Output shape:", out.shape)
    assert out.shape == x.shape
    print("Test 1 passed.\n")

    # 2) 常见 DDPM 设置：有时间条件，无类别条件
    print("==== Test 2: time_emb_dim>0, no class conditioning ====")
    test_unet_once(
        img_size=32,
        img_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        time_emb_dim=256,
        num_classes=None,
        attention_resolutions=(1,),  # 在第 1 个层级加注意力（按你的实现约定）
        batch_size=4,
        device=device,
    )

    # 3) 有时间 + 类别条件的情况
    print("==== Test 3: time_emb_dim>0, with class conditioning ====")
    test_unet_once(
        img_size=32,
        img_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        time_emb_dim=256,
        num_classes=10,          # 假设有 10 个类别
        attention_resolutions=(1,),
        batch_size=4,
        device=device,
    )


if __name__ == "__main__":
    main()