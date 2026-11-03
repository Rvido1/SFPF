import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from torchvision import transforms
from PIL import Image
import sys
from pathlib import Path
# 关键：获取项目根目录的绝对路径（注意是 parent.parent）
ROOT_DIR = Path(__file__).resolve().parent.parent  # 假设脚本在 PASS-main/source_training/
sys.path.insert(0, str(ROOT_DIR))  # 插入到搜索路径首位
from models.unet_prompt import UNet_SPTTA

def visualize_perturbation(x, perturbation, save_dir='vis_output', prefix='sample'):
    os.makedirs(save_dir, exist_ok=True)

    for i in range(min(4, x.size(0))):
        img = x[i].detach().cpu().permute(1, 2, 0).numpy()
        pert = perturbation[i].detach().cpu().permute(1, 2, 0).numpy()
        adapted = img + pert

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(np.clip(img, 0, 1))
        axs[0].set_title('Input')
        axs[0].axis('off')

        axs[1].imshow(np.clip(pert * 5 + 0.5, 0, 1))  # scale for visibility
        axs[1].set_title('Perturbation')
        axs[1].axis('off')

        axs[2].imshow(np.clip(adapted, 0, 1))
        axs[2].set_title('Adapted Input')
        axs[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_perturb_{i}.png'))
        plt.close()

def visualize_attention(attn_weights, save_dir='vis_output', prefix='sample'):
    os.makedirs(save_dir, exist_ok=True)

    if attn_weights.dim() == 3:  # [B, C, P]
        for i in range(min(4, attn_weights.shape[0])):
            attn = attn_weights[i].detach().cpu().numpy()  # [C, P]
            mean_attn = attn.mean(axis=1)  # 对每个 channel 取平均

            # 你可以把它 reshape 成 1D 的 attention 曲线，也可以显示为热图（假设 C=512）
            plt.figure(figsize=(8, 3))
            plt.plot(mean_attn)
            plt.title('Channel-wise Mean Attention')
            plt.xlabel('Channel Index')
            plt.ylabel('Attention')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{prefix}_attn_{i}.png'))
            plt.close()
    else:
        print(f"⚠️ Unsupported attn_weights shape: {attn_weights.shape}")

def visualize_feature_map(features, save_dir='vis_output', prefix='sample', layer_idx=0):
    os.makedirs(save_dir, exist_ok=True)

    fmap = features[layer_idx].features[0]  # first sample in batch
    fmap = torch.mean(fmap.detach().cpu(), dim=0).numpy()  # mean over channels

    plt.imshow(fmap, cmap='viridis')
    plt.title(f'Feature Map from Layer {layer_idx}')
    plt.axis('off')
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, f'{prefix}_feature_layer{layer_idx}.png'))
    plt.close()


# ========================
# Example Demo Usage (with fixed image path)
# ========================
if __name__ == '__main__': # 假设你模型文件叫 model.py

    # 加载并处理固定路径的一张 tif 图片
    image_path = 'data/RIGAPlus/RIGA/MESSIDOR_Base1/Unlabeled/20051019_38557_0100_PP.tif'
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    x = transform(image).unsqueeze(0)  # shape: [1, 3, 512, 512]

    # 加载模型（这里需指定预训练路径）
    model = UNet_SPTTA(pretrained_path='log_dir/UNet_Source_RIGA/checkpoints/model_best.model', pretrained=False)
    model.eval()

    with torch.no_grad():
        output, bn_f, perturbation, attn_weights = model(x, training=True)

    visualize_perturbation(x, perturbation, prefix='demo')
    visualize_attention(attn_weights, prefix='demo')
    visualize_feature_map(bn_f, layer_idx=0, prefix='demo')