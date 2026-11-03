import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from torchvision import transforms
from PIL import Image
import nibabel as nib
from pathlib import Path
import sys

# ========== 设置路径 ==========
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))  # 插入项目根目录到搜索路径

from models.unet_prompt import UNet_SPTTA

# ========== 可视化函数 ==========
def visualize_perturbation(x, perturbation, save_dir='vis_output', prefix='sample'):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(4, x.size(0))):
        img = x[i].detach().cpu().squeeze().numpy()
        pert = perturbation[i].detach().cpu().squeeze().numpy()
        adapted = img + pert

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(np.clip(img, 0, 1), cmap='gray')
        axs[0].set_title('Input')
        axs[0].axis('off')

        axs[1].imshow(np.clip(pert * 5 + 0.5, 0, 1), cmap='bwr')  # 显示扰动
        axs[1].set_title('Perturbation')
        axs[1].axis('off')

        axs[2].imshow(np.clip(adapted, 0, 1), cmap='gray')
        axs[2].set_title('Adapted Input')
        axs[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_perturb_{i}.png'))
        plt.close()

def visualize_attention(attn_weights, save_dir='vis_output', prefix='sample'):
    os.makedirs(save_dir, exist_ok=True)
    if attn_weights.dim() == 3:  # [B, C, P]
        for i in range(min(4, attn_weights.shape[0])):
            attn = attn_weights[i].detach().cpu().numpy()
            mean_attn = attn.mean(axis=1)
            plt.figure(figsize=(8, 3))
            plt.plot(mean_attn)
            plt.title('Channel-wise Mean Attention')
            plt.xlabel('Channel Index')
            plt.ylabel('Attention')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{prefix}_attn_{i}.png'))
            plt.close()

def visualize_feature_map(features, save_dir='vis_output', prefix='sample', layer_idx=0):
    os.makedirs(save_dir, exist_ok=True)
    fmap = features[layer_idx].features[0]  # first sample
    fmap = torch.mean(fmap.detach().cpu(), dim=0).numpy()
    plt.imshow(fmap, cmap='viridis')
    plt.title(f'Feature Map from Layer {layer_idx}')
    plt.axis('off')
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, f'{prefix}_feature_layer{layer_idx}.png'))
    plt.close()

# ========== 主程序 ==========
if __name__ == '__main__':
    # 读取 NIfTI 文件（前列腺 MRI）
    nii_path = 'data/MRI_prostate/UCL/Case37.nii.gz'
    img = nib.load(nii_path).get_fdata()  # [H, W, D]
    slice_data = img[:, :, img.shape[2] // 2]  # 中间切片

    # 归一化 & 转换为 PIL 灰度图
    slice_norm = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-5)
    slice_uint8 = (slice_norm * 255).astype(np.uint8)
    slice_pil = Image.fromarray(slice_uint8).convert('L')

    # 转为 tensor（保持单通道）
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),  # 输出 shape: [1, H, W]
    ])
    x = transform(slice_pil).unsqueeze(0)  # shape: [1, 1, 512, 512]


    # 加载模型（不改结构）
    model = UNet_SPTTA(pretrained_path='log_dir/UNet_Source_Prostate/SiteA_RUNMC_batch_aug/checkpoints/model_final.model', pretrained=False)
    model.eval()

    # 前向推理
    with torch.no_grad():
        output, bn_f, perturbation, attn_weights = model(x, training=True)

    # 可视化结果
    visualize_perturbation(x, perturbation, prefix='prostate_demo')
    visualize_attention(attn_weights, prefix='prostate_demo')
    visualize_feature_map(bn_f, layer_idx=0, prefix='prostate_demo')
