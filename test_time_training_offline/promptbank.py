import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import sys
from pathlib import Path
# 关键：获取项目根目录的绝对路径（注意是 parent.parent）
ROOT_DIR = Path(__file__).resolve().parent.parent  # 假设脚本在 PASS-main/source_training/
sys.path.insert(0, str(ROOT_DIR))  # 插入到搜索路径首位
from models.unet_prompt import UNet_SPTTA

def visualize_shape_prompt_bank(shape_prompt, save_dir='vis_output', prefix='shape_prompt', cmap='viridis'):
    os.makedirs(save_dir, exist_ok=True)
    
    # shape_prompt 是 [num_prompts, embed_dim, H, W]
    # 假设我们只需要查看第一层（num_prompts）或降维后的数据
    prompt_np = shape_prompt.squeeze(0).detach().cpu().numpy()  # 去除 batch 维度
    
    # 你可以选择显示其中一个通道的切片，或者展开为一维并绘制
    # 例如，选择第一行的所有数据
    prompt_np = prompt_np[0, :, :]  # 如果是[512, 16, 16]的形状，可以选择第一层

    # 可视化为热图
    plt.figure(figsize=(10, 4))
    plt.imshow(prompt_np, aspect='auto', cmap=cmap)
    plt.colorbar()
    plt.title("Shape Prompt Bank")
    plt.xlabel("Embedding Dim")
    plt.ylabel("Prompt Index")
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_bank.png'))
    plt.close()


def process_tif_image(image_path, target_size=(512, 512)):
    # 加载并处理 TIFF 图像
    image = Image.open(image_path)
    if image.mode != 'RGB':  # 如果是灰度图则转换为 RGB
        image = image.convert('RGB')
    
    # 图像转换：resize + 转换为 Tensor
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    x = transform(image).unsqueeze(0)  # 维度变为 [1, 3, 512, 512]
    
    return x

def visualize_input_image(image_tensor, save_dir='vis_output', prefix='input_image'):
    os.makedirs(save_dir, exist_ok=True)
    
    # 将 Tensor 转为 NumPy 数组并显示
    img = image_tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(np.clip(img, 0, 1))
    plt.title("Input Image")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_input.png'))
    plt.close()

# ========================
# Example Demo Usage
# ========================
if __name__ == '__main__':
    # 设置数据路径
    image_path = 'data/RIGAPlus/RIGA/MESSIDOR_Base1/Unlabeled/20051019_38557_0100_PP.tif'  # 替换为你的 .tif 图像路径
    x = process_tif_image(image_path)  # 处理 .tif 图像

    # 加载模型（假设你有预训练的模型）
    model = UNet_SPTTA(pretrained_path='log_dir/Tent-EMA_Base1_test/checkpoints/model_final.model', pretrained=False)
    model.eval()

    with torch.no_grad():
        # 使用模型进行前向传播
        output, bn_f, perturbation, attn_weights = model(x, training=True)

        # 可视化图像
        visualize_input_image(x, save_dir='vis_output', prefix='demo')

        # 可视化 shape prompt bank
        shape_prompt = model.data_prompt  # 假设模型里有 shape_prompt
        visualize_shape_prompt_bank(shape_prompt, save_dir='vis_output', prefix='demo')

    print("Visualization saved!")
