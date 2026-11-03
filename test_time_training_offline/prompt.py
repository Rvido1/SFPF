import torch
import matplotlib.pyplot as plt

# ====== 路径部分 ======
model_path = 'log_dir/Tent-EMA_Base1_test/checkpoints/model_final.model'  # ← 替换为你的 .model 路径
pretrained_path = 'log_dir/UNet_Source_RIGA/checkpoints/model_best.model'      # ← 替换为预训练路径

# ====== 加载模型结构 ======
import sys
from pathlib import Path
# 关键：获取项目根目录的绝对路径（注意是 parent.parent）
ROOT_DIR = Path(__file__).resolve().parent.parent  # 假设脚本在 PASS-main/source_training/
sys.path.insert(0, str(ROOT_DIR))  # 插入到搜索路径首位
from models.unet_prompt import UNet_SPTTA  # ← 替换为实际模型导入路径

# ====== 初始化模型结构（不载入参数） ======
model = UNet_SPTTA(pretrained_path=pretrained_path)

# ====== 加载 .model 文件并提取模型权重 ======
ckpt = torch.load(model_path, map_location='cpu')

# 提取 state_dict
state_dict = ckpt.get('model_state_dict') or ckpt.get('target_model_state_dict') or ckpt
model.load_state_dict(state_dict, strict=False)

# ====== 提取 data_prompt ======
data_prompt = model.data_prompt.detach().cpu()  # shape: [1, C, H, W]

print("data_prompt shape:", data_prompt.shape)

# ====== 可视化前几个通道 ======
n_channels = min(6, data_prompt.shape[1])
fig, axs = plt.subplots(1, n_channels, figsize=(16, 4))

for i in range(n_channels):
    axs[i].imshow(data_prompt[0, i], cmap='viridis')
    axs[i].set_title(f'Channel {i}')
    axs[i].axis('off')

plt.suptitle('Visualized data_prompt Channels')
plt.tight_layout()
plt.show()
