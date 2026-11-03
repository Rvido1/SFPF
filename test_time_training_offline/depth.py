import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import pandas as pd
import time
import copy

import sys
from pathlib import Path

# 关键：获取项目根目录的绝对路径（注意是 parent.parent）
ROOT_DIR = Path(__file__).resolve().parent.parent  # 假设脚本在 PASS-main/source_training/
sys.path.insert(0, str(ROOT_DIR))  # 插入到搜索路径


# 配置参数
class Config:
    def __init__(self):
        # 数据路径配置
        self.target_data_path = "data/MRI_prostate/BMC_all.csv"  # 目标域数据路径
        
        # 模型配置
        self.pretrained_model_path = "log_dir/UNet_Source_Prostate/SiteA_RUNMC_batch_aug/checkpoints/model_final.model"  # 预训练好的源模型路径
        self.output_dir = "./output"  # 输出保存路径
        
        # 补全网络配置
        self.completion_channels = 64  # 补全网络通道数
        self.completion_blocks = 3  # 补全网络残差块数
        
        # 测试时适应配置
        self.batch_size = 1  # 一次处理一个样本
        self.entropy_threshold = 0.7  # 熵阈值，高于此值认为是不确定区域
        self.tta_iterations = 5  # 每个样本测试时适应迭代次数
        self.tta_lr = 0.0001  # 测试时适应学习率
        self.consistency_weight = 0.5  # 一致性损失权重
        self.completion_weight = 1.0  # 补全损失权重
        
        # 多序列配置
        self.modalities = ['T2WI', 'ADC', 'DWI']  # 使用的MRI序列
        self.primary_modality = 'T2WI'  # 主模态，其他模态辅助
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 解剖先验配置
        self.use_anatomical_prior = True  # 是否使用解剖学先验
        self.prostate_zones = ['PZ', 'TZ', 'CZ']  # 前列腺分区：外周带、移行带、中央腺
        
        # 数据增强配置
        self.enable_augmentation = True  # 测试时增强开关
        
        # 保存配置
        self.save_uncertainty_map = True  # 是否保存不确定性图
        self.save_completion_results = True  # 是否保存补全结果
        
# 读取和预处理 MRI 数据集
class ProstateMRIDataset(Dataset):
    def __init__(self, data_path, modalities, transform=None):
        self.data_path = data_path
        self.modalities = modalities
        self.transform = transform
        
        # 假设 base_dir 是您的基础数据目录，例如 "data/MRI_prostate"
        base_dir = "data/MRI_prostate"  # 请根据实际情况修改
        df = pd.read_csv(os.path.join(base_dir, "BMC_all.csv"))  # 加载 CSV

        self.patient_dirs = df['image'].tolist()  # 或您CSV里的正确列名
        self.valid_patients = []

        for patient in self.patient_dirs:
            # 直接构建完整路径，不再使用 data_path
            patient_path = os.path.join(base_dir, patient)
    
            # 检查路径是否存在且是目录
            if os.path.exists(patient_path):
                if os.path.isdir(patient_path):
                    # 原来的处理逻辑
                    modality_files = {mod: None for mod in modalities}
            
                    for file in os.listdir(patient_path):
                        for mod in modalities:
                            if mod in file and file.endswith(('.nii.gz', '.nii', '.dcm')):
                                modality_files[mod] = os.path.join(patient_path, file)
                                
                # 检查是否所有模态都存在
                    if all(modality_files.values()):
                        self.valid_patients.append((patient, modality_files))
                else:
                    # 如果路径指向的是文件而非目录，直接处理该文件
                    file_path = patient_path
                    # 根据需要进行处理
                    for mod in modalities:
                        if mod in os.path.basename(file_path) and file_path.endswith(('.nii.gz', '.nii', '.dcm')):
                            self.valid_patients.append((patient, {mod: file_path}))
            
                            # 检查是否所有模态都存在
                            if all(modality_files.values()):
                                self.valid_patients.append((patient, modality_files))
    
    def __len__(self):
        return len(self.valid_patients)
    
    def __getitem__(self, idx):
        patient, modality_files = self.valid_patients[idx]
        
        # 为简化示例，假设我们已将 DICOM/NIfTI 转换为可处理的张量
        # 实际应用中需要添加适当的读取和预处理逻辑
        images = {}
        for mod, file_path in modality_files.items():
            # 这里应该有读取医学图像的代码
            # 简化示例，生成随机张量代替实际图像
            image = torch.randn(1, 256, 256)  # 假设大小为 256x256 的单通道图像
            images[mod] = image
            
        # 将多个模态组合为多通道输入
        combined_image = torch.cat([images[mod] for mod in self.modalities], dim=0)
        
        # 假设我们也有分割掩码
        # 实际中需要加载真实掩码或设为 None
        mask = torch.zeros(1, 256, 256)  # 在测试时适应中通常不需要真实标签
        
        sample = {
            'patient_id': patient,
            'image': combined_image,
            'mask': mask,
            'modalities': {mod: images[mod] for mod in self.modalities}
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

# 解剖学先验编码器 - 学习和应用前列腺解剖结构知识
class AnatomicalPriorEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AnatomicalPriorEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
        
    def forward(self, x):
        return self.encoder(x)

# 残差块实现
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual
        return self.relu(out)

# 深度补全网络 - 用于优化分割中的不确定区域
class CompletionNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels=64, n_blocks=3):
        super(CompletionNetwork, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, inner_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(inner_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        # 残差块
        res_blocks = []
        for _ in range(n_blocks):
            res_blocks.append(ResidualBlock(inner_channels * 2))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(inner_channels * 2, inner_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels, inner_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出分割概率图
        )
        
    def forward(self, x, uncertainty_map=None):
        # 如果提供了不确定性图，将其与输入连接
        if uncertainty_map is not None:
            x = torch.cat([x, uncertainty_map], dim=1)
            
        features = self.encoder(x)
        features = self.res_blocks(features)
        completion = self.decoder(features)
        return completion

# 测试时自适应模块 - 在推理时调整模型以适应目标域特性
class TestTimeAdaptation:
    def __init__(self, model, completion_net, config):
        self.model = model
        self.completion_net = completion_net
        self.config = config
        
        # 冻结预训练模型的编码器部分
        # 这里假设编码器包含在self.model.rn中
        for param in self.model.rn.parameters():
            param.requires_grad = False
            
        # 只优化解码器部分
        # 您的模型已经有独立的decoder模块，很适合此处使用
        for param in self.model.up1.parameters():
            param.requires_grad = True
        for param in self.model.up2.parameters():
            param.requires_grad = True
        for param in self.model.up3.parameters():
            param.requires_grad = True
        for param in self.model.up4.parameters():
            param.requires_grad = True
        for param in self.model.up5.parameters():
            param.requires_grad = True
            
        # 设置补全网络为训练模式
        self.completion_net.train()
        
        # 优化器只包含可训练参数
        self.optimizer = torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, self.model.parameters())) + 
            list(self.completion_net.parameters()),
            lr=config.tta_lr
        )
        
    def calculate_uncertainty(self, pred):
        """计算分割预测的不确定性(熵)"""
        # 确保预测在有效范围内
        pred = torch.clamp(pred, 1e-7, 1 - 1e-7)
        entropy = -pred * torch.log(pred) - (1 - pred) * torch.log(1 - pred)
        return entropy
    
    def adapt(self, sample):
        """对单个样本执行测试时适应"""
        # 获取多模态输入
        image = sample['image'].to(self.config.device)
        
        # 1. 首先获取初始分割结果
        with torch.no_grad():
            initial_pred = self.model(image)  # 标准forward模式仅返回分割结果
        
        # 计算不确定性图和其他步骤...
    
        # 在测试时适应迭代过程中，可以利用encoder和decoder分离
        for i in range(self.config.tta_iterations):
            self.optimizer.zero_grad()
        
            # 使用分离的encoder和decoder便于测试时适应
            encoded_features = self.model.encoder(image)  # 固定编码器特征
            current_pred = self.model.decoder(encoded_features)  # 只优化解码器
            
        # 2. 计算不确定性图
        uncertainty_map = self.calculate_uncertainty(initial_pred)
        
        # 3. 确定需要补全的区域
        uncertain_regions = (uncertainty_map > self.config.entropy_threshold).float()
        
        # 4. 准备补全网络的输入 - 图像和不确定性图的组合
        completion_input = torch.cat([image, initial_pred, uncertainty_map], dim=1)
        
        # 5. 对每个样本执行多次测试时适应
        adapted_pred = initial_pred.clone()
        
        # 如果使用解剖学先验，添加先验信息
        anatomical_prior = None
        if self.config.use_anatomical_prior and hasattr(self.model, 'get_anatomical_prior'):
            anatomical_prior = self.model.get_anatomical_prior(image)
        
        # 执行测试时适应迭代
        for i in range(self.config.tta_iterations):
            # 在测试时微调
            self.optimizer.zero_grad()
            
            # 当前模型预测
            current_pred = self.model(image)
            
            # 使用补全网络优化不确定区域
            completion_output = self.completion_net(completion_input)
            
            # 构建新的预测：在确定区域使用原始预测，在不确定区域使用补全结果
            blended_pred = (1 - uncertain_regions) * current_pred + uncertain_regions * completion_output
            
            # 计算一致性损失 - 鼓励模型在确定区域保持一致
            consistency_loss = F.mse_loss(current_pred * (1 - uncertain_regions), 
                                       initial_pred * (1 - uncertain_regions))
            
            # 如果有解剖学先验，添加额外约束
            prior_loss = 0
            if anatomical_prior is not None:  
                # 简化的先验损失示例 - 实际实现应根据具体解剖先验设计
                # 例如：鼓励预测结果符合前列腺形状分布
                prior_loss = F.mse_loss(blended_pred, anatomical_prior)
            
            # 假设我们有某种形式的自监督信号
            # 这里使用简化的熵最小化作为自监督目标
            entropy_min_loss = torch.mean(self.calculate_uncertainty(blended_pred))
            
            # 总损失
            total_loss = (self.config.consistency_weight * consistency_loss + 
                         self.config.completion_weight * entropy_min_loss + 
                         prior_loss)
            
            # 反向传播和优化
            total_loss.backward()
            self.optimizer.step()
            
            # 更新预测结果用于下一次迭代
            adapted_pred = blended_pred.detach()
            
            # 更新补全网络输入
            completion_input = torch.cat([image, adapted_pred, 
                                        self.calculate_uncertainty(adapted_pred)], dim=1)
            
        # 返回适应后的结果
        return {
            'initial_prediction': initial_pred,
            'adapted_prediction': adapted_pred,
            'uncertainty_map': uncertainty_map,
            'completion_result': completion_output
        }

# 用于预处理和测试时增强的变换
class TestTimeTransform:
    def __init__(self, config):
        self.config = config
        self.n_augmentations = 4  # TTA增强数量
        
    def __call__(self, sample):
        """应用测试时增强"""
        if not self.config.enable_augmentation:
            return sample

        # 原始样本
        augmented_samples = [copy.deepcopy(sample)]
        
        # 生成增强版本
        for i in range(1, self.n_augmentations):
            aug_sample = copy.deepcopy(sample)
            
            # 随机选择增强类型
            aug_type = random.choice(['flip', 'rotate', 'intensity'])
            
            if aug_type == 'flip':
                # 水平翻转
                for mod in self.config.modalities:
                    aug_sample['modalities'][mod] = torch.flip(aug_sample['modalities'][mod], dims=[-1])
                aug_sample['image'] = torch.flip(aug_sample['image'], dims=[-1])
                
            elif aug_type == 'rotate':
                # 小角度旋转 (±15度)
                angle = random.uniform(-15, 15)
                for mod in self.config.modalities:
                    aug_sample['modalities'][mod] = transforms.functional.rotate(
                        aug_sample['modalities'][mod], angle)
                aug_sample['image'] = transforms.functional.rotate(aug_sample['image'], angle)
                
            elif aug_type == 'intensity':
                # 强度调整 (±10%)
                factor = random.uniform(0.9, 1.1)
                for mod in self.config.modalities:
                    aug_sample['modalities'][mod] = aug_sample['modalities'][mod] * factor
                aug_sample['image'] = aug_sample['image'] * factor
            
            augmented_samples.append(aug_sample)
            
        return augmented_samples

# 测试时适应预测流程
def run_test_time_adaptation(model, completion_net, test_loader, config):
    """执行测试时适应流程"""
    model.eval()  # 预测模式但部分参数可能会更新
    completion_net.train()  # 补全网络训练模式
    
    # 创建变换
    transform = TestTimeTransform(config)
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 处理每个测试样本
    results = []
    
    for i, sample in enumerate(tqdm(test_loader, desc="Processing target domain samples")):
        patient_id = sample['patient_id'][0]  # 假设batch_size=1
        print(f"\nProcessing patient: {patient_id}")
        
        # 应用测试时增强
        augmented_samples = transform(sample)
        if not isinstance(augmented_samples, list):
            augmented_samples = [augmented_samples]
        
        tta_module = TestTimeAdaptation(model, completion_net, config)
        
        # 对所有增强版本执行测试时适应
        aug_predictions = []
        for aug_sample in augmented_samples:
            tta_result = tta_module.adapt(aug_sample)
            aug_predictions.append(tta_result['adapted_prediction'])
        
        # 逆变换并平均所有增强预测
        # 简化处理，假设只需要对结果求平均
        final_prediction = torch.mean(torch.stack(aug_predictions), dim=0)
        
        # 将结果二值化得到最终分割掩码
        final_mask = (final_prediction > 0.5).float()
        
        # 可视化和保存结果
        if config.save_uncertainty_map or config.save_completion_results:
            save_visualization(sample, tta_result, final_mask, patient_id, config)
        
        # 存储结果
        results.append({
            'patient_id': patient_id,
            'mask': final_mask.cpu().numpy(),
            'uncertainty': tta_result['uncertainty_map'].cpu().numpy()
        })
        
    return results

# 可视化和保存预测结果
def save_visualization(sample, tta_result, final_mask, patient_id, config):
    """保存分割可视化结果"""
    # 创建患者输出目录
    patient_dir = os.path.join(config.output_dir, patient_id)
    os.makedirs(patient_dir, exist_ok=True)
    
    # 获取主模态图像用于可视化
    t2_image = sample['modalities'][config.primary_modality][0, 0].cpu().numpy()
    
    # 将预测结果转换为numpy数组
    initial_pred = tta_result['initial_prediction'][0, 0].cpu().numpy()
    adapted_pred = tta_result['adapted_prediction'][0, 0].cpu().numpy()
    final_binary = final_mask[0, 0].cpu().numpy()
    uncertainty = tta_result['uncertainty_map'][0, 0].cpu().numpy()
    completion = tta_result['completion_result'][0, 0].cpu().numpy()
    
    # 绘制结果
    plt.figure(figsize=(15, 10))
    
    # 原始图像
    plt.subplot(2, 3, 1)
    plt.imshow(t2_image, cmap='gray')
    plt.title('T2WI 图像')
    plt.axis('off')
    
    # 初始预测
    plt.subplot(2, 3, 2)
    plt.imshow(t2_image, cmap='gray')
    plt.imshow(initial_pred, alpha=0.5, cmap='jet')
    plt.title('初始分割')
    plt.axis('off')
    
    # 适应后预测
    plt.subplot(2, 3, 3)
    plt.imshow(t2_image, cmap='gray')
    plt.imshow(adapted_pred, alpha=0.5, cmap='jet')
    plt.title('适应后分割')
    plt.axis('off')
    
    # 二值化最终掩码
    plt.subplot(2, 3, 4)
    plt.imshow(t2_image, cmap='gray')
    plt.imshow(final_binary, alpha=0.5, cmap='jet')
    plt.title('最终分割掩码')
    plt.axis('off')
    
    # 不确定性图
    plt.subplot(2, 3, 5)
    plt.imshow(uncertainty, cmap='hot')
    plt.title('不确定性图')
    plt.colorbar()
    plt.axis('off')
    
    # 补全结果
    plt.subplot(2, 3, 6)
    plt.imshow(t2_image, cmap='gray')
    plt.imshow(completion, alpha=0.5, cmap='cool')
    plt.title('补全后结果')
    plt.axis('off')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(patient_dir, 'segmentation_results.png'))
    plt.close()
    
    # 保存单独的结果图像
    if config.save_uncertainty_map:
        plt.figure()
        plt.imshow(uncertainty, cmap='hot')
        plt.colorbar()
        plt.title('不确定性图')
        plt.axis('off')
        plt.savefig(os.path.join(patient_dir, 'uncertainty_map.png'))
        plt.close()
    
    if config.save_completion_results:
        plt.figure()
        plt.imshow(t2_image, cmap='gray')
        plt.imshow(completion, alpha=0.5, cmap='cool')
        plt.title('深度补全结果')
        plt.axis('off') 
        plt.savefig(os.path.join(patient_dir, 'completion_result.png'))
        plt.close()

# 主函数
def main():
    # 加载配置
    config = Config()
    
    # 确保输出目录存在
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 加载目标域数据集
    test_dataset = ProstateMRIDataset(
        config.target_data_path,
        config.modalities
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0  # 根据系统调整
    )
    
    # 加载预训练模型 - 这里假设是分割模型
    # 注意：根据实际模型架构，可能需要更改此处的加载方式
    # 此处示例的假设模型具有以下特征：
    # 1. in_channels：输入通道数等于模态数量
    # 2. out_channels：输出通道数为1（二分类分割）
    from models.unet import UNet  # 假设的模型导入
    
    # 这里需要使用您实际的模型类和架构
    model = UNet(
        resnet='resnet34',  # 可选：'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
        num_classes=1,      # 二分类分割，输出通道为1
        pretrained=False    # 不使用预训练权重，因为我们会加载您的源模型权重
    )
    model.load_state_dict(torch.load(config.pretrained_model_path))
    model = model.to(config.device)
    
    # 创建补全网络
    # 补全网络输入：原始图像 + 初始分割 + 不确定性图
    in_channels = len(config.modalities) + 1 + 1
    completion_net = CompletionNetwork(
        in_channels=in_channels,
        out_channels=1,
        inner_channels=config.completion_channels, 
        n_blocks=config.completion_blocks
    )
    completion_net = completion_net.to(config.device)
    
    # 执行测试时适应
    results = run_test_time_adaptation(model, completion_net, test_loader, config)
    
    print(f"处理完成。结果保存在: {config.output_dir}")

if __name__ == "__main__":
    main()