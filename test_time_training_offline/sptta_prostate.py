"""
Shape-Prompt-Learning: The proposed SP-TTA for MRI-Prostate dataset in offline setup.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import sys
from pathlib import Path

# 关键：获取项目根目录的绝对路径（注意是 parent.parent）
ROOT_DIR = Path(__file__).resolve().parent.parent  # 假设脚本在 PASS-main/source_training/
sys.path.insert(0, str(ROOT_DIR))  # 插入到搜索路径首位
from utils.init import init_random_and_cudnn, Recorder
from time import time
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data
import torch.nn as nn
from utils.file_utils import *
# from batchgenerators.utilities.file_and_folder_operations import *
from models import *
from models.unet_promptv2 import UNet_SPTTA_V2
from models.unet_prompt_v3 import UNet_SPTTA_V3
from models.unet_prompt_v4 import UNet_SPTTA_V4
from models.unet_prompt_v5 import UNet_SPTTA_V5
from models.unet_prompt_v6 import UNet_SPTTA_V6
from models.unet_prompt_v7 import UNet_SPTTA_V7
from models.unet_prompt_v8 import UNet_SPTTA_V11_FreqAttention
import models.moment_tta.losses as moment_tta_losses
from models.moment_tta.bounds import *
from typing import Any, Callable, List, Tuple
from test_time_training_offline.base_tta import BaseAdapter
from loss_functions.bn_loss import bn_loss
from utils.losses.seg_loss import  BCEDiceLoss
from utils.tools import *
from torchvision.utils import save_image

####
TENT_Prostate = ['Weighted_self_entropy_loss',
         {'weights':[1], 'idc':[0], 'act':'sigmoid'}]


RN_w_CR = ['RN_w_CR_loss',
         {'idc':[0, 1], 'act':'sigmoid', 'k':4, 'd':4, 'alpha':0.001, 'tag':'2d'}]

LSIZE_RIGA = ['KL_class_ratio_entropy_loss',
         {'weights':[1, 5], 'idc':[0, 1],
         'class_ratio_prior':[0.0708947674981479, 0.01705511685075431], 'act':'sigmoid'}]

LSIZE_Prostate = ['KL_class_ratio_entropy_loss',
         {'weights':[1], 'idc':[0],
         'class_ratio_prior':[0.034565616183810766], 'act':'sigmoid_onelabel'}]

def visualize_tensor(tensor, file_name, output_dir="visualizations", normalize=True):
    """
    保存一个批次的张量中的第一张图片。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 从批次中取出第一张图片用于可视化，并确保在CPU上
    image_tensor = tensor[0].clone().detach().cpu()
    
    # save_image 会自动处理归一化 (将范围拉伸到 [0, 255])
    full_path = os.path.join(output_dir, file_name)
    save_image(image_tensor, full_path, normalize=normalize)
    # print(f"Saved visualization to {full_path}") # 可以取消注释来确认保存


class Offline_Adapter(BaseAdapter):
    def __init__(self, args):
        super(Offline_Adapter, self).__init__(args)
        print('device:', self.device, 'gpus', self.gpus)
        if args.model == 'Moment-TTA':
            self.LOSS_Dict = LSIZE_Prostate
        elif args.model == 'Tent-TTA':
            self.LOSS_Dict = TENT_Prostate
        elif args.model == 'RN-CR-TTA':
            self.LOSS_Dict = RN_w_CR
        else:
            self.LOSS_Dict = LSIZE_Prostate

        self.criterion = BCEDiceLoss().to(self.device)
        
        # 新增：单张特征图保存设置
        self.prompt_save_dir = os.path.join(self.model_folder, "single_channel_prompts")
        os.makedirs(self.prompt_save_dir, exist_ok=True)
        self.max_saved_channels = 20  # 最多保存20张最具代表性的特征图

    def visualize_single_channels(self):
        """
        生成并保存约20张最具代表性的单通道特征图
        选择标准：通道激活值的方差排名前20的通道
        """
        prompt = self.model.data_prompt[0].detach().cpu()  # [512, H, W]
        
        # 计算每个通道的激活方差（作为重要性指标）
        channel_variances = torch.var(prompt, dim=(1, 2))  # [512]
        
        # 选择方差最大的前20个通道
        _, topk_indices = torch.topk(channel_variances, k=self.max_saved_channels)
        
        # 保存选中的通道
        for i, channel_idx in enumerate(topk_indices):
            channel_data = prompt[channel_idx]  # [H, W]
            
            plt.figure(figsize=(8, 8))
            plt.imshow(channel_data, cmap='viridis')
            plt.colorbar()
            plt.title(f"Channel {channel_idx} (Var: {channel_variances[channel_idx]:.2f})")
            plt.axis('off')
            
            # 保存为PNG
            save_path = os.path.join(self.prompt_save_dir, f"channel_{channel_idx}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            print(f"Saved channel {channel_idx} to {save_path}")    
    
    def init_from_source_model(self):
        print("Trying to load model from:", args.pretrained_model)
        assert isfile(args.pretrained_model), 'missing model checkpoint!'
        self.pretrained_params = torch.load(args.pretrained_model)
        if self.args.arch == 'unet_2d':  
            self.model = UNet(num_classes=self.args.num_classes)
            self.model.load_state_dict(self.pretrained_params['model_state_dict'])
            self.model = self.model.to(self.device)

        elif self.args.arch == 'unet_2d_sptta':
            self.model = UNet_SPTTA_V7(
                pretrained_path=self.args.pretrained_model,
                num_classes=self.args.num_classes,
                patch_size=self.args.patch_size,
                # 从 args 中获取 Transformer 的超参数
                # 你需要在你的 argparse 中定义这些参数，或者在 tta_configs 中设置
                transformer_dim=getattr(self.args, 'transformer_dim', 512), # 提供默认值
                transformer_heads=getattr(self.args, 'transformer_heads', 8),
                transformer_mlp_dim=getattr(self.args, 'transformer_mlp_dim', 1024),
                transformer_layers=getattr(self.args, 'transformer_layers', 1),
                transformer_dropout=getattr(self.args, 'transformer_dropout', 0.1)
            )
            self.model = self.model.to(self.device)
            self.amp_grad_scaler = GradScaler()
            if self.args.only_bn_updated:
                self.model = self.configure_model_2d()
                self.logger.info(f'{self.args.arch}: Normalization statistics and prompt/data_adaptor to be updated. Transformer frozen.')
            else:
                self.model = self.configure_model_2d() # 仍然调用以正确设置requires_grad
                self.logger.info(f'{self.args.arch}: All specified TTA parameters (including Transformer) to be updated.')
            
        elif self.args.arch == 'unet_2d_sptta_v11_freq_attention': # 🔸 新增对 V11 傅里叶版本的处理
            self.model = UNet_SPTTA_V11_FreqAttention(
                pretrained_path=self.args.pretrained_model,
                num_classes=self.args.num_classes,
                patch_size=self.args.patch_size,
                # 原有的特征 Transformer 参数
                transformer_dim=getattr(self.args, 'transformer_dim', 512),
                transformer_heads=getattr(self.args, 'transformer_heads', 8),
                transformer_mlp_dim=getattr(self.args, 'transformer_mlp_dim', 1024),
                transformer_layers=getattr(self.args, 'transformer_layers', 1), # 这是 feature_transformer_layers
                transformer_dropout=getattr(self.args, 'transformer_dropout', 0.1),
                # 新增的傅里叶自适应模块参数
                use_freq_adaptor=getattr(self.args, 'use_freq_adaptor', True),
                # bofea_freq_adaptor_channels 通常由模型内部的 bottleneck_channels 决定，
                # 或者如果想让它可配置，也可以从args传入。这里假设它与bottleneck_channels一致 (512)。
                # 如果你的 UNet_SPTTA_V11_FreqAttention 的 __init__ 将其作为参数，那么这里也要提供。
                # 例如： 
                
                bofea_freq_adaptor_channels=getattr(self.args, 'bofea_freq_adaptor_channels', 512),
                bofea_freq_adaptor_attn_dim=getattr(self.args, 'bofea_freq_adaptor_attn_dim', 128),
                bofea_freq_adaptor_heads=getattr(self.args, 'bofea_freq_adaptor_heads', 4),
                bofea_freq_adaptor_mlp_factor=getattr(self.args, 'bofea_freq_adaptor_mlp_factor', 2),
                bofea_freq_adaptor_dropout=getattr(self.args, 'bofea_freq_adaptor_dropout', 0.1),
                bofea_freq_transformer_layers=getattr(self.args, 'bofea_freq_transformer_layers', 1)
            )
            self.model = self.model.to(self.device)
            if torch.cuda.is_available():
                self.amp_grad_scaler = GradScaler()

            # configure_model_2d 会处理哪些部分可训练
            self.model = self.configure_model_2d()
            # 日志在 configure_model_2d 中处理
            
        else:
            raise NotImplementedError(f"Architecture '{self.args.arch}' not implemented.")
            
    def configure_model_2d(self):
        """Configure model for TTA."""
        self.model.train()
        self.model.requires_grad_(False)

        # 1. 解冻BN层
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)

        trainable_components_names = [] # 用于日志记录的组件名称列表

        # 2. 解冻模型特定的可TTA参数
        for name, param in self.model.named_parameters():
            is_trainable_component = False
            component_name = None

            if name.startswith('data_prompt'): # 更精确的匹配，假设参数名为 'data_prompt'
                is_trainable_component = True
                component_name = 'data_prompt'
            elif name.startswith('spatial_data_adaptor') or name.startswith('data_adaptor'): # 适配V7和V11
                is_trainable_component = True
                component_name = 'spatial_data_adaptor' # 或 'data_adaptor'
            elif name.startswith('prompt2feature'):
                is_trainable_component = True
                component_name = 'prompt2feature'
            elif hasattr(self.model, 'structure_head') and name.startswith('structure_head'):
                is_trainable_component = True
                component_name = 'structure_head'
            
            # 特征 Transformer (在V7中是 transformer_encoder, 在V11中是 feature_transformer_encoder)
            # 我们用一个通用的检查
            if (hasattr(self.model, 'transformer_encoder') and name.startswith('transformer_encoder')) or \
               (hasattr(self.model, 'feature_transformer_encoder') and name.startswith('feature_transformer_encoder')):
                if not self.args.only_bn_updated:
                    is_trainable_component = True
                    component_name = 'feature_transformer'
                # else: param.requires_grad remains False

            # 🔸 新增：傅里叶自适应模块 (freq_data_adaptor) 和融合参数 (fusion_alpha)
            if hasattr(self.model, 'freq_data_adaptor') and name.startswith('freq_data_adaptor'):
                if getattr(self.args, 'use_freq_adaptor', False): # 仅当启用时才考虑训练
                    if not self.args.only_bn_updated: # 通常这类新模块我们希望在TTA中训练
                        is_trainable_component = True
                        component_name = 'freq_data_adaptor'
                    # else: param.requires_grad remains False
            
            if hasattr(self.model, 'fusion_alpha') and name.startswith('fusion_alpha'):
                 if getattr(self.args, 'use_freq_adaptor', False):
                    if not self.args.only_bn_updated:
                        is_trainable_component = True
                        component_name = 'fusion_alpha'
                    # else: param.requires_grad remains False
            
            if is_trainable_component:
                param.requires_grad = True
                if component_name and component_name not in trainable_components_names:
                    trainable_components_names.append(component_name)
        
        # 日志记录
        self.logger.info(f"--- Model: {self.args.arch} ---")
        self.logger.info(f"TTA configuration: Model in train mode. All BatchNorm2d layers set to trainable.")
        if self.args.only_bn_updated:
            self.logger.info(f"Mode 'only_bn_updated=True': Besides BN layers, NO other components are explicitly made trainable by this flag.")
            self.logger.info(f"   (Note: prompt, adaptors etc. are always made trainable by default unless 'only_bn_updated' also controls them, which it currently doesn't in this logic).")
            self.logger.info(f"   The following core TTA components were set to trainable (independent of 'only_bn_updated'): {', '.join(trainable_components_names) if trainable_components_names else 'None'}.")
            if hasattr(self.model, 'feature_transformer_encoder') or hasattr(self.model, 'transformer_encoder'):
                 self.logger.info("   Feature Transformer is FROZEN in 'only_bn_updated' mode.")
            if hasattr(self.model, 'freq_data_adaptor') and getattr(self.args, 'use_freq_adaptor', False):
                 self.logger.info("   Frequency Adaptor is FROZEN in 'only_bn_updated' mode.")

        else: # not only_bn_updated
            self.logger.info(f"Mode 'only_bn_updated=False': BN layers AND other specified TTA components are trainable.")
            self.logger.info(f"   The following TTA components were set to trainable: {', '.join(trainable_components_names) if trainable_components_names else 'None'}.")
            if 'feature_transformer' not in trainable_components_names and (hasattr(self.model, 'feature_transformer_encoder') or hasattr(self.model, 'transformer_encoder')):
                self.logger.warning("   Feature Transformer was expected to be trainable but not found in trainable_components_names. Check naming or logic.")
            if 'freq_data_adaptor' not in trainable_components_names and hasattr(self.model, 'freq_data_adaptor') and getattr(self.args, 'use_freq_adaptor', False):
                self.logger.warning("   Frequency Adaptor was expected to be trainable but not found in trainable_components_names. Check naming or logic.")


        # 调试：打印所有实际可训练的参数
        # trainable_param_count = 0
        # self.logger.debug("Actual trainable parameters after configuration:")
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         self.logger.debug(f"  TRAINABLE: {name} - Shape: {param.shape}")
        #         trainable_param_count += param.numel()
        # self.logger.info(f"Total trainable parameters: {trainable_param_count}")
        
        return self.model
            
#     def configure_model_2d(self):
#         """Configure model for TTA."""
#         self.model.train()  # TTA通常在训练模式下进行（例如，为了BN的批统计或dropout）
#         self.model.requires_grad_(False)  # 首先冻结所有参数
        
#         #1.解冻BN层
#         for m in self.model.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.requires_grad_(True)
                
#         trainable_components = []
        
#         #2.解冻模型特定的可TTA参数
#         for name, param in self.model.named_parameters():
#             if name.find('data_prompt') != -1: # 精确匹配 self.data_prompt
#                 param.requires_grad = True
#                 if 'data_prompt' not in trainable_components: trainable_components.append('data_prompt')
#             elif name.find('data_adaptor') != -1:
#                 param.requires_grad = True
#                 if 'data_adaptor' not in trainable_components: trainable_components.append('data_adaptor')
#             # 假设CrossAttention (prompt2feature) 的参数也想在TTA中更新
#             elif name.find('prompt2feature') != -1:
#                 param.requires_grad = True
#                 if 'prompt2feature' not in trainable_components: trainable_components.append('prompt2feature')

#             # 对于 UNet_SPTTA_V7 及更高版本 (包含 structure_head)
#             if hasattr(self.model, 'structure_head') and name.find('structure_head') != -1:
#                 param.requires_grad = True
#                 if 'structure_head' not in trainable_components: trainable_components.append('structure_head')

#             # 🔸 对于 UNet_SPTTA_V9_Transformer (包含 transformer_encoder)
#             # 只有在 not only_bn_updated 时才解冻 Transformer，或者你也可以选择总是解冻它
#             if hasattr(self.model, 'transformer_encoder') and name.find('transformer_encoder') != -1:
#                 if not self.args.only_bn_updated: # 或者如果你总是想更新它，就移除这个if条件
#                     param.requires_grad = True
#                     if 'transformer_encoder' not in trainable_components: trainable_components.append('transformer_encoder')
#                 # else:
#                 #     param.requires_grad = False # 确保在 only_bn_updated 时是冻结的

#         self.logger.info(f"TTA configuration: Training mode. BN layers requires_grad=True.")
#         if self.args.only_bn_updated:
#             self.logger.info(f"Mode 'only_bn_updated=True': Only BN layers, data_prompt, data_adaptor, prompt2feature, structure_head (if exists) are trainable.")
#             self.logger.info(f"Specifically, these components were set to trainable: {', '.join(trainable_components) if trainable_components else 'None (check naming)'}, plus all BatchNorm2d layers.")
#             if hasattr(self.model, 'transformer_encoder'):
#                  self.logger.info("Transformer encoder is frozen in 'only_bn_updated' mode.")
#         else:
#             self.logger.info(f"Mode 'only_bn_updated=False': BN layers and all specified TTA components (including Transformer if present) are trainable.")
#             self.logger.info(f"Specifically, these components were set to trainable: {', '.join(trainable_components) if trainable_components else 'None (check naming)'}, plus all BatchNorm2d layers.")

#         # 打印所有可训练参数进行验证
#         # self.logger.debug("Trainable parameters after configuration:")
#         # for name, param in self.model.named_parameters():
#         #     if param.requires_grad:
#         #         self.logger.debug(name)
        
#         return self.model
        

#         elif args.arch == 'unet_2d_sptta':
#             self.model = UNet_SPTTA_V7(pretrained_path=args.pretrained_model, num_classes=self.args.num_classes,  patch_size=self.args.patch_size)
#             self.model = self.model.to(self.device)
#             self.amp_grad_scaler = GradScaler()
#             if self.args.only_bn_updated:
#                 self.model = self.configure_model_2d()
#                 self.logger.info('normalization statistics updated')
#             else:
#                 self.logger.info('WARNING all var updated')

#         else:
#             raise NotImplementedError

#     def configure_model_2d(self):
#         """Configure model for use with tent."""
#         # train mode, because tent optimizes the model to minimize entropy
#         self.model.train()
#         # disable grad, to (re-)enable only what tent updates
#         self.model.requires_grad_(False)
#         # configure norm for tent updates: enable grad + force batch statisics
#         for m in self.model.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.requires_grad_(True)
#                 # force use of batch stats in train and eval modes
#                 # if self.args.dataset_name == 'RIGA':
#                 # m.track_running_stats = False
#                 # m.running_mean = None
#                 # m.running_var = None


#         for name, param in self.model.named_parameters():
#             if name.find('prompt') != -1 or name.find('data') != -1:
#                 param.requires_grad = True
#         return self.model
    
  
    def set_train_state(self):
        self.model.train()
     
        
    def optimize_parameters(self, input, loss_fn):
        self.optimizer.zero_grad()

        pred, _, _, _, _, _= self.model(input, training=True) 
        loss = loss_fn(pred)

        loss.backward()
        self.optimizer.step()
        
        return pred, loss
    
    def visualize_adaptation_effect(self, output_dir="visualizations/default", epoch_info=""):
        """
        运行一个批次的数据，并保存原始输入、扰动和自适应后的图像。
        """
        self.logger.info(f"--- Generating visualizations in '{output_dir}' ---")
        self.model.eval()
    
        data_loader = self.ts_dataloader
        
        with torch.no_grad():
            try:
                batch_data = next(iter(data_loader))
                # Original input is a 3D volume, e.g., [1, 32, 384, 384]
                full_volume_inputs = batch_data[0] 
            except StopIteration:
                self.logger.error("Dataloader is empty, cannot visualize.")
                return

            # --- START: THE FIX IS HERE ---
            # 1. Select a single slice for visualization. Let's take the middle one (slice 16).
            # full_volume_inputs shape: [B, D, H, W], e.g., [1, 32, 384, 384]
            # single_slice shape: [B, H, W], e.g., [1, 384, 384]
            single_slice = full_volume_inputs[:, 16, :, :].clone() 

            # 2. Reshape the slice to be a valid 2D input for the model.
            # It needs a channel dimension.
            # Add a channel dimension: [B, 1, H, W], e.g., [1, 1, 384, 384]
            inputs_for_model = single_slice.unsqueeze(1)

            # 3. The model expects 3 channels, so repeat the single channel 3 times.
            # Final shape: [B, 3, H, W], e.g., [1, 3, 384, 384]
            inputs_for_model = inputs_for_model.repeat(1, 3, 1, 1)
            # --- END OF FIX ---

            inputs_for_model = inputs_for_model.to(self.device)
    
            # Now, call the model with the correctly shaped 2D input
            output, spatial_perturbation, adapted_x_spatial = self.model(inputs_for_model, training=False)

            prefix = f"{epoch_info}_" if epoch_info else ""

            # We visualize the 3-channel input we created
            visualize_tensor(inputs_for_model, f"{prefix}01_original_input.png", output_dir=output_dir, normalize=True)
            visualize_tensor(spatial_perturbation, f"{prefix}02_spatial_perturbation.png", output_dir=output_dir, normalize=True)
            visualize_tensor(adapted_x_spatial, f"{prefix}03_adapted_input.png", output_dir=output_dir, normalize=True)
    
        self.logger.info("--- Visualizations saved. ---")

        
        
    

    def evaluate(self):
        start_epoch = 0
        if args.continue_training:
            assert isfile(join(self.model_folder, 'model_final.model')), 'missing model checkpoint!'
            params = torch.load(join(self.model_folder, 'model_final.model'))
            self.model.load_state_dict(params['model_state_dict'])
            self.optimizer.load_state_dict(params['optimizer_state_dict'])
            start_epoch = params['epoch']
        self.logger.info('start epoch: {}'.format(start_epoch))

        loss_name, loss_params = self.LOSS_Dict
        self.logger.info('loss_name:{}'.format(loss_name))
        self.logger.info('Loss params:{}'.format(loss_params))

        loss_class = getattr(moment_tta_losses, loss_name)
        loss_fn = loss_class(**loss_params)

        for k, v in self.model.named_parameters():
            if v.requires_grad:
                print('Trainable', k)

        start = time()
        for epoch in range(start_epoch, self.args.num_epochs):
            self.do_prostate_seg_one_epoch(epoch, loss_fn)
           
            saved_model = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
            print('Saving model_{}.model...'.format('latest'))
            torch.save(saved_model, join(self.model_folder, 'model_latest.model'))
         
        saved_model = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        print('Saving model_{}.model...'.format('final'))
        torch.save(saved_model, join(self.model_folder, 'model_final.model'))
        if isfile(join(self.model_folder, 'model_latest.model')):
            os.remove(join(self.model_folder, 'model_latest.model'))
        total_time = time() - start
        print("Running %d epochs took a total of %.2f seconds." % (args.num_epochs, total_time))

     
        from inference.inference_nets_3d.inference_prostate import test3d_single_label_seg
        dice_avg_all_classes, hd_avg_all_classes, assd_avg_all_classes = test3d_single_label_seg(self.args.model, self.model, self.ts_dataloader, self.logger, self.device, self.visualization_folder, self.metrics_folder, num_classes=self.args.num_classes, test_batch=self.args.batch_size, save_pre=False)
        
        if epoch == self.args.num_epochs - 1:
            self.visualize_single_channels()
      

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="TTA", required=False,choices=['Moment-TTA', 'Tent-TTA', 'RN-CR-TTA'],
                        help='Model name.')
    parser.add_argument('--arch', default="unet_2d_sptta_v11_freq_attention", required=False,
                        help='Network architecture.')
    parser.add_argument('--num_classes', default=1, required=False,
                        help='Num of classes.')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], required=False,
                        help='Device id.')
    parser.add_argument('--manualseed', type=int, default=100, required=False,
                        help='random seed.')
    parser.add_argument('--log_folder', default='log_dir', required=False,
                        help='Log folder.')
    parser.add_argument('--only_bn_updated', default=True, required=False,
                        help='which part to be updated.')
    parser.add_argument('--tag', default="Base2_test", required=False,
                        help='Run identifier.')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[512, 512], required=False,
                        help='patch size.')
    parser.add_argument('--batch_size', type=int, default=32, required=False,
                        help='batch size.')
    parser.add_argument('--initial_lr', type=float, default=1e-3, required=False,
                        help='initial learning rate.')
    parser.add_argument('--optimizer', type=str, default='adam', required=False,
                        help='optimizer method.')
    parser.add_argument('-c', '--continue_training', default=False, required=False, action='store_true',
                        help="restore from checkpoint and continue training.")
    parser.add_argument('--num_threads', type=int, default=0, required=False,
                        help="Threads number of dataloader.")
    parser.add_argument('-r', '--root', default='../../ProSFDA_master/prosfda/RIGAPlus/', required=False,
                        help='dataset root folder.')
    parser.add_argument('--ts_csv', nargs='+', default=['../../ProSFDA_master/prosfda/RIGAPlus/MESSIDOR_Base2_test.csv'],
                        required=False, help='test csv file.')
    parser.add_argument('--num_epochs', type=int, default=100, required=False,
                        help='num_epochs.')
    parser.add_argument('--pretrained_model', default='log_dir/UNet_Source_prostate/checkpoints/model_best.model', required=False,
                        help='pretrained model path.')
    parser.add_argument('--model_episodic', type=bool, default=False, required=False,
                        help='To make adaptation episodic, and reset the  model for each batch, choose True.')
    parser.add_argument('--alpha', type=float, default=0.01, required=False,
                        help='alpha in BN loss.')
    parser.add_argument('--layers', type=int, default=5, required=False,
                        help='layers to calculate bn loss.')
    parser.add_argument('--gamma', type=float, default=0.01, required=False,
                        help='gamma in feature alignment loss.')
    args = parser.parse_args()
    # evaluate(args)
    from scripts.tta_configs import Base1_config,\
    Prostate_RUNMC2UCL_config, Prostate_RUNMC2BMC_config, \
    Prostate_RUNMC2HK_config, Prostate_RUNMC2BIDMC_config, Prostate_RUNMC2I2CVB_config 
    args.__dict__.update(Prostate_RUNMC2I2CVB_config)
    args.tag = args.tag +'_sptta'
    # 1. 创建 Adapter 实例
    adpater = Offline_Adapter(args)
    # 2. 【新增】在 TTA 训练开始前进行一次可视化
    # 这将显示未经优化的 prompt/adaptor 的初始效果
    print("\n--- Visualizing effect BEFORE Test-Time Adaptation ---")
    adpater.visualize_adaptation_effect(output_dir=os.path.join(adpater.visualization_folder, "before_tta"))

    # 3. 运行 TTA 训练和评估
    print("\n--- Starting Test-Time Adaptation (evaluate method) ---")
    adpater.evaluate()

    # 4. 【新增】在 TTA 训练结束后再进行一次可视化
    # 这将显示经过优化的 prompt/adaptor 的最终效果
    print("\n--- Visualizing effect AFTER Test-Time Adaptation ---")
    adpater.visualize_adaptation_effect(output_dir=os.path.join(adpater.visualization_folder, "after_tta"))

    print("\n--- TTA process finished. ---")