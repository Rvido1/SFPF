"""
Shape-Prompt-Learning: The proposed SP-TTA for RIGA dataset in offline setup.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
from time import time
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import sys
from pathlib import Path
# 关键：获取项目根目录的绝对路径（注意是 parent.parent）
ROOT_DIR = Path(__file__).resolve().parent.parent  # 假设脚本在 PASS-main/source_training/
sys.path.insert(0, str(ROOT_DIR))  # 插入到搜索路径首位
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
from datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set, RIGA_unlabeled_set
from datasets.utils.convert_csv_to_list import convert_labeled_list, convert_unlabeled_list
from datasets.utils.transform import target_collate_fn_tr_fda, collate_fn_ts,collate_fn_tr
from utils.lr import adjust_learning_rate
from utils.metrics.dice import get_hard_dice
from torchvision.utils import make_grid
from torch.cuda.amp import autocast, GradScaler
from models.TENT.tent import configure_model
import models.moment_tta.losses as moment_tta_losses
import models.moment_tta.utils as moment_tta_utils
from models.moment_tta.bounds import *
from typing import Any, Callable, List, Tuple
from test_time_training_offline.base_tta import BaseAdapter
from loss_functions.bn_loss import bn_loss
from datasets.utils.normalize import normalize_image
import torch.nn.functional as F

#### Loss for adaptation

BN_sta = ['bn_loss',
         {'layers':5, 'alpha':0.01}]

TENT = ['Weighted_self_entropy_loss',
         {'weights':[1, 10], 'idc':[0, 1], 'act':'sigmoid'}]

RN_w_CR = ['RN_w_CR_loss',
         {'idc':[0, 1], 'act':'sigmoid', 'k':4, 'd':4, 'alpha':0.001, 'tag':'2d'}]

LSIZE = ['KL_class_ratio_entropy_loss',
         {'weights':[1, 5], 'idc':[0, 1],
         'class_ratio_prior':[0.0708947674981479, 0.01705511685075431], 'act':'sigmoid'}]

LSIZECentroid = ['Constrain_prior_w_self_entropy_loss',
                 {'idc': [0, 1],
                  'weights_se':[0.8, 0.2],'lamb_se':1,
                  'class_ratio_prior':[0.0708947674981479, 0.01705511685075431],
                  'lamb_moment':0.0001, 'temp':1.01,'margin':0,
                  'mom_est':[[254.9747, 255.98499], [253.97656, 255.04263]],
                  'moment_fn':'soft_centroid', 'lamb_consprior':1,
                  'power': 1, 'act':'sigmoid'}]

LSIZEDistCentroid = ['Constrain_prior_w_self_entropy_loss',
                     {'idc':[0, 1],
                      'weights_se':[0.8, 0.2],'lamb_se':1,
                      'class_ratio_prior':[0.0708947674981479, 0.01705511685075431],
                     'lamb_moment':0.0001, 'temp':1.01,'margin':0,
                     'mom_est':[[39.157185, 37.18185 ],[17.430944,18.484102]],
                     'moment_fn':'soft_dist_centroid', 'lamb_consprior':1,
                     'power': 1, 'act':'sigmoid'}]

class Offline_Adapter(BaseAdapter):
    def __init__(self, args):
        #super(SPTTA_Offline, self).__init__(args)
        super(Offline_Adapter, self).__init__(args)
        print('device:', self.device, 'gpus', self.gpus)
        self.loss_fn = None
        if args.model == 'Moment-TTA':
            loss_name, loss_params = LSIZECentroid
            loss_class = getattr(moment_tta_losses, loss_name)
        elif args.model == 'Tent-TTA':
            loss_name, loss_params = TENT
            loss_class = getattr(moment_tta_losses, loss_name)
        elif args.model == 'RN-CR-TTA':
            loss_name, loss_params = RN_w_CR
            loss_class = getattr(moment_tta_losses, loss_name)
        elif args.model == 'BN-Sta-TTA':
            loss_name, loss_params = BN_sta
            loss_class = None
        else:
            loss_name, loss_params = LSIZE
            loss_class = getattr(moment_tta_losses, loss_name)

        self.logger.info('loss_name:{}'.format(loss_name))
        self.logger.info('Loss params:{}'.format(loss_params))
        if loss_class is not None:
            self.loss_fn = loss_class(**loss_params)

    def init_from_source_model(self):
        assert isfile(args.pretrained_model), 'missing model checkpoint!'
        self.pretrained_params = torch.load(args.pretrained_model)
        if self.args.arch == 'unet_2d':
            self.model = UNet()
            self.model.load_state_dict(self.pretrained_params['model_state_dict'])
            self.model = self.model.to(self.device)
            if self.args.only_bn_updated:
                self.model = self.configure_model_2d()
                self.logger.info('normalization statistics updated')
            else:
                self.logger.info('WARNING all var updated')

            self.amp_grad_scaler = GradScaler()
            
            
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
#         else:
#             raise NotImplementedError(f"Architecture '{self.args.arch}' not implemented.")
            
#     def configure_model_2d(self):
#         """Configure model for TTA."""
#         self.model.train()  # TTA通常在训练模式下进行（例如，为了BN的批统计或dropout）
#         self.model.requires_grad_(False)  # 首先冻结所有参数
        
#         #1.解冻BN层
#         for m in self.model.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.requires_grad_(True)
                
#         trainable_components = []
        
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
#             self.model = self.configure_model_2d()

#         else:
#             NotImplementedError('Not implemented optimizing step for this architecture')

#         self.init_optimizer_and_scheduler()

#     def configure_model_2d(self):
#         """Configure model for use with tent."""
#         # train mode, because tent optimizes the model to minimize entropy
#         # self.model.train()
#         # disable grad, to (re-)enable only what tent updates
#         # self.model.requires_grad_(False)
#         # # configure norm for tent updates: enable grad + force batch statisics
#         for m in self.model.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.requires_grad_(True)
#                 # force use of batch stats in train and eval modes
#                 # m.track_running_stats = False
#                 # m.running_mean = None
#                 # m.running_var = None

#         for name, param in self.model.named_parameters():
#             if name.find('data') != -1 or name.find('prompt') != -1 :
#                 param.requires_grad = True

#         return self.model


    def init_dataloader(self):
        tr_img_list, tr_label_list = convert_unlabeled_list( self.args.tr_csv, r=1)
        if tr_label_list is not None:
            self.logger.info('-----------Train:img-{}-label-{} -----------'.format(len(tr_img_list), len(tr_label_list)))
        else:
            self.logger.info('-----------Train:img-{}-----------'.format(len(tr_img_list)))
        tr_dataset = RIGA_unlabeled_set( self.args.root, tr_img_list, self.args.patch_size)
        ts_img_list, ts_label_list = convert_labeled_list(self.args.ts_csv, r=1)
        self.logger.info('-----------Test:img-{}-label-{} -----------'.format(len(ts_img_list), len(ts_label_list)))
        ts_dataset = RIGA_labeled_set( self.args.root, ts_img_list, ts_label_list,  self.args.patch_size)
        self.tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
                                                    batch_size= self.args.batch_size,
                                                    num_workers= self.args.num_threads,
                                                    shuffle= True,
                                                    pin_memory=True,
                                                    collate_fn=target_collate_fn_tr_fda)
        self.ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                    batch_size= self.args.batch_size,
                                                    num_workers= self.args.num_threads // 2,
                                                    shuffle=False,
                                                    pin_memory=True,
                                                    collate_fn=collate_fn_ts)

    def optimize_parameters(self, input, loss_fn=None):
        if args.model == 'BN-Sta-TTA':
            self.optimizer.zero_grad()
            pred, bn_f, _, _= self.model(input, training=True) 
            loss = bn_loss(self.model, self.pretrained_params, bn_f)
            loss.backward()
            self.optimizer.step()
            return pred, loss
        else:
            self.optimizer.zero_grad()
            pred = self.model(input)
            loss = loss_fn(pred)
            loss.backward()
            self.optimizer.step()
            return pred, loss

    def evaluate(self):
        start_epoch = 0
        if args.continue_training:
            assert isfile(join(self.model_folder, 'model_latest.model')), 'missing model checkpoint!'
            params = torch.load(join(self.model_folder, 'model_latest.model'))
            self.model.load_state_dict(params['model_state_dict'])
            self.optimizer.load_state_dict(params['optimizer_state_dict'])
            start_epoch = params['epoch']
        self.logger.info('start epoch: {}'.format(start_epoch))
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                print('Trainable', k)
        for epoch in range(start_epoch, args.num_epochs):
            self.do_disc_cup_seg_one_epoch(epoch, self.loss_fn)
            saved_model = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict()
            }
            print('  Saving model_{}.model...'.format('final'))
            torch.save(saved_model, join(self.model_folder, 'model_final.model'))
           

        # inference
        from inference.inference_nets.inference_tta import inference
        for ts_csv_path in self.ts_csv:
            inference_tag = split_path(ts_csv_path)[-1].replace('.csv', '')
            self.logger.info("Running inference: {}".format(inference_tag))
            inference(args, self.model, self.device, self.log_folder, [ts_csv_path],
                      inference_tag, self.logger)

    def set_train_state(self):
        self.model.train()

    def do_disc_cup_seg_one_epoch(self, epoch, loss_fn):
        """
        Use another unlabelled shuffled ts datasets and test on the unshuffled tr dataset
         """
        self.logger.info('Epoch {}:'.format(epoch))
        start_epoch = time()
        self.set_train_state()
        lr = adjust_learning_rate(self.optimizer, epoch, self.args.initial_lr, 20)
        self.logger.info('  lr: {}'.format(lr))

        train_loss_list = list()
        for iter, batch in enumerate(self.tr_dataloader):
            data = torch.from_numpy(normalize_image(batch['data'])).cuda().to(dtype=torch.float32)
            output, loss = self.optimize_parameters(data, loss_fn)
            train_loss_list.append(loss.detach().cpu().numpy())
        mean_tr_loss = np.mean(train_loss_list)
        self.writer.add_scalar("Train Scalars/Learning Rate", lr, epoch)
        self.writer.add_scalar("Train Scalars/Train Loss", mean_tr_loss, epoch)
        self.logger.info('  Tr loss: {}\n'.format(mean_tr_loss))

        val_disc_dice_list = list()
        val_cup_dice_list = list()
        with torch.no_grad():
            self.model.eval()
            for iter, batch in enumerate(self.ts_dataloader):
                data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
                seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
                with autocast():
                    output = self.model(data)
                output_sigmoid = torch.sigmoid(output)
                val_disc_dice_list.append(get_hard_dice(output_sigmoid[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0))
                val_cup_dice_list.append(get_hard_dice(output_sigmoid[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0))
           
            
        mean_val_disc_dice = np.mean(val_disc_dice_list)
        mean_val_cup_dice = np.mean(val_cup_dice_list)


        self.writer.add_scalar("Val Scalars/Disc Dice", mean_val_disc_dice, epoch)
        self.writer.add_scalar("Val Scalars/Cup Dice", mean_val_cup_dice, epoch)
        self.writer.add_scalar("Val Scalars/Class-Avg Dice", (mean_val_cup_dice+mean_val_disc_dice)/2, epoch)

   
        self.logger.info(' Val disc dice: {}; Cup dice: {}'.format( mean_val_disc_dice, mean_val_cup_dice))

        time_per_epoch = time() - start_epoch
        self.logger.info('  Durations: {}'.format(time_per_epoch))
        self.writer.add_scalar("Time/Time per epoch", time_per_epoch, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="Moment-TTA", required=False,choices=['Moment-TTA', 'Tent-TTA', 'RN-CR-TTA', "BN-Sta-TTA"],
                        help='Model name.')
    parser.add_argument('--arch', default="unet_2d_sptta_v11_freq_attention", required=False,
                        help='Network architecture.')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], required=False,
                        help='Device id.')
    parser.add_argument('--num_classes', default=2, required=False,
                        help='Num of classes.')
    parser.add_argument('--manualseed', type=int, default=47, required=False,
                        help='random seed.')
    parser.add_argument('--log_folder', default='log_dir', required=False,
                        help='Log folder.')
    parser.add_argument('--only_bn_updated', default=False, required=False,
                        help='which part to be updated.')
    parser.add_argument('--tag', default="Base2_test_sptta", required=False,
                        help='Run identifier.')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[512, 512], required=False,
                        help='patch size.')
    parser.add_argument('--batch_size', type=int, default=8, required=False,
                        help='batch size.')
    parser.add_argument('--initial_lr', type=float, default=5e-3, required=False,
                        help='initial learning rate.')
    parser.add_argument('--optimizer', type=str, default='sgd', required=False,
                        help='optimizer method.')
    parser.add_argument('-c', '--continue_training', default=False, required=False, action='store_true',
                        help="restore from checkpoint and continue training.")
    parser.add_argument('--num_threads', type=int, default=0, required=False,
                        help="Threads number of dataloader.")
    parser.add_argument('-r', '--root', default='data/RIGAPlus/', required=False,
                        help='dataset root folder.')
    parser.add_argument('--tr_csv', nargs='+',
                        required=False, default=['data/RIGAPlus/MESSIDOR_Base2_test.csv'], help='training csv file.')
    parser.add_argument('--ts_csv', nargs='+',
                        required=False, default=['data/RIGAPlus/MESSIDOR_Base2_test.csv'],
                        help='test csv file.')
    parser.add_argument('--num_epochs', type=int, default=20, required=False,
                        help='num_epochs.')

    parser.add_argument('--pretrained_model', default='log_dir/UNet_Source_RIGA/checkpoints/model_best.model',
                        required=False, help='pretrained model path.')
    parser.add_argument('--alpha', type=float, default=0.01, required=False,
                        help='alpha in BN loss.')
    parser.add_argument('--layers', type=int, default=5, required=False,
                        help='layers to calculate bn loss.')
    parser.add_argument('--gamma', type=float, default=0.01, required=False,
                        help='gamma in feature alignment loss.')
    parser.add_argument("--norm_layer", type=str, default='batch_norm', help=
                        "select which normalization layer to be used in the model")
    
    parser.add_argument('--transformer_dim', type=int, default=512, help='Dimension for Transformer.')
    parser.add_argument('--transformer_heads', type=int, default=8, help='Number of heads for Transformer.')
    parser.add_argument('--transformer_mlp_dim', type=int, default=1024, help='MLP dimension for Transformer.')
    parser.add_argument('--transformer_layers', type=int, default=7, help='Number of layers for Transformer.')
    parser.add_argument('--transformer_dropout', type=float, default=0.1, help='Dropout for Transformer.')
    
    parser.add_argument('--use_freq_adaptor', type=lambda x: (str(x).lower() == 'true'), default=True, help='Whether to use the frequency domain adaptor.')
    parser.add_argument('--bofea_freq_adaptor_channels', type=int, default=512, help='Channels for freq adaptor input (usually 3 for RGB).')
    parser.add_argument('--bofea_freq_adaptor_attn_dim', type=int, default=64, help='Attention dimension in freq adaptor.')
    parser.add_argument('--bofea_freq_adaptor_heads', type=int, default=1, help='Attention heads in freq adaptor.')
    parser.add_argument('--bofea_freq_adaptor_mlp_factor', type=int, default=1, help='MLP factor in freq adaptor attention.')
    parser.add_argument('--bofea_freq_adaptor_dropout', type=float, default=0.1, help='Dropout in freq adaptor.')
    parser.add_argument('--bofea_freq_transformer_layers', type=int, default=3, help = 'bofea_freq_transformer_layers.')

    args = parser.parse_args()
    adpater = Offline_Adapter(args)
    adpater.evaluate()

