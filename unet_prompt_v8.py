# -*- coding:utf-8 -*-
# For the prompt construction
from torch import nn
import torch
from models.resnet import resnet34, resnet18, resnet50, resnet101, resnet152
import torch.nn.functional as F
from models.unet import SaveFeatures, UnetBlock, UNet
from einops import rearrange, repeat

# 保持 SpatialTransformerEncoderLayer 定义不变 (需要加入位置编码)
class SpatialTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim, sequence_length, dropout=0.1): # 增加了 sequence_length
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive for positional encoding.")
        self.pos_embedding = nn.Parameter(torch.randn(1, sequence_length, dim))


    def forward(self, x): # x: (B, S, C) where S is sequence_length
        b, s, c = x.shape
        if s != self.pos_embedding.shape[1]:
            # 简单的处理方式：如果序列长度不匹配，打印警告并尝试使用部分或重复位置编码
            # 更鲁棒的方式是使用可插值的位置编码或确保输入序列长度固定
            # print(f"Warning: Input sequence length {s} != pos_embedding length {self.pos_embedding.shape[1]}. Adjusting pos_emb.")
            if s < self.pos_embedding.shape[1]:
                pos_emb_to_add = self.pos_embedding[:, :s, :]
            else: # s > self.pos_embedding.shape[1] - 这种情况不应该发生，如果发生了，说明 pos_embedding 设置小了
                # 可以尝试 repeat_interleave 或者简单地截断x，但这会丢失信息
                # 这里我们假设 s <= self.pos_embedding.shape[1] 是通过外部保证的
                # 或者在初始化时确保 sequence_length 足够大
                pos_emb_to_add = self.pos_embedding # 期望 s == self.pos_embedding.shape[1]
                # raise ValueError(f"Input sequence length {s} > pos_embedding length {self.pos_embedding.shape[1]}. Increase sequence_length.")
        else:
            pos_emb_to_add = self.pos_embedding

        x_with_pos = x + pos_emb_to_add

        x_norm = self.norm1(x_with_pos)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)

        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + self.dropout(mlp_out)
        return x

class CrossAttention(nn.Module):
    def __init__(self, C, K):
        super(CrossAttention, self).__init__()
        # Q, K, V mapping

        self.query = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, groups=C)
        self.key = nn.Conv2d(K, K, kernel_size=3, stride=1, padding=1, groups=K)
        self.value = nn.Conv2d(K, K, kernel_size=3, stride=1, padding=1, groups=K)
    
    def forward(self, x_q, x_kv):
        query = self.query(x_q)
        key = self.key(x_kv)
        value = self.value(x_kv)

        batch_size, channels, height, width = query.size()
        batch_size, num_prompt, height, width = key.size()
        query = query.view(batch_size, channels, -1)
        key = key.view(batch_size,num_prompt, -1).permute(0, 2, 1)
        value = value.view(batch_size, num_prompt, -1)

        # calculate attention weights with scaled dot-product attention
        attn_weights = torch.matmul(query, key) / (num_prompt ** 0.5)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        # sparsity atten
        num_non_zero = int(0.1 * attn_weights.size(1))
        sorted_values, sorted_indices = torch.sort(attn_weights, descending=True, dim=1)
        mask = torch.zeros_like(attn_weights)
        mask.scatter_(1, sorted_indices[:, :num_non_zero], 1)
        attn_weights = attn_weights * mask

        # calculate attentive output
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.view(batch_size, channels, height, width)

        return x_q + attn_output, attn_weights
    
class FrequencyAttentionBlock(nn.Module):
    def __init__(self, channels, freq_attn_dim, n_heads, mlp_factor=2, 
                 sequence_length=None, dropout=0.1, 
                 num_transformer_layers=1): # 🔸 新增：Transformer 层数参数
        super().__init__()
        self.channels = channels
        self.freq_attn_dim = freq_attn_dim

        self.input_proj = nn.Conv2d(channels, freq_attn_dim, kernel_size=1) if channels != freq_attn_dim else nn.Identity()
        
        if sequence_length is None:
            print("Warning: sequence_length for FrequencyAttentionBlock not provided. It's crucial for pos_embedding.")
            # 这是一个示例值，你需要根据实际情况计算
            sequence_length_freq_default = (256 // 2 + 1) * 256 # 假设 H=256, W=256
            print(f"Using default sequence_length_freq_default = {sequence_length_freq_default} for FreqAttention")
            actual_sequence_length = sequence_length_freq_default
        else:
            actual_sequence_length = sequence_length

        if num_transformer_layers > 0:
            self.freq_transformer_layers = nn.ModuleList([ # 🔸 使用 ModuleList
                SpatialTransformerEncoderLayer(
                    dim=freq_attn_dim,
                    n_heads=n_heads,
                    mlp_dim=freq_attn_dim * mlp_factor,
                    sequence_length=actual_sequence_length, # 传递计算好的或默认的序列长度
                    dropout=dropout
                ) for _ in range(num_transformer_layers) # 🔸 根据层数创建
            ])
        else:
            self.freq_transformer_layers = None # 如果层数为0，则不使用Transformer

        self.output_proj = nn.Conv2d(freq_attn_dim, channels, kernel_size=1) if channels != freq_attn_dim else nn.Identity()
        # self.norm_out = nn.LayerNorm(channels) # (可选)

    def forward(self, x): # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        x_proj = self.input_proj(x)
        D_attn = x_proj.shape[1]

        x_freq = torch.fft.rfft2(x_proj, norm='ortho')
        
        # 简单处理：只对实部应用Transformer
        x_freq_real = x_freq.real
        x_freq_imag = x_freq.imag # 保留虚部
        
        real_flat = rearrange(x_freq_real, 'b d h w_f -> b (h w_f) d', d=D_attn, h=H, w_f=(W//2+1))
        
        attended_real_flat = real_flat # 默认值，如果不用Transformer
        if self.freq_transformer_layers is not None:
            current_attended_real_flat = real_flat
            for layer in self.freq_transformer_layers: # 🔸 遍历每一层 Transformer
                current_attended_real_flat = layer(current_attended_real_flat)
            attended_real_flat = current_attended_real_flat
        
        attended_real = rearrange(attended_real_flat, 'b (h w_f) d -> b d h w_f', d=D_attn, h=H, w_f=(W//2+1))
        
        attended_freq = torch.complex(attended_real, x_freq_imag)

        x_reconstructed = torch.fft.irfft2(attended_freq, s=(H, W), norm='ortho')
        
        x_out_proj = self.output_proj(x_reconstructed)
        
        output = x + x_out_proj # 残差连接
        return output
    
# --- CrossAttention 和 UNet 定义保持不变 ---
# --- SpatialTransformerEncoderLayer 需要有 sequence_length 参数 ---

class UNet_SPTTA_V11_FreqAttention(nn.Module):
    def __init__(self, pretrained_path, patch_size=(512, 512), resnet='resnet34', num_classes=2, pretrained=False,
                 transformer_dim=512, transformer_heads=8, transformer_mlp_dim=1024, transformer_layers=1, transformer_dropout=0.1,
                 use_freq_adaptor=True,
                 # freq_adaptor_channels=3, # 这个参数通常是固定的 (例如3 for RGB, or bottleneck_channels)
                                         # 或者也作为参数传入
                 # 新增傅里叶自适应模块的参数
                 bofea_freq_adaptor_channels=512, # 通常是 bottleneck_channels
                 bofea_freq_adaptor_attn_dim=128, # 提供默认值
                 bofea_freq_adaptor_heads=4,      # 提供默认值
                 bofea_freq_adaptor_mlp_factor=2, # 提供默认值
                 bofea_freq_adaptor_dropout=0.1,
                 bofea_freq_transformer_layers=1): # 提供默认值
        super().__init__()
        self.use_freq_adaptor = use_freq_adaptor
        self.patch_height, self.patch_width = patch_size

        # --- 原有的 prompt, CrossAttention, data_adaptor, structure_head, UNet, feature_transformer_encoder ---
        # Data Prompt
        self.data_prompt_channels = 512
        self.data_prompt_h = self.patch_height // 32
        self.data_prompt_w = self.patch_width // 32
        self.data_prompt = nn.Parameter(torch.zeros(1, self.data_prompt_channels, self.data_prompt_h, self.data_prompt_w))
        
        # Cross-Attention
        self.bottleneck_channels = 512
        self.prompt2feature = CrossAttention(C=self.bottleneck_channels, K=self.data_prompt_channels)

        # 空间域 Data Adaptor (同 V7)
        self.spatial_data_adaptor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1), nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1), nn.Tanh()
        )
        
        # 增强结构头 (同 V7)
        self.structure_head = nn.Sequential(
            nn.Conv2d(self.bottleneck_channels, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        # UNet Core (同 V7)
        self.unet = UNet(resnet=resnet, num_classes=num_classes, pretrained=pretrained)
        pretrained_params = torch.load(pretrained_path)
        self.unet.load_state_dict(pretrained_params['model_state_dict'])

        # 特征 Transformer (同 V7/V9 - 需要传入正确的 sequence_length)
        self.feature_transformer_layers_count = transformer_layers
        self.feature_sequence_length = self.data_prompt_h * self.data_prompt_w # 假设瓶颈特征图与prompt有相同H,W
        if transformer_layers > 0:
            self.feature_transformer_encoder = nn.ModuleList([
                SpatialTransformerEncoderLayer(dim=transformer_dim, n_heads=transformer_heads,
                                               mlp_dim=transformer_mlp_dim, 
                                               sequence_length=self.feature_sequence_length, # 传递
                                               dropout=transformer_dropout)
                for _ in range(transformer_layers)
            ])
            self.feat_to_transformer_proj = nn.Conv2d(self.bottleneck_channels, transformer_dim, 1) if self.bottleneck_channels != transformer_dim else nn.Identity()
            self.transformer_to_feat_proj = nn.Conv2d(transformer_dim, self.bottleneck_channels, 1) if self.bottleneck_channels != transformer_dim else nn.Identity()
        else:
            self.feature_transformer_encoder = None
        # --- 结束原有模块 ---

        # 确保 bofea_freq_adaptor_channels 与实际的 bottleneck_channels 一致
        if bofea_freq_adaptor_channels != self.bottleneck_channels:
            print(f"Warning: bofea_freq_adaptor_channels ({bofea_freq_adaptor_channels}) "
                  f"differs from self.bottleneck_channels ({self.bottleneck_channels}). Ensure consistency.")


        # 🔸 修改：频率域自适应模块作用于瓶颈特征 bo_fea
        if self.use_freq_adaptor:
            self.bo_fea_h = self.patch_height // 32 # 假设下采样32倍
            self.bo_fea_w = self.patch_width // 32 # 假设下采样32倍
            
            freq_domain_sequence_length_for_bofea = self.bo_fea_h * (self.bo_fea_w // 2 + 1)
            
            self.bottleneck_freq_adaptor = FrequencyAttentionBlock(
                channels=bofea_freq_adaptor_channels, # 使用传入的参数
                freq_attn_dim=bofea_freq_adaptor_attn_dim,      # 使用传入的参数
                n_heads=bofea_freq_adaptor_heads,               # 使用传入的参数
                mlp_factor=bofea_freq_adaptor_mlp_factor,       # 使用传入的参数
                sequence_length=freq_domain_sequence_length_for_bofea,
                dropout=bofea_freq_adaptor_dropout,            # 使用传入的参数
                num_transformer_layers=bofea_freq_transformer_layers # 🔸 传递层数
            )

        # bn_f 列表 (确保与你的 UNet 结构完全匹配)
        self.bn_f = [SaveFeatures(self.unet.rn[0]),
                     SaveFeatures(self.unet.rn[4][0].conv1), SaveFeatures(self.unet.rn[4][0].conv2),
                     SaveFeatures(self.unet.rn[4][1].conv1), SaveFeatures(self.unet.rn[4][1].conv2),
                     SaveFeatures(self.unet.rn[4][2].conv1), SaveFeatures(self.unet.rn[4][2].conv2),  # 7
                     SaveFeatures(self.unet.rn[5][0].conv1), SaveFeatures(self.unet.rn[5][0].conv2),
                     SaveFeatures(self.unet.rn[5][0].downsample[0]),
                     SaveFeatures(self.unet.rn[5][1].conv1), SaveFeatures(self.unet.rn[5][1].conv2),
                     SaveFeatures(self.unet.rn[5][2].conv1), SaveFeatures(self.unet.rn[5][2].conv2),
                     SaveFeatures(self.unet.rn[5][3].conv1), SaveFeatures(self.unet.rn[5][3].conv2),  # 16
                     SaveFeatures(self.unet.rn[6][0].conv1), SaveFeatures(self.unet.rn[6][0].conv2),
                     SaveFeatures(self.unet.rn[6][0].downsample[0]),
                     SaveFeatures(self.unet.rn[6][1].conv1), SaveFeatures(self.unet.rn[6][1].conv2),
                     SaveFeatures(self.unet.rn[6][2].conv1), SaveFeatures(self.unet.rn[6][2].conv2),
                     SaveFeatures(self.unet.rn[6][3].conv1), SaveFeatures(self.unet.rn[6][3].conv2),
                     SaveFeatures(self.unet.rn[6][4].conv1), SaveFeatures(self.unet.rn[6][4].conv2),
                     SaveFeatures(self.unet.rn[6][5].conv1), SaveFeatures(self.unet.rn[6][5].conv2),  # 29
                     SaveFeatures(self.unet.rn[7][0].conv1), SaveFeatures(self.unet.rn[7][0].conv2),
                     SaveFeatures(self.unet.rn[7][0].downsample[0]),
                     SaveFeatures(self.unet.rn[7][1].conv1), SaveFeatures(self.unet.rn[7][1].conv2),
                     SaveFeatures(self.unet.rn[7][2].conv1), SaveFeatures(self.unet.rn[7][2].conv2),  # 36
                     SaveFeatures(self.unet.up1.tr_conv), SaveFeatures(self.unet.up1.x_conv),
                     SaveFeatures(self.unet.up2.tr_conv),  SaveFeatures(self.unet.up2.x_conv),
                     SaveFeatures(self.unet.up3.tr_conv),  SaveFeatures(self.unet.up3.x_conv),
                     SaveFeatures(self.unet.up4.tr_conv),  SaveFeatures(self.unet.up4.x_conv),]
        if hasattr(self.unet, 'rn'):
            self.bn_f.append(SaveFeatures(self.unet.rn[0]))
            # ... (在这里填充与V7中完全相同的bn_f列表，确保其正确性) ...
            # 再次强调，这个列表的正确性对BN Loss至关重要
            if len(self.unet.rn) > 7 : # 仅为示例
                 self.bn_f.extend([
                     SaveFeatures(self.unet.rn[4][0].conv1), SaveFeatures(self.unet.rn[4][0].conv2),
                     SaveFeatures(self.unet.rn[4][1].conv1), SaveFeatures(self.unet.rn[4][1].conv2),
                     SaveFeatures(self.unet.rn[4][2].conv1), SaveFeatures(self.unet.rn[4][2].conv2),
                     SaveFeatures(self.unet.rn[5][0].conv1), SaveFeatures(self.unet.rn[5][0].conv2),
                     SaveFeatures(self.unet.rn[5][0].downsample[0]),
                     SaveFeatures(self.unet.rn[5][1].conv1), SaveFeatures(self.unet.rn[5][1].conv2),
                     SaveFeatures(self.unet.rn[5][2].conv1), SaveFeatures(self.unet.rn[5][2].conv2),
                     SaveFeatures(self.unet.rn[5][3].conv1), SaveFeatures(self.unet.rn[5][3].conv2),
                     SaveFeatures(self.unet.rn[6][0].conv1), SaveFeatures(self.unet.rn[6][0].conv2),
                     SaveFeatures(self.unet.rn[6][0].downsample[0]),
                     SaveFeatures(self.unet.rn[6][1].conv1), SaveFeatures(self.unet.rn[6][1].conv2),
                     SaveFeatures(self.unet.rn[6][2].conv1), SaveFeatures(self.unet.rn[6][2].conv2),
                     SaveFeatures(self.unet.rn[6][3].conv1), SaveFeatures(self.unet.rn[6][3].conv2),
                     SaveFeatures(self.unet.rn[6][4].conv1), SaveFeatures(self.unet.rn[6][4].conv2),
                     SaveFeatures(self.unet.rn[6][5].conv1), SaveFeatures(self.unet.rn[6][5].conv2),
                     SaveFeatures(self.unet.rn[7][0].conv1), SaveFeatures(self.unet.rn[7][0].conv2),
                     SaveFeatures(self.unet.rn[7][0].downsample[0]),
                     SaveFeatures(self.unet.rn[7][1].conv1), SaveFeatures(self.unet.rn[7][1].conv2),
                     SaveFeatures(self.unet.rn[7][2].conv1), SaveFeatures(self.unet.rn[7][2].conv2),
                     SaveFeatures(self.unet.up1.tr_conv), SaveFeatures(self.unet.up1.x_conv),
                     SaveFeatures(self.unet.up2.tr_conv),  SaveFeatures(self.unet.up2.x_conv),
                     SaveFeatures(self.unet.up3.tr_conv),  SaveFeatures(self.unet.up3.x_conv),
                     SaveFeatures(self.unet.up4.tr_conv),  SaveFeatures(self.unet.up4.x_conv),
                 ])


        for name, param in self.unet.named_parameters():
            param.requires_grad = False

    def forward(self, x, training=False):
        spatial_perturbation = self.spatial_data_adaptor(x) # 仍然作用于输入 x
        adapted_x_spatial = x + spatial_perturbation

        bo_fea_from_spatial_adapt = self.unet.encoder(adapted_x_spatial) # (B, C_bottle, H_bottle, W_bottle)
    
        # 🔸 应用频率域自适应到瓶颈特征
        processed_bo_fea = bo_fea_from_spatial_adapt
        if self.use_freq_adaptor and hasattr(self, 'bottleneck_freq_adaptor'):
            processed_bo_fea = self.bottleneck_freq_adaptor(bo_fea_from_spatial_adapt)
            # bottleneck_freq_adaptor 内部有残差连接，所以它输出的是处理后的 bo_fea

        current_batch_size = x.shape[0]
        repeated_data_prompt = self.data_prompt.repeat(current_batch_size, 1, 1, 1)
    
        # CrossAttention 使用 (可能经过频率处理的) processed_bo_fea
        adapted_bo_fea_after_crossattn, attn_weights = self.prompt2feature(x_q=processed_bo_fea, x_kv=repeated_data_prompt)
    
        final_features_for_decoder = adapted_bo_fea_after_crossattn
        if self.feature_transformer_encoder is not None: # 这里的 feature_transformer 作用于 CrossAttention 之后
            feat_for_transformer = self.feat_to_transformer_proj(adapted_bo_fea_after_crossattn)
            B_tf, C_tf, H_tf, W_tf = feat_for_transformer.shape # H_tf, W_tf 应该与 bo_fea 的 H,W 一致
        
            # 确保 feature_transformer_encoder 的 sequence_length 与 H_tf * W_tf 匹配
            # self.feature_sequence_length 应该在 __init__ 中基于 bo_fea 的 H,W (即 self.data_prompt_h, self.data_prompt_w) 计算
            if H_tf * W_tf != self.feature_sequence_length:
                 print(f"Warning: Mismatch in feature_transformer input seq len ({H_tf*W_tf}) and expected ({self.feature_sequence_length})")

            transformer_input = rearrange(feat_for_transformer, 'b c h w -> b (h w) c')
            for layer in self.feature_transformer_encoder:
                transformer_input = layer(transformer_input)
            transformer_output_fea = rearrange(transformer_input, 'b (h w) c -> b c h w', h=H_tf, w=W_tf)
            final_features_for_decoder = self.transformer_to_feat_proj(transformer_output_fea)

        structure_map = self.structure_head(final_features_for_decoder)
        output = self.unet.decoder(final_features_for_decoder)
    
        if training:
            # spatial_perturbation 是我们关心的输入扰动
            return output, self.bn_f, spatial_perturbation, attn_weights, structure_map, adapted_x_spatial
        else:
            return output, adapted_x_spatial, spatial_perturbation