# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from models.ops.modules import MSDeformAttn
from models.positional_encoding import build_position_encoding

class DEER_Encoder(nn.Module): 
    def __init__(self,d_model=256,dim_feedforward=1024,dropout=0.1,activation="relu",num_feature_levels=4,nhead=8,enc_n_points=4,num_encoder_layers=6)  -> None:
        super().__init__()
        self.hidden_dim = d_model
        self.fc1 = nn.Linear(256, d_model)
        self.fc2 = nn.Linear(512 , d_model)
        self.fc3 = nn.Linear(768 , d_model)
        self.fc4 = nn.Linear(1024 , d_model)
        self.gn1 = nn.GroupNorm(32, d_model)
        self.gn2 = nn.GroupNorm(32, d_model)
        self.gn3 = nn.GroupNorm(32, d_model)
        self.gn4 = nn.GroupNorm(32, d_model)

        self.level_embed = nn.Parameter(torch.Tensor(4,d_model))
        self.position_embedding = build_position_encoding(hidden_dim=self.hidden_dim,position_embedding="sine")
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
        
        self.transformer_encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
    
    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, feature_maps): 
        feature_tokens = []
        mask_flatten = []
        valid_ratios = []
        spatial_shapes = []
        level_start_index = [0]
        lvl_pos_embed_flatten = []

        for i, (key, feature_map) in enumerate(feature_maps.items()) :
            b, c, h, w = feature_map.size()
            spatial_shapes.append([h,w])

            # feature
            feature_flat = rearrange(feature_map, 'b c h w -> b (h w) c')
            fc_layer = getattr(self, f'fc{i+1}')
            gn_layer = getattr(self, f'gn{i+1}')
            feature_token = gn_layer(fc_layer(feature_flat).transpose(1,2)).transpose(1,2)
            feature_tokens.append(feature_token)
            mask = torch.zeros((b, h, w), dtype=torch.bool, device=feature_map.device)

            mask_flatten.append(mask.flatten(1))
            valid_ratios.append(self.get_valid_ratio(mask))

            # positional embedding
            if not self.training and i == 0:
                continue

            pos_embed = self.position_embedding(feature_token.reshape(b,self.hidden_dim,h,w), mask).to(feature_map.dtype)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[i].view(1,1,-1)        
            lvl_pos_embed_flatten.append(lvl_pos_embed)

            

 
        feature_tokens = torch.cat(feature_tokens, dim=1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        valid_ratios = torch.stack(valid_ratios, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feature_tokens.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # DETR Encoder
        refined_features = self.transformer_encoder(feature_tokens, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        return refined_features,level_start_index,spatial_shapes,valid_ratios
            
class DEER_Decoder(nn.Module): 
    def __init__(self,max_len,d_model=256,dim_feedforward=1024,dropout=0.1,activation="relu",
                 num_feature_levels=4,nhead=8,dec_n_points=4,
                 num_decoder_layers=6,tokenizer=None,
                 return_intermediate_dec=False)  -> None:
        super().__init__()
        self.hidden_dim = d_model
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                        dropout, activation,
                                                        num_feature_levels, nhead, dec_n_points)
        normal_decoder_layer = NormalTransformerDecoderLayer(d_model, dim_feedforward)
        self.transformer_decoder = DeformableTransformerDecoder(decoder_layer, normal_decoder_layer, num_decoder_layers, return_intermediate_dec)
        self.register_buffer('causal_mask', nn.Transformer(batch_first=True).generate_square_subsequent_mask(max_len-1))

        self.char_embedding = nn.Embedding(tokenizer.vocab_size, self.hidden_dim, padding_idx=tokenizer.pad_token_id)
        self.char_position_embedding = nn.Embedding(max_len, self.hidden_dim)
        self.char_classifier = nn.Linear(self.hidden_dim, tokenizer.vocab_size) # 32000 is tokenizer vocab size ( klue/roberta )

    def forward(self,labels,pref,refined_features,spatial_shapes,level_start_index,valid_ratios,mask_flatten,attn_mask=False) :
        _, G, S = labels.shape
        position_ids = torch.arange(S, dtype=torch.long, device=labels.device)
        position_ids = position_ids.unsqueeze(0).expand_as(labels)
        char_embed = self.char_embedding(labels)
        char_position_embed = self.char_position_embedding(position_ids)
        attn_mask = self.causal_mask if attn_mask else None

        hs,_ = self.transformer_decoder(char_embed,pref,
                                        refined_features,spatial_shapes,
                                        level_start_index,valid_ratios,char_position_embed,
                                        mask_flatten,attn_mask)

        return self.char_classifier(hs[-1])

class LocationHead(nn.Module) : 
    def __init__(self,hidden_dim) :
        super().__init__() 
        self.k = 50
        self.hidden_dim = hidden_dim

        self.in2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1, bias=True)
        self.out2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=True)

        # Probability map
        self.binarize = nn.Sequential( 
            nn.Conv2d(self.hidden_dim, self.hidden_dim // 4, 3, padding=1, bias=True),
            nn.GroupNorm(32,self.hidden_dim//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.hidden_dim//4, self.hidden_dim//4, 2, 2),
            nn.GroupNorm(32,self.hidden_dim//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.hidden_dim//4, 1, 2, 2),
            nn.Sigmoid()
            )
        
        # Treshold map
        self.thresh = nn.Sequential( 
            nn.Conv2d(self.hidden_dim, self.hidden_dim //
                      4, 3, padding=1, bias=True),
            nn.GroupNorm(32, self.hidden_dim//4),
            nn.ReLU(inplace=True),
            self._init_upsample(self.hidden_dim // 4, self.hidden_dim//4, smooth=False, bias=True),
            nn.GroupNorm(32, self.hidden_dim//4),
            nn.ReLU(inplace=True),
            self._init_upsample(self.hidden_dim // 4, 1, smooth=False, bias=True),
            nn.Sigmoid())

        self.binarize.apply(self.weights_init)
        self.thresh.apply(self.weights_init)

        self.in2.apply(self.weights_init)
        self.out2.apply(self.weights_init)
        
    def weights_init(self, m) -> None:
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('GroupNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
            
    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    # approximate binary map
    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
    
    def forward(self,refined_features,feature_maps,level_start_index) :
        b,c,h,w = feature_maps["stage2"].size()

        location_features = refined_features[:,level_start_index[0].item():level_start_index[1].item()].permute(0,2,1).contiguous().reshape(b,c,h,w)
        p2 = self.out2(self.in2(location_features + feature_maps["stage2"]))

        binary = self.binarize(p2)

        if self.training:
            thresh = self.thresh(p2)
            thresh_binary = self.step_function(binary, thresh)
            db_maps = {
                'binary' : binary,
                'thresh' : thresh,
                'thresh_binary' : thresh_binary,
            }
        else:
            db_maps = {
                'binary' : binary
            }

        return db_maps

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        if self.training:
            # self attention
            src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, None)
            src = src + self.dropout1(src2)
            src = self.norm1(src)

            # ffn
            src = self.forward_ffn(src)
            if torch.isinf(src).any() or torch.isnan(src).any():
                clamp_value = torch.finfo(src.dtype).max - 1000
                src = torch.clamp(src, min=-clamp_value, max=clamp_value)

        else:
            # self attention
            delete_stride4 = src[:,level_start_index[1]:]

            src2 = self.self_attn(self.with_pos_embed(delete_stride4, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
            src[:, -src2.size(1):, :] = src[:, -src2.size(1):, :] + self.dropout1(src2)
            # expanded_src2 = torch.zeros_like(src)
            # expanded_src2[:, -src2.size(1):, :] = self.dropout1(src2)
            # src = src + expanded_src2
            src = self.norm1(src)

            # ffn
            src = self.forward_ffn(src)

        return src

class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device, is_training):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            if not is_training and lvl == 0:
                continue            
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, src.device, self.training)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output
 
class NormalTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu", n_heads=8):
        super().__init__()

        # cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None, attn_mask=None):
        # self attention
        # tgt shape : (Batch, GT, Seq, Dim)
        B,G,S,D = tgt.shape
        tgt, query_pos = tgt.view(B*G,S,D), query_pos.view(B*G,S,D)
        q = k = self.with_pos_embed(tgt, query_pos) # (batch, seq, d_model)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1),attn_mask=attn_mask)[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.cross_attn(self.with_pos_embed(tgt,query_pos).reshape(B,G*S,D).transpose(0, 1),
                               src.transpose(0, 1),
                               src.transpose(0, 1))[0].transpose(0, 1).reshape(B*G,S,D)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt.view(B,G,S,D)

class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None, attn_mask=None):
        # self attention
        # tgt shape : (Batch, GT, Seq, Dim)
        # query -> char_pos
        B,G,S,D = tgt.shape
        tgt, query_pos = tgt.view(B*G,S,D), query_pos.view(B*G,S,D)
        q = k = self.with_pos_embed(tgt, query_pos) # (batch, seq, d_model)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), attn_mask=attn_mask)[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
 
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos).reshape(B,G*S,D),
                        reference_points.reshape(B*G, 1, 4, 2).repeat(1,S,1,1).reshape(B,G*S,4,2),
                        src, src_spatial_shapes, level_start_index, None).reshape(B*G,S,D)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt.view(B,G,S,D)


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, normal_decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = decoder_layer if i % 2 == 0 else normal_decoder_layer
            self.layers.append(copy.deepcopy(layer))
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, attn_mask=None):
        output = tgt

        intermediate = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask, attn_mask)

            intermediate.append(output)

        return intermediate, reference_points

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)