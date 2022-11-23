import torch
import torch.nn as nn

class FusionLayer(nn.Module):
    def __init__(self, opt):
        super(FusionLayer, self).__init__()
        if opt.self_attn:
            self.attn = SelfAttention(opt.id_emb_size, opt.num_heads)
        self.opt = opt
        self.linear = nn.Linear(opt.feature_dim, opt.feature_dim)
        self.drop_out = nn.Dropout(0.5)

        # 在[-0.1, 0.1]之间初始化参数
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        # 用第二个参数填充第一个张量
        nn.init.constant_(self.linear.bias, 0.1)

    def forward(self, user_out, item_out):
        if self.opt.self_att:
            output = self.attn(user_out, item_out)
            self_attn_user_out, self_attn_item_out = torch.split(output, output.size(1)//2, dim=1)
            user_out = user_out + self_attn_user_out
            item_out = item_out + self_attn_item_out

        if self.opt.r_id_merge == 'cat':
            user_out = user_out.reshape(user_out.size(0), -1)
            item_out = item_out.reshape(item_out.size(0), -1)
        else:
            user_out = user_out.sum(1)
            item_out = item_out.sum(1)

        if self.opt.ui_merge == 'cat':
            output = torch.cat([user_out, item_out], 1)
        elif self.opt.ui_merge == 'add':
            output = user_out + item_out
        else:
            output = user_out * item_out
        return output


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(SelfAttention, self).__init__()
        self.encodeLayer = nn.TransformerEncoderLayer(dim, num_heads, 128, 0.4)
        self.encoder = nn.TransformerEncoder(self.encodeLayer ,1)

    def forward(self, user_feature, item_feature):
        feature = torch.cat([user_feature, item_feature], 1).permute(1, 0, 2)
        output = self.encoder(feature)
        return output