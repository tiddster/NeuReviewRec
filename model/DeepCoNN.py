import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DeepCoNNModel(nn.Module):
    '''
    deep conn 2017
    '''
    def __init__(self, opt, uori='user'):
        super(DeepCoNNModel, self).__init__()
        self.opt = opt
        self.num_feature = 1 # DOC

        self.user_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  # vocab_size * 300
        self.item_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  # vocab_size * 300

        self.user_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))
        self.item_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))

        self.user_fc_linear = nn.Linear(opt.filters_num, opt.fc_dim)
        self.item_fc_linear = nn.Linear(opt.filters_num, opt.fc_dim)
        self.dropout = nn.Dropout(self.opt.drop_out)

        self.reset_para()

    def reset_para(self):

        # 初始化cnn参数
        for cnn in [self.user_cnn, self.item_cnn]:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        # 初始化线性层参数
        for fc in [self.user_fc_linear, self.item_fc_linear]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)

        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.user_word_embs.weight.data.copy_(w2v.cuda())
                self.item_word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.user_word_embs.weight.data.copy_(w2v)
                self.item_word_embs.weight.data.copy_(w2v)
        else:
            nn.init.uniform_(self.user_word_embs.weight, -0.1, 0.1)
            nn.init.uniform_(self.item_word_embs.weight, -0.1, 0.1)

    def forward(self, datas):
        _, _, user_ids, item_ids, _, _, user_doc, item_doc = datas

        user_doc = self.user_word_embs(user_doc).unsqueeze(1)
        item_doc = self.item_word_embs(item_doc).unsqueeze(1)

        user_feature = F.relu(self.user_cnn(user_doc)).squeeze(3)
        item_feature = F.relu(self.item_cnn(item_doc)).squeeze(3)

        user_feature = F.max_pool1d(user_feature, user_feature.size(2)).squeeze(2)
        item_feature = F.max_pool1d(item_feature, item_feature.size(2)).squeeze(2)
        user_feature = self.dropout(self.user_fc_linear(user_feature))
        item_feature = self.dropout(self.item_fc_linear(item_feature))

        return torch.stack([user_feature], 1), torch.stack([item_feature], 1)