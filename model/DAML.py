import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DAMLModel(nn.Module):
    def __init__(self, opt):
        self.opt = opt
        self.num_feature = 2
        self.user_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.item_word_embs = nn.Embedding(opt.vacab_size, opt.word_dim)

        self.word_cnn = nn.Conv2d(1, 1, (5, opt.word_dim), padding=(2, 0))

        # document-level cnn
        self.user_doc_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim), padding=(1, 0))
        self.item_doc_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim), padding=(1, 0))
        # abstract-level cnn
        self.user_abs_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.filters_num))
        self.item_abs_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.filters_num))

        self.unfold = nn.Unfold((3, opt.filters_num), padding=(1,0))

        self.user_fc = nn.Linear(opt.filters_num, opt.id_emb_size)
        self.item_fc = nn.Linear(opt.filters_num, opt.id_emb_size)

        self.user_id_embs = nn.Embedding(opt.user_num + 2, opt.id_emb_size)
        self.item_id_embs = nn.Embedding(opt.item_num + 2, opt.id_emb_size)

        self.reset_para()

    # 设置参数
    def reset_para(self):

        cnns = [self.word_cnn, self.user_doc_cnn, self.item_doc_cnn, self.user_abs_cnn, self.item_abs_cnn]

        for cnn in cnns:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.uniform_(cnn.bias, -0.1, 0.1)

        fcs = [self.user_fc, self.item_fc]
        for fc in fcs:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)

        nn.init.uniform_(self.uid_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.iid_embedding.weight, -0.1, 0.1)

        w2v = torch.from_numpy(np.load(self.opt.w2v_path))
        self.user_word_embs.weight.data.copy_(w2v.cuda())
        self.item_word_embs.weight.data.copy_(w2v.cuda())

    def local_attention_cnn(self, word_embs, doc_cnn):
        '''
        :Eq1 - Eq7
        '''
        local_att_words = self.word_cnn(word_embs.unsqueeze(1))
        local_word_weight = torch.sigmoid(local_att_words.squeeze(1))
        word_embs = word_embs * local_word_weight
        doc_feature = doc_cnn(word_embs.unsqueeze(1))
        return doc_feature

    def local_pooling_cnn(self, feature, attention, cnn, fc):
        '''
        :Eq11 - Eq13
        feature: (BATCH, 100, DOC_LEN ,1)
        attention: (BATCH, DOC_LEN)
        '''
        bs, n_filters, doc_len, _ = feature.shape
        feature = feature.permute(0, 3, 2, 1)  # bs * 1 * doc_len * embed
        attention = attention.reshape(bs, 1, doc_len, 1)  # bs * doc
        pools = feature * attention
        pools = self.unfold(pools)
        pools = pools.reshape(bs, 3, n_filters, doc_len)
        pools = pools.sum(dim=1, keepdims=True)  # bs * 1 * n_filters * doc_len
        pools = pools.transpose(2, 3)  # bs * 1 * doc_len * n_filters

        abs_feature = cnn(pools).squeeze(3)  # ? (DOC_LEN-2), 100
        abs_feature = F.avg_pool1d(abs_feature, abs_feature.size(2))  # ? 100
        abs_feature = F.relu(fc(abs_feature.squeeze(2)))  # ? 32

        return abs_feature

    def forward(self, datas):
        '''
        user_reviews, item_reviews, uids, iids, \
        user_item2id, item_user2id, user_doc, item_doc = datas
        '''
        _, _, uids, iids, _, _, user_doc, item_doc = datas

        # ------------------ review encoder ---------------------------------
        user_word_embs = self.user_word_embs(user_doc)
        item_word_embs = self.item_word_embs(item_doc)
        # (BS, 100, DOC_LEN, 1)
        user_local_feature = self.local_attention_cnn(user_word_embs, self.user_doc_cnn)
        item_local_feature = self.local_attention_cnn(item_word_embs, self.item_doc_cnn)

        # DOC_LEN * DOC_LEN
        euclidean = (user_local_feature - item_local_feature.permute(0, 1, 3, 2)).pow(2).sum(1).sqrt()
        attention_matrix = 1.0 / (1 + euclidean)
        # (?, DOC_LEN)
        user_attention = attention_matrix.sum(2)
        item_attention = attention_matrix.sum(1)

        # (?, 32)
        user_doc_feature = self.local_pooling_cnn(user_local_feature, user_attention, self.user_abs_cnn, self.user_fc)
        item_doc_feature = self.local_pooling_cnn(item_local_feature, item_attention, self.item_abs_cnn, self.item_fc)

        # ------------------ id embedding ---------------------------------
        user_id_emb_output = self.user_id_embs(uids)
        item_id_emb_output = self.item_id_embs(iids)

        use_fea = torch.stack([user_doc_feature, user_id_emb_output], 1)
        item_fea = torch.stack([item_doc_feature, item_id_emb_output], 1)

        return use_fea, item_fea

