import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class MPCN(nn.Module):
    '''
    Multi-Pointer Co-Attention Network for Recommendation
    WWW 2018
    '''
    def __init__(self, opt, head=3):
        '''
        head: the number of pointers
        '''
        super(MPCN, self).__init__()

        self.opt = opt
        self.num_fea = 1  # ID + DOC
        self.head = head
        self.user_word_embedding = nn.Embedding(opt.vocab_size, opt.word_dim)  # vocab_size * 300
        self.item_word_embedding = nn.Embedding(opt.vocab_size, opt.word_dim)  # vocab_size * 300

        # review gate
        self.user_word_fc = nn.Linear(opt.word_dim, opt.word_dim)
        self.item_word_fc = nn.Linear(opt.word_dim, opt.word_dim)

        # multi points
        self.review_co_att = nn.ModuleList([Co_Attention(opt.word_dim, gumbel=True, pooling='max') for _ in range(head)])
        self.word_co_att = nn.ModuleList([Co_Attention(opt.word_dim, gumbel=False, pooling='avg') for _ in range(head)])

        # final fc
        self.user_fc = self.fc_layer()
        self.item_fc = self.fc_layer()

        self.drop_out = nn.Dropout(opt.drop_out)
        self.reset_para()

    def fc_layer(self):
        return nn.Sequential(
            nn.Linear(self.opt.word_dim * self.head, self.opt.word_dim),
            nn.ReLU(),
            nn.Linear(self.opt.word_dim, self.opt.id_emb_size)
        )

    def forward(self, datas):
        '''
        user_reviews, item_reviews, uids, iids, \
        user_item2id, item_user2id, user_doc, item_doc = datas
        :user_reviews: B * L1 * N
        :item_reviews: B * L2 * N
        '''
        user_reviews, item_reviews, _, _, _, _, _, _ = datas

        # ------------------review-level co-attention ---------------------------------
        user_word_embs = self.user_word_embedding(user_reviews)
        item_word_embs = self.item_word_embedding(item_reviews)
        user_reviews = self.review_gate(user_word_embs)
        item_reviews = self.review_gate(item_word_embs)
        user_feature = []
        item_feature = []
        for i in range(self.head):
            review_co_att = self.review_co_att[i]
            word_co_att = self.word_co_att[i]

            # ------------------review-level co-attention ---------------------------------
            p_u, p_i = review_co_att(user_reviews, item_reviews)             # B * L1 * 1

            # ------------------word-level co-attention ---------------------------------
            u_r_words = user_reviews.permute(0, 2, 1).float().bmm(p_u)   # (B * N * L1) X (B * L1 * 1)
            i_r_words = item_reviews.permute(0, 2, 1).float().bmm(p_i)   # (B * N * L2) X (B * L2 * 1)
            u_words = self.user_word_embedding(u_r_words.squeeze(2).long())  # B * N * d
            i_words = self.item_word_embedding(i_r_words.squeeze(2).long())  # B * N * d

            p_u, p_i = word_co_att(u_words, i_words)                 # B * N * 1
            u_w_fea = u_words.permute(0, 2, 1).bmm(p_u).squeeze(2)
            i_w_fea = u_words.permute(0, 2, 1).bmm(p_i).squeeze(2)
            user_feature.append(u_w_fea)
            item_feature.append(i_w_fea)

        user_feature = torch.cat(user_feature, 1)
        item_feature = torch.cat(item_feature, 1)

        user_feature = self.drop_out(self.user_fc(user_feature))
        item_feature = self.drop_out(self.item_fc(item_feature))

        return torch.stack([user_feature], 1), torch.stack([item_feature], 1)

    def review_gate(self, reviews):
        # Eq 1
        reviews = reviews.sum(2)
        return torch.sigmoid(self.user_word_fc(reviews)) * torch.tanh(self.item_word_fc(reviews))

    def reset_para(self):
        for fc in [self.user_word_fc, self.item_word_fc, self.user_fc[0], self.user_fc[-1], self.item_fc[0], self.item_fc[-1]]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.uniform_(fc.bias, -0.1, 0.1)

        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.user_word_embedding.weight.data.copy_(w2v.cuda())
                self.item_word_embedding.weight.data.copy_(w2v.cuda())
            else:
                self.user_word_embedding.weight.data.copy_(w2v)
                self.item_word_embedding.weight.data.copy_(w2v)
        else:
            nn.init.uniform_(self.user_word_embedding.weight, -0.1, 0.1)
            nn.init.uniform_(self.item_word_embedding.weight, -0.1, 0.1)


class Co_Attention(nn.Module):
    '''
    review-level and word-level co-attention module
    Eq (2,3, 10,11)
    '''
    def __init__(self, dim, gumbel, pooling):
        super(Co_Attention, self).__init__()
        self.gumbel = gumbel
        self.pooling = pooling

        self.M = nn.Parameter(torch.randn(dim, dim))
        self.fc_user = nn.Linear(dim, dim)
        self.fc_item = nn.Linear(dim, dim)

        self.reset_para()

    def reset_para(self):
        nn.init.xavier_uniform_(self.M, gain=1)
        nn.init.uniform_(self.fc_user.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_user.bias, -0.1, 0.1)
        nn.init.uniform_(self.fc_item.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_item.bias, -0.1, 0.1)

    def forward(self, user_feature, item_feature):
        '''
        u_fea: B * L1 * d
        i_fea: B * L2 * d
        return:
        B * L1 * 1
        B * L2 * 1
        '''
        user_fc_out = self.fc_user(user_feature)
        item_fc_out = self.fc_item(item_feature)

        S = (user_fc_out @ self.M).bmm(item_fc_out.permute(0, 2, 1))  #  B * L1 * L2 Eq(2/10), we transport item instead user

        if self.pooling == 'max':
            user_score = S.max(2)[0]  # B * L1
            item_score = S.max(1)[0]  # B * L2
        else:
            user_score = S.mean(2)  # B * L1
            item_score = S.mean(1)  # B * L2

        if self.gumbel:
            p_u = F.gumbel_softmax(user_score, hard=True, dim=1)
            p_i = F.gumbel_softmax(item_score, hard=True, dim=1)
        else:
            p_u = F.softmax(user_score, dim=1)
            p_i = F.softmax(item_score, dim=1)
        return p_u.unsqueeze(2), p_i.unsqueeze(2)