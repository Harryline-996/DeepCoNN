import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


class DeepCoNNDataset(Dataset):
    def __init__(self, data_path, word_dict, config, retain_rui=True):
        self.word_dict = word_dict
        self.config = config
        self.retain_rui = retain_rui  # 是否在最终样本中，保留user和item的公共review
        self.PAD_WORD_idx = self.word_dict[config.PAD_WORD]
        self.review_length = config.review_length
        self.review_count = config.review_count
        self.lowest_r_count = config.lowest_review_count  # lowest amount of reviews wrote by exactly one user/item

        df = pd.read_csv(data_path, header=None, names=['userID', 'itemID', 'review', 'rating'])
        df['review'] = df['review'].apply(self._review2id)  # 分词->数字
        self.sparse_idx = set()  # 暂存稀疏样本的下标，最后删除他们
        user_reviews = self._get_reviews(df)  # 收集每个user的评论列表
        item_reviews = self._get_reviews(df, 'itemID', 'userID')
        rating = torch.Tensor(df['rating'].to_list()).view(-1, 1)

        self.user_reviews = user_reviews[[idx for idx in range(user_reviews.shape[0]) if idx not in self.sparse_idx]]
        self.item_reviews = item_reviews[[idx for idx in range(item_reviews.shape[0]) if idx not in self.sparse_idx]]
        self.rating = rating[[idx for idx in range(rating.shape[0]) if idx not in self.sparse_idx]]

    def __getitem__(self, idx):
        return self.user_reviews[idx], self.item_reviews[idx], self.rating[idx]

    def __len__(self):
        return self.rating.shape[0]

    def _get_reviews(self, df, lead='userID', costar='itemID'):
        # 对于每条训练数据，生成用户的所有评论汇总
        reviews_by_lead = dict(list(df[[costar, 'review']].groupby(df[lead])))  # 每个user/item评论汇总
        lead_reviews = []
        for idx, (lead_id, costar_id) in enumerate(zip(df[lead], df[costar])):
            df_data = reviews_by_lead[lead_id]  # 取出lead的所有评论：DataFrame
            if self.retain_rui:
                reviews = df_data['review'].to_list()  # 取lead所有评论：列表
            else:
                reviews = df_data['review'][df_data[costar] != costar_id].to_list()  # 不含lead与costar的公共评论
            if len(reviews) < self.lowest_r_count:
                self.sparse_idx.add(idx)
            reviews = self._adjust_review_list(reviews, self.review_length, self.review_count)
            lead_reviews.append(reviews)
        return torch.LongTensor(lead_reviews)

    def _adjust_review_list(self, reviews, r_length, r_count):
        reviews = reviews[:r_count] + [[self.PAD_WORD_idx] * r_length] * (r_count - len(reviews))  # 评论数量固定
        reviews = [r[:r_length] + [0] * (r_length - len(r)) for r in reviews]  # 每条评论定长
        return reviews

    def _review2id(self, review):  # 将一个评论字符串分词并转为数字
        if not isinstance(review, str):
            return []  # 貌似pandas的一个bug，读取出来的评论如果是空字符串，review类型会变成float
        wids = []
        for word in review.split():
            if word in self.word_dict:
                wids.append(self.word_dict[word])  # 单词映射为数字
            else:
                wids.append(self.PAD_WORD_idx)
        return wids


class CNN(nn.Module):

    def __init__(self, config, word_dim):
        super(CNN, self).__init__()

        self.kernel_count = config.kernel_count
        self.review_count = config.review_count

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=word_dim,
                out_channels=config.kernel_count,
                kernel_size=config.kernel_size,
                padding=(config.kernel_size - 1) // 2),  # out shape(new_batch_size, kernel_count, review_length)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, config.review_length)),  # out shape(new_batch_size,kernel_count,1)
            nn.Dropout(p=config.dropout_prob))

        self.linear = nn.Sequential(
            nn.Linear(config.kernel_count * config.review_count, config.cnn_out_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_prob))

    def forward(self, vec):  # input shape(new_batch_size, review_length, word2vec_dim)
        latent = self.conv(vec.permute(0, 2, 1))  # out(new_batch_size, kernel_count, 1) kernel count指一条评论潜在向量
        latent = self.linear(latent.reshape(-1, self.kernel_count * self.review_count))
        return latent  # out shape(batch_size, cnn_out_dim)


class FactorizationMachine(nn.Module):

    def __init__(self, p, k):  # p=cnn_out_dim
        super(FactorizationMachine, self).__init__()
        self.v = nn.Parameter(torch.zeros(p, k))
        self.linear = nn.Linear(p, 1, bias=True)

    def forward(self, x):
        linear_part = self.linear(x)  # input shape(batch_size, cnn_out_dim), out shape(batch_size, 1)
        inter_part1 = torch.mm(x, self.v)
        inter_part2 = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(inter_part1 ** 2 - inter_part2, dim=1)
        output = linear_part.transpose(1, 0) + 0.5 * pair_interactions
        return output.view(-1, 1)  # out shape(batch_size, 1)


class DeepCoNN(nn.Module):

    def __init__(self, config, word_emb):
        super(DeepCoNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
        self.cnn_u = CNN(config, word_dim=self.embedding.embedding_dim)
        self.cnn_i = CNN(config, word_dim=self.embedding.embedding_dim)
        self.fm = FactorizationMachine(config.cnn_out_dim * 2, 10)

    def forward(self, user_review, item_review):  # input shape(batch_size, review_count, review_length)
        new_batch_size = user_review.shape[0] * user_review.shape[1]
        user_review = user_review.reshape(new_batch_size, -1)
        item_review = item_review.reshape(new_batch_size, -1)

        u_vec = self.embedding(user_review)
        i_vec = self.embedding(item_review)

        user_latent = self.cnn_u(u_vec)
        item_latent = self.cnn_i(i_vec)

        concat_latent = torch.cat((user_latent, item_latent), dim=1)
        prediction = self.fm(concat_latent)
        return prediction
