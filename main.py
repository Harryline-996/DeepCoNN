import os
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from gensim.models import KeyedVectors

from config import Config
from model import DeepCoNNDataset, DeepCoNN


def date(format='%Y-%m-%d %H:%M:%S'):
    return time.strftime(format, time.localtime())


def predict_mse(model, dataloader):
    mse, sample_count = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            user_reviews, item_reviews, ratings = map(lambda x: x.to(config.device), batch)
            predict = model(user_reviews, item_reviews)
            mse += F.mse_loss(predict, ratings, reduction='sum').item()
            sample_count += len(ratings)
    return mse / sample_count  # dataloader上的均方误差


def train(train_dataloader, valid_dataloader, model, config, model_path):
    print(f'{date()}## Start the training!')
    train_mse = predict_mse(model, train_dataloader)
    valid_mse = predict_mse(model, valid_dataloader)
    print(f'{date()}#### Initial train mse {train_mse:.6f}, validation mse {valid_mse:.6f}')
    start_time = time.perf_counter()

    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    torch.optim.lr_scheduler.ExponentialLR(opt, config.learning_rate_decay)

    best_loss, best_epoch = 100, 0
    for epoch in range(config.train_epochs):
        model.train()  # 将模型设置为训练状态
        total_loss, total_samples = 0, 0
        for batch in train_dataloader:
            user_reviews, item_reviews, ratings = map(lambda x: x.to(config.device), batch)
            predict = model(user_reviews, item_reviews)
            loss = F.mse_loss(predict, ratings, reduction='sum')  # 平方和误差
            opt.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播计算梯度
            opt.step()  # 根据梯度信息更新所有可训练参数

            total_loss += loss.item()
            total_samples += len(predict)

        model.eval()  # 停止训练状态
        valid_mse = predict_mse(model, valid_dataloader)
        train_loss = total_loss / total_samples
        print(f"{date()}#### Epoch {epoch:3d}; train mse {train_loss:.6f}; validation mse {valid_mse:.6f}")

        if best_loss > valid_mse:
            best_loss = valid_mse
            torch.save(model, model_path)

    end_time = time.perf_counter()
    print(f'{date()}## End of training! Time used {end_time - start_time:.0f} seconds.')


def test(dataloader, model_path):
    print(f'{date()}## Start the testing!')
    start_time = time.perf_counter()

    best_model = torch.load(model_path)
    test_loss = predict_mse(best_model, dataloader)

    end_time = time.perf_counter()
    print(f"{date()}## Test end, test ems is {test_loss:.6f}, time used {end_time - start_time:.0f} seconds.")


if __name__ == '__main__':
    config = Config()
    print(f'{date()}## Load embedding and data...')
    word2vec = KeyedVectors.load_word2vec_format('data/embedding/GoogleNews-vectors-negative300.bin', binary=True)
    word_emb = word2vec.vectors
    word_dict = {w: i.index for w, i in word2vec.vocab.items()}

    train_dataset = DeepCoNNDataset('data/music/train.csv', word_dict, config)
    valid_dataset = DeepCoNNDataset('data/music/valid.csv', word_dict, config, retain_rui=False)
    test_dataset = DeepCoNNDataset('data/music/test.csv', word_dict, config, retain_rui=False)
    train_dlr = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dlr = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True)
    test_dlr = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    model = DeepCoNN(config, word_emb).to(config.device)
    del train_dataset, valid_dataset, test_dataset, word2vec, word_emb, word_dict

    os.makedirs('model', exist_ok=True)  # 文件夹不存在则创建
    model_Path = 'model/best_model.pt'
    train(train_dlr, valid_dlr, model, config, model_Path)
    test(test_dlr, model_Path)
