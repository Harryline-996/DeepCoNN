import torch


class Config:
    # device = torch.device("cuda:0")
    device = torch.device("cpu")

    train_epochs = 20
    batch_size = 100
    learning_rate = 2e-3
    l2_regularization = 1e-6  # 权重衰减程度
    learning_rate_decay = 0.99  # 学习率衰减程度

    review_count = 10  # max review count
    review_length = 40  # max review length
    lowest_review_count = 2  # reviews wrote by a user/item will be delete if its amount less than such value.
    PAD_WORD = '</s>'

    kernel_count = 100
    kernel_size = 3

    dropout_prob = 0.5

    cnn_out_dim = 50  # CNN输出维度
