#!/usr/bin/env python
# coding:utf8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(123456)

"""

CNN for Stance Classification

"""


class CNN(nn.Module):
    """
        Implementation of AoA for Stance Classification Task
        Procedure:
        1. sen   : CNN encoder
        2. Ask   : CNN encoder

        - Lookup layer
        - Convolution Layer
        - Activation Layer
        - Max-pooling Layer
        - Voting Scheme (may not be used in this case)

    """
    def __init__(self, embeddings, input_dim, hidden_dim, num_layers, output_dim, max_len=40, dropout=0.5):
        super(CNN, self).__init__()#将子类的参数传递给父类

        self.emb = nn.Embedding(num_embeddings=embeddings.size(0), #第一个维度是词表长度
                                embedding_dim=embeddings.size(1), #第二个维度是指嵌入到多少维
                                padding_idx=0)
        #embeds = nn.Embedding(2, 5) 这里的2表示有2个词，5表示5维度，其实也就是一个2x5的矩阵，所以如果你有1000个词，每个词希望是100维，你就可以这样建立一个word embedding，nn.Embedding(1000, 100)。
        self.emb.weight = nn.Parameter(embeddings)

        self.input_dim = input_dim
        self.output_dim = output_dim

        '''
        Convolution
            25 * 50, kernel(3, 50), out_map(16) --> (25-3+1) * 16
        Max-Pooling
            23 * 16 --> 1 * 16
        Fully-connected
            16 --> 50
        '''
        self.sen_len = max_len
        self.sen_conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, input_dim)) 
        self.sen_fc1 = nn.Linear(16, 50)

        self.output = nn.Linear(50, output_dim)

    def forward(self, sen_batch, sen_lengths, sen_mask_matrix):
        """

        :param sen_batch:
        :param sen_lengths:
        :param sen_mask_matrix:
        :return:
        """
        ''' Embedding Layer | Padding | Sequence_length 25/45'''
        sen_batch = self.emb(sen_batch)

        batch_size = len(sen_batch)

        sen_batch = sen_batch.view(batch_size, 1, self.sen_len, self.input_dim)
        sen_batch = F.relu(self.sen_conv1(sen_batch))
        sen_batch = sen_batch.view(batch_size, 16, -1)
        sen_batch = F.max_pool2d(sen_batch, (1, self.sen_len-3+1))
        sen_batch = self.sen_fc1(sen_batch.view(batch_size, -1))

        # print ask_batch.size(), sen_batch.size()
        representation = sen_batch
        representation = self.output(representation)
        out_scores = F.softmax(representation)

        return out_scores



