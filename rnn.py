import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input, hidden):
        # 将input和之前的网络中的隐藏层参数合并。
        combined = torch.cat((input, hidden), 1)

        hidden = self.i2h(combined)  # 计算隐藏层参数
        output = self.i2o(combined)  # 计算网络输出的结果
        return output, hidden

    def init_hidden(self):
        # 初始化隐藏层参数hidden
        return torch.zeros(1, self.hidden_size)