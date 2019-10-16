import torch
import torch.nn as nn

class LSTM(torch.nn.Module):
    def __init__(self, seq_len, emb_dim, hidden_dim, output_dim, embedding, batch_size, max_vocab_size, num_layers=1,
                 dropout=0.2, bidirectional=False):
        super(LSTM, self).__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim  # glove dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding = embedding  # glove embedding
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        # Initalize look-up table and assign weight
        self.word_emb = torch.nn.Embedding(max_vocab_size, emb_dim)

        # Layers: one LSTM, one Fully-connected
        self.lstm = torch.nn.LSTM(emb_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, batch):
        #例如输入【200，32】数据，200是序列长度 seq_length, 32 是批大小batch_size,
        #经过 embedding 层，批数据中每个 token 都被替换为词向量，大小变为【200，32，100】即【seq_length, batch_size, emd_dim]
        x = self.word_emb(x)
        # 初始化 h_0
        h_0 = self._init_state(batch_size=batch)
        out, (h_t, c_t) = self.lstm(x, h_0)
        self.dropout(h_t)
        #取最后一个时刻的输出（形状为 [batch_size, hidden_dim]）传给线性层，将其映射到输出类别，
        y_pred = self.fc(h_t[-1])
        y_pred = F.log_softmax(y_pred)
        return y_pred

    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return (
            weight.new(self.num_layers, batch_size, self.hidden_dim).zero_(),
            weight.new(self.num_layers, batch_size, self.hidden_dim).zero_()
        )
