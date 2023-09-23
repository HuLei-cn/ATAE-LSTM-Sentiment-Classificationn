import Tool_and_Layor
from Tool_and_Layor import *


class ATAE_LSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ATAE_LSTM, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = Dynamic_LSTM(opt.embedding_dim * 2, opt.hidden_dim,
                                 num_layers=opt.num_layer, batch_first=opt.batch_first)
        self.attention = NoQueryAttention(opt.hidden_dim + opt.embedding_dim, score_function='dot_product')
        self.dense = nn.Linear(opt.hidden_dim, opt.num_class)

    def forward(self, inputs):
        # 1_放入 model 之前进行的处理
        # 1.1_得到 x、aspect 的下标信息
        x_indices, aspect_indices = inputs['text'], inputs['aspect']
        x_len = torch.sum(x_indices != 0, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1).float()
        # 1.2_得到 x、aspect 的 embedding（注意 aspect 也可能是多个词）
        x = self.embed(x_indices)
        x = Tool_and_Layor.squeeze_sequence(x, x_len)
        aspect = self.embed(aspect_indices)

        # 2_model 部分
        # 2.1_拼接 x、aspect
        # 2.1.1_因为 aspect 可能是多个词，所以将 aspect 中每个词的 embedding 加起来然后求平均
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.unsqueeze(1))
        # 因为是用一个 sequence 合起来的 vector 除以这个 sequence 中包含着的 word 个数
        # 且在 torch.sum(aspect,dim=1) 之后，其 shape 为 batch × embedding_dim
        # 每个batch 中每一行即为一个 sequence 合起来的 vector，所以应该每行想要分别除以该 sequence
        # 的长度，需要做 aspect_len.unsqueeze(1) 操作，让其变成一个shape为 batch × 1 的长度向量
        # 示意图如下
        # [a00,a01,…,a0n]        [len(a0)]  len(a0)代表构成 a0 时，相加的 word vector 的个数
        # [a10,a11,…,a1n]   ÷    [len(a1)]
        # […………………………………]        […………………]
        # [am0,am1,…,amn]        [len(am)]
        # 2.1.2_扩充 aspect
        aspect = aspect_pool.unsqueeze(1).expand(-1, x_len_max, -1)
        # 2.1.3_拼接
        x = torch.cat((aspect, x), dim=-1)

        # 3_进入 LSTM 层处理
        h, (_, _) = self.lstm(x, x_len)
        # 4_进行 Attention 处理，得到各个 seq 的 representation
        ha = torch.cat((h, aspect), dim=-1)
        _, score = self.attention(ha)
        output = torch.squeeze(torch.bmm(score, h), dim=1)
        # 5_预测 -> 结果为向量形式（之后 softmax 等环节交给 CrossEntropyLoss
        out = self.dense(output)
        return out
