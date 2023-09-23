from torch import nn
import torch
import torch.nn.functional as F
import math


class Dynamic_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM'):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).
        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(Dynamic_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, x_len):
        """
        用 lstm 处理变长序列
        :param x: padding 之后的 input
        :param x_len: 记录 x 中每个 seqence 的长度（即每行去掉 padding mark 后 word 的个数）
        :return: lstm layor 的 output（包括 output、ht、ct）
        """
        # 1.1_pack paded sequences：
        # sort：因为pytorch 处理 paded sequences 时要求长句子在上
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        # 设置原因：因为x 会根据x_sort_idx，变位置（长的在前），变化之后，x 中每个 vector 的下标也会重新排列，那么我们
        #        需要知道原先 x 是怎样的。当然这也是因为 python 赋值也为引用的问题，否则直接用一个新的变量存储原来的信
        #        息即可。
        # 语句解释: 假如      x：3 1 2 4  ->           1 2 3 4
        #            indices：0 1 2 3  -> x_sort_id: 1 2 0 3
        #       （因为排列之后下标也发生了变化） indices: 0 1 2 3
        # ----------------------------------------------------------------------------------
        # 我们知道如果按新的 x 的 indices 给原来的 x 进行标号应该是这样的:
        #  new_x：3 1 2 4  -> 由此可以发现 -> 对 x_sort_id 再排序一次后其下标就是 2 0 1 3
        # indices：2 0 1 3
        # ----------------------------------------------------------------------------------
        # !!! intuitive explain：
        # 因为 x_sort_idx 是用 old_indices 来记录 old_x 排序后的顺序，而在 x 改变之前其是按序排列的（0、1……）（废话）
        # 所以，将 x_sort_idx 和 new_x 合起来看，而 indices 可以作为这个整体的 indices。直观上我们容易看出对
        # x_sort_idx 进行排序之后对应的就是 old_x 的排列顺序，所以我们对其进行 sort，此时得到的 indices 自然就是 new_x
        # 按 old_x 顺序排列时的 indices
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        # 1.2_pack
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len.to('cpu'), batch_first=self.batch_first)
        # 1.3 process using LSTM
        # if self.rnn_type == 'LSTM':
        out_pack, (ht, ct) = self.RNN(x_emb_p, None)  # ht.shape = ct.shape = [num_layor,batch_size,hidden_num]
        # else:
        #     out_pack, ht = self.RNN(x_emb_p, None)
        #     ct = None
        # 1.4_unsort
        ht = torch.transpose(ht, 0, 1)[x_unsort_idx]
        ht = torch.transpose(ht, 0, 1)  # 因为我们主要是用 ht batch_size,hidden_num 部分，所以把 num_layor 换到
        # 第一个维度方便选择
        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]  #
            out = out[x_unsort_idx]
            """unsort: out c"""
            # if self.rnn_type == 'LSTM':
            ct = torch.transpose(ct, 0, 1)[
                x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
            ct = torch.transpose(ct, 0, 1)
            return out, (ht, ct)


def squeeze_sequence(x, x_len, batch_first=True):
    """
    因为原来 padding 的序列组的长度均为人为设定的，可能会出现这些 seq 的长度都小于预先设定值
    -> 压缩序列到长度均为该序列组中最长的 seq 的长度。
    :param x: 序列组
    :param x_len: 每个序列的个数
    :return: new x
    """
    """sort"""
    x_sort_idx = torch.sort(-x_len)[1]
    x_unsort_idx = torch.sort(x_sort_idx)[1]
    x_len = x_len[x_sort_idx]
    x = x[x_sort_idx]
    """pack"""
    x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len.to('cpu'), batch_first=batch_first)
    """unpack: out"""
    out = torch.nn.utils.rnn.pad_packed_sequence(x_emb_p,
                                                 batch_first=batch_first)  # (sequence, lengths)
    out = out[0]  # (sequence, lengths)
    """unsort"""
    out = out[x_unsort_idx]
    return out


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product',
                 dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.proj = nn.Linear(embed_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        # 得分函数的选择
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        if self.n_head != q_len:
            raise RuntimeError('n_head != query num')
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score


class NoQueryAttention(Attention):
    '''q is a parameter'''

    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', q_len=1,
                 dropout=0):
        super(NoQueryAttention, self).__init__(embed_dim, hidden_dim, out_dim, n_head, score_function, dropout)
        self.q_len = q_len
        self.q = nn.Parameter(torch.Tensor(q_len, embed_dim))
        self.reset_q()

    def reset_q(self):
        stdv = 1. / math.sqrt(self.embed_dim)
        self.q.data.uniform_(-stdv, stdv)

    def forward(self, k, **kwargs):
        mb_size = k.shape[0]
        q = self.q.expand(mb_size, -1, -1)
        return super(NoQueryAttention, self).forward(k, q)
