import torch
import logging
import sys


# 将类当作命名空间用
class opt:
    # 选择训练和测试的数据集
    dataset = 'restaurant'
    data = {
        'twitter':
            {'train_path': "E:\\Desktop\\ATAE-LSTM_论文ABSA\\datasets\\twitter\\train.raw",
             'test_path': "E:\\Desktop\\ATAE-LSTM_论文ABSA\\datasets\\twitter\\test.raw"},
        'restaurant':
            {'train_path': "E:\\Desktop\\ATAE-LSTM_论文ABSA\\datasets\\semeval14\\restaurant\\Restaurants_Train.xml.seg",
             'test_path': "E:\\Desktop\\ATAE-LSTM_论文ABSA\\datasets\\semeval14\\restaurant\\Restaurants_Test_Gold.xml"
                          ".seg"},
        'laptop':
            {'train_path': "E:\\Desktop\\ATAE-LSTM_论文ABSA\\datasets\\semeval14\\laptop\\Laptops_Train.xml.seg",
             'test_path': "E:\\Desktop\\ATAE-LSTM_论文ABSA\\datasets\\semeval14\\laptop\\Laptops_Test_Gold.xml.seg"}
    }
    # 只用了restaurant的数据
    train_path, test_path = data[dataset].values()  # the path of traning data & the path of test data
    embedding_name = '100_6B'
    embedding_set = {
        '100_6B': "./.vector_cache/glove.6B.100d.txt",
        '300_6B': "./.vector_cache/glove.6B.300d.txt",
        '300_42B': "./vector_cache/glove.42B.300d.txt"}
    embedding_file_path = embedding_set[embedding_name]

    # preTrain_args:                #主要的超参数
    train_ratio = 1  # the size ratio of train
    validation_ratio = 0  # the size ratio of validation
    embedding_dim = 100  # dimension of word embedding
    max_seq_len = 80
    paded_mark = 0.1
    seed = int(42)
    uniform_range = 0.01
    l2reg = 0.001

    # model_args:
    model_name = 'ATAE-LSTM'
    input_dim = embedding_dim
    hidden_dim = 300
    num_layer = 1
    bias = True
    batch_first = True
    num_class = 5

    # train_args:
    epoch = 30
    learn_rate = 0.003  # learning rate
    batch_size = 32  # the mini-batch size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    patience = 5


# 预处理
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
