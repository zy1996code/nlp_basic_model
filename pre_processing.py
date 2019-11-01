from nltk.corpus import stopwords #停用词
from string import punctuation #标点符号
from collections import Counter #计数器
import pandas as pd
import numpy as np #用于做线性代数操作
import torch #PyTorch 框架
from torch.utils.data import DataLoader, TensorDataset #PyTorch 的数据格式


def load_data():
    '''
    加载数据集，返回文本和目标标签
    例如加载到数据集中 review 列的第 2 个数据为：
    review[1]为 One of the most frightening game experiences ever that will make you keep the lights on next to your bed. Great storyline with a romantic, horrific, and ironic plot. Fans of the original Resident Evil will be in for a surprise of a returning character! Not to mention that the voice-acting have drastically improved over the previous of the series. Don't miss out on the best of the series.
    '''
    train_data = pd.read_csv('data_IMDB/train.csv')
    test_data = pd.read_csv('data_IMDB/test.csv')
    reviews = train_data['review'].get_values()
    labels = train_data["sentiment"].get_values()
    # input_test = test_data['review'].get_values()
    # y_test=list()
    return reviews, labels


def text_formatting(reviews):
    '''
    文本格式化，进行包括但不限于去掉标点和停用词，缩写还原等数据清洗工作
    review[1]变为 one frightening game experiences ever make keep lights next bed great storyline romantic horrific ironic plot fans original resident evil surprise returning character mention voiceacting drastically improved previous series do not miss best series
    '''
    stop_words = stopwords.words('english')
    appos = {
        "aren't": "are not",
        "can't": "cannot",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "i'd": "I would",
        "i'd": "I had",
        "i'll": "I will",
        "i'm": "I am",
        "isn't": "is not",
        "it's": "it is",
        "it'll": "it will",
        "i've": "I have",
        "let's": "let us",
        "mightn't": "might not",
        "mustn't": "must not",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "we'd": "we would",
        "we're": "we are",
        "weren't": "were not",
        "we've": "we have",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "where's": "where is",
        "who'd": "who would",
        "who'll": "who will",
        "who're": "who are",
        "who's": "who is",
        "who've": "who have",
        "won't": "will not",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have",
        "'re": " are",
        "wasn't": "was not",
        "we'll": " will",
        "didn't": "did not"
    }  # 缩写
    all_reviews = list()
    for text in reviews:
        lower_case = text.lower()
        #review 是词序列，this is one of the dumbest films  i  ve ever seen . it rips off nearly ever type of thriller and manages to make a mess of them all .  br    br   there  s not a single good line or character in the whole mess . if there was a plot  it was an afterthought and as far as acting goes  there  s nothing good to say so ill say nothing . i honestly cant understand how this type of nonsense gets produced and actually released  does somebody somewhere not at some stage think   oh my god this really is a load of shite  and call it a day . its crap like this that has people downloading illegally  the trailer looks like a completely different film  at least if you have download it  you haven  t wasted your time or money don  t waste your time  this is painful .
        words = lower_case.split()
        #review 词列表'''['this', 'is', 'one', 'of', 'the', 'dumbest', 'films', 'i', 've', 'ever', 'seen', '.', 'it', 'rips', 'off', 'nearly', 'ever', 'type', 'of', 'thriller', 'and', 'manages', 'to', 'make', 'a', 'mess', 'of', 'them', 'all', '.', 'br', 'br', 'there', 's', 'not', 'a', 'single', 'good', 'line', 'or', 'character', 'in', 'the', 'whole', 'mess', '.', 'if', 'there', 'was', 'a', 'plot', 'it', 'was', 'an', 'afterthought', 'and', 'as', 'far', 'as', 'acting', 'goes', 'there', 's', 'nothing', 'good', 'to', 'say', 'so', 'ill', 'say', 'nothing', '.', 'i', 'honestly', 'cant', 'understand', 'how', 'this', 'type', 'of', 'nonsense', 'gets', 'produced', 'and', 'actually', 'released', 'does', 'somebody', 'somewhere', 'not', 'at', 'some', 'stage', 'think', 'oh', 'my', 'god', 'this', 'really', 'is', 'a', 'load', 'of', 'shite', 'and', 'call', 'it', 'a', 'day', '.', 'its', 'crap', 'like', 'this', 'that', 'has', 'people', 'downloading', 'illegally', 'the', 'trailer', 'looks', 'like', 'a', 'completely', 'different', 'film', 'at', 'least', 'if', 'you', 'have', 'download', 'it', 'you', 'haven', 't', 'wasted', 'your', 'time', 'or', 'money', 'don', 't', 'waste', 'your', 'time', 'this', 'is', 'painful', '.']'''
        reformed = [appos[word] if word in appos else word for word in words] #去掉缩略词
        reformed_test=list()
        for word in reformed:
            if word not in stop_words: #去掉停用词
                reformed_test.append(word)
        #print(reformed_test)'''['rented', 'matrix', 'revisited', 'friend', 'mine.', 'loved', 'matrix', 'love', 'filmmaking', 'wanted', 'see', 'going', 'behind', 'scenes', 'matrix.', 'turns', 'matrix', 'revisited', 'tells', 'hardly', 'anything', 'art', 'filmmaking', 'even', 'matrix', 'made.', 'basically', 'huge', 'commercial', 'matrix,', 'movie', 'target', 'audience', 'matrix', 'revisited', 'already', 'seen!<br', '/><br', '/>if', 'really', 'want', 'know', 'process', 'troubles', 'stress', 'detail', 'went', 'making', 'matrix,', 'look', 'bonus', 'features', 'original', 'dvd', 'matrix.', 'things', 'show', 'documentaries', 'even', 'realized', 'done', 'done.', 'matrix', 'difficult', 'challenging', 'film', 'make', 'deserves', 'credit', '"documentary"', 'that is', 'informative', 'interesting', 'mtv', 'special.']'''
        reformed = " ".join(reformed_test)

        punct_text = "".join([ch for ch in reformed if ch not in punctuation]) #去掉标点符号
        all_reviews.append(punct_text)
    all_text = " ".join(all_reviews)
    all_words = all_text.split()
    return all_reviews, all_words


def word2integer():
    '''
    文本数字化，进行单词到整数的映射，给每个单词标号
    例如{...,'i':12, ..., 'love':5, ...,'china':66...}'''
    all_reviews, all_words = text_formatting(reviews)
    count_words = Counter(all_words)
    total_words = len(all_words)
    sorted_words = count_words.most_common(total_words)
    vocab_to_int = {w: i + 1 for i, (w, c) in enumerate(sorted_words)}
    return vocab_to_int

def text_encoding(reviews, vocab_to_int):
    '''
    文段数字化，例如：'i love china'已经被编码为['12', '5', '66']
    review[1]被编码为[[4670], [3305], [4083], [4670], [2293], [6193], [3796], [4083], [6174], [4670], [3436], [6193],[2293], [2168], [59], [3323], [3796], [6193], [4083], [3305], [59], [3305], [3323], [], [3323], [230], [6174], [4083], [], [4083], [2955], [4733], [4083], [2168], [59], [4083], [3305], [1941], [4083], [3436], [], [4083], [5203], [4083], [2168], [], [6193], [3796], [230], [6193], [], [3695], [59], [3580], [3580], [], [6174], [230], [7900], [4083], [], [21996], [4670], [2829], [], [7900], [4083], [4083], [4733], [], [6193], [3796], [4083], [], [3580], [59], [3323], [3796], [6193], [3436], [], [4670], [3305], [], [3305], [4083], [2955], [6193], [], [6193], [4670], [], [21996], [4670], [2829], [2168], [], [892], [4083], [2938], [], [], [3323], [2168], [4083], [230], [6193], [], [3436], [6193], [4670], [2168], [21996], [3580], [59], [3305], [4083], [], [3695], [59], [6193], [3796], [], [230], [], [2168], [4670], [6174], [230], [3305], [6193], [59], [1941], [], [], [3796], [4670], [2168], [2168], [59], [2293], [59], [1941], [], [], [230], [3305], [2938], [], [59], [2168], [4670], [3305], [59], [1941], [], [4733], [3580], [4670], [6193], [], [], [2293], [230], [3305], [3436], [], [4670], [2293], [], [6193], [3796], [4083], [], [4670], [2168], [59], [3323], [59], [3305], [230], [3580], [], [2168], [4083], [3436], [59], [2938], [4083], [3305], [6193], [], [4083], [5203], [59], [3580], [], [3695], [59], [3580], [3580], [], [892], [4083], [], [59], [3305], [], [2293], [4670], [2168], [], [230], [], [3436], [2829], [2168], [4733], [2168], [59], [3436], [4083], [], [4670], [2293], [], [230], [], [2168], [4083], [6193], [2829], [2168], [3305], [59], [3305], [3323], [], [1941], [3796], [230], [2168], [230], [1941], [6193], [4083], [2168], [], [], [3305], [4670], [6193], [], [6193], [4670], [], [6174], [4083], [3305], [6193], [59], [4670], [3305], [], [6193], [3796], [230], [6193], [], [6193], [3796], [4083], [], [5203], [4670], [59], [1941], [4083], [], [230], [1941], [6193], [59], [3305], [3323], [], [3796], [230], [5203], [4083], [], [2938], [2168], [230], [3436], [6193], [59], [1941], [230], [3580], [3580], [21996], [], [59], [6174], [4733], [2168], [4670], [5203], [4083], [2938], [], [4670], [5203], [4083], [2168], [], [6193], [3796], [4083], [], [4733], [2168], [4083], [5203], [59], [4670], [2829], [3436], [], [4670], [2293], [], [6193], [3796], [4083], [], [3436], [4083], [2168], [59], [4083], [3436], [], [], [2938], [4670], [3305], [], [6193], [], [6174], [59], [3436], [3436], [], [4670], [2829], [6193], [], [4670], [3305], [], [6193], [3796], [4083], [], [892], [4083], [3436], [6193], [], [4670], [2293], [], [6193], [3796], [4083], [], [3436], [4083], [2168], [59], [4083], [3436], []]
    '''
    all_reviews=list()
    for text in reviews:
        text = text.lower()
        text = "".join([ch for ch in text if ch not in punctuation])
        all_reviews.append(text)
    encoded_reviews=list()
    for review in all_reviews:
        encoded_review=list()
        for word in review.split():
            if word not in vocab_to_int.keys():
                encoded_review.append(0)
            else:
                encoded_review.append(vocab_to_int[word])
        encoded_reviews.append(encoded_review)

    return encoded_reviews

def padding(encoded_reviews, sequence_length=250):
    '''
    文段长度等长化， 一般固定每句话的长度（单词数）形成一个大小（句子数×每句话长度）大矩阵作为整个数据集的大feature_map'''
    features = np.zeros((len(encoded_reviews), sequence_length), dtype=int) #搞成了一个大小（句子数×每句话长度）大矩阵

    for i, review in enumerate(encoded_reviews):
        review_len = len(review)
        if (review_len <= sequence_length):
            zeros = list(np.zeros(sequence_length - review_len))
            new = zeros + review
        else:
            new = review[:sequence_length]
        features[i, :] = np.array(new)
    return features



def dataset_deviding(features, labels):
    '''
    数据集切分为训练集、验证集和测试集
    1.训练集用来训练模型，反向传播
    2.验证集用来验证该组超参数是否合适，以及监测过拟合情况
    3.测试集用以评价模型，评价模型泛化能力，实际就是看看你的模型是否 work，给模型考考试看看是不是学傻了'''
    train_x = features[:int(0.90 * len(features))]
    train_y = labels[:int(0.90 * len(features))]
    valid_x = features[int(0.90 * len(features)):]
    valid_y = labels[int(0.90 * len(features)):]
    return train_x, train_y, valid_x, valid_y

import torch #PyTorch 框架
from torch.utils.data import DataLoader, TensorDataset #PyTorch 的数据格式
def torch_dataloader(train_x, train_y, valid_x, valid_y):
    #创建 PyTorch 能看懂的数据格式
    train_data=TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data=TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
    #加载 PyTorch 能看懂的数据
    batch_size=50
    train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader=DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    # obtain one batch of training data
    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()
    print('Sample input size: ', sample_x.size())  # batch_size, seq_length
    print('Sample input: \n', sample_x)
    print()
    print('Sample label size: ', sample_y.size())  # 大小等于batch_size，因为有多少句话就有多少个输出预测值
    print('Sample label: \n', sample_y)

if __name__ == '__main__':
    reviews, labels = load_data()
    formated_reviews, all_words = text_formatting(reviews)
    vocab_to_int =  word2integer()
    encoded_reviews=text_encoding(formated_reviews, word2integer())
    features=padding(encoded_reviews, sequence_length= 250)
    train_x, train_y, valid_x, valid_y = dataset_deviding(features, labels)
    torch_dataloader(train_x, train_y, valid_x, valid_y)

    '''
    Sample input size:  torch.Size([50, 250])
    Sample input:
     tensor([[    0,     0,     0,  ...,   524,   582,   352],
            [    0,     0,     0,  ...,   113,  4982,   616],
            [    0,     0,     0,  ..., 15326,     1, 15419],
            ...,
            [    0,     0,     0,  ...,  6678,  2454,   351],
            [    0,     0,     0,  ...,     7,   362, 26039],
            [    0,     0,     0,  ...,   685,  2437,   963]])

    Sample label size:  torch.Size([50])
    Sample label:
     tensor([0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0,
            0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0,
            0, 1])
    '''
