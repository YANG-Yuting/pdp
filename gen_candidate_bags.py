import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import pickle
from config import args

task = 'imdb'
train = True
sym = True

# data_path = args.data_path
sizes = {'imdb': 50000, 'mr': 20000, 'fake': 50000}
max_vocab_size = sizes[task]
with open(args.data_path + '%s/dataset_%d_has_punctuation.pkl' % (task, max_vocab_size), 'rb') as f:
    datasets = pickle.load(f)
inv_full_dict = datasets.inv_full_dict
full_dict = datasets.full_dict
full_dict['<oov>'] = len(full_dict.keys())
inv_full_dict[len(full_dict.keys())] = '<oov>'
if sym:
    with open(args.data_path + '%s/word_candidates_sense_top5_sym_has_punctuation.pkl' % task, 'rb') as fp:  # 我们过滤的同义词表
        word_candidate = pickle.load(fp)
else:
    with open(args.data_path + '%s/word_candidates_sense_top5_has_punctuation.pkl' % task, 'rb') as fp:  # 我们过滤的同义词表
        word_candidate = pickle.load(fp)
pos_list = ['NN', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

train_x = datasets.train_seqs2
train_x = [[datasets.inv_full_dict[word] for word in text] for text in train_x]
train_y = datasets.train_y
test_x = datasets.test_seqs2
test_x = [[datasets.inv_full_dict[word] for word in text] for text in test_x]
test_y = datasets.test_y


"""获得数据的各个位置同义词，写入文件"""
def gen_texts_candidates():
    if train:  # 对于训练集
        data = train_x
        pos_tags_file = args.data_path + '%s/pos_tags_train_has_punctuation.pkl' % task
        if sym:
            outfile = args.data_path + '%s/candidates_train_sym_has_punctuation.pkl' % task
        else:
            outfile = args.data_path + '%s/candidates_train_has_punctuation.pkl' % task

    else:  # 对于测试集
        data = test_x  # [:200]
        pos_tags_file = args.data_path + '%s/pos_tags_test_has_punctuation.pkl' % task
        if sym:
            outfile = args.data_path + '%s/candidates_test_sym_has_punctuation.pkl' % task
        else:
            outfile = args.data_path + '%s/candidates_test_has_punctuation.pkl' % task


    # load pos_tags
    with open(pos_tags_file, 'rb') as fp:
        pos_tags = pickle.load(fp)

    # get candidate
    candidate_bags = {}
    for text_str in data:
        """以下都针对去除\x85的"""
        # 获得词性
        # 词性标注的时候会去除掉\x85，所以取词性的时候，也要先去除该符号
        # query_text = []
        # for word in text_str:
        #     if word == '\x85':
        #         continue
        #     else:
        #         query_text.append(word.replace('\x85',''))
        # pos_tag = pos_tags[' '.join(query_text)].copy()
        pos_tag = pos_tags[' '.join(text_str)].copy()
        # str转id（同义词集是用id存的），有不存在于词汇表的，保留原str，之后分类需要
        text_ids = []
        for word_str in text_str:
            if word_str in full_dict.keys():
                text_ids.append(full_dict[word_str])  # id
            else:
                text_ids.append(word_str)  # str
        # 获得候选集
        candidate_bag = {}
        for j in range(len(text_ids)):  # 对于每个位置
            word = text_ids[j]
            pos = pos_tag[j][1]
            neigbhours = [word]
            if isinstance(word, int) and pos in pos_list and word < len(word_candidate): # word_candidate只保留了50000个词
                if pos.startswith('JJ'):
                    pos = 'adj'
                elif pos.startswith('NN'):
                    pos = 'noun'
                elif pos.startswith('RB'):
                    pos = 'adv'
                elif pos.startswith('VB'):
                    pos = 'verb'
                neigbhours.extend(word_candidate[word][pos])  # 候选集
            # 转str
            neigbhours = [inv_full_dict[n] if isinstance(n, int) else n for n in neigbhours]
            candidate_bag[inv_full_dict[word] if isinstance(word, int) else word] = neigbhours

        """对于\x85做处理，与原文对齐"""
        # candidate_bag['\x85'] = ['\x85']  # 因为candidate_bag是字典，所以直接对所有的句子都加入这个，任何时候出现都能查到即可
        candidate_bags[' '.join(text_str)] = candidate_bag  # 保存的候选集里保留特殊符号

    # write to file
    f = open(outfile, 'wb')
    pickle.dump(candidate_bags, f)


if __name__ == '__main__':
    gen_texts_candidates()

