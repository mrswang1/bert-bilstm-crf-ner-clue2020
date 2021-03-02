import config as args
from transformers import BertModel,BertTokenizer
import torch
from torch.utils.data import TensorDataset
#---------------------得到example----------------------------------
def get_data_list(filename):
    '''
    读取BIO文件，得到字，标签列表
    格式：
    [['word','word',...],[一句]...[一句]]
    '''
    data = []
    sentence = []
    tag = []
    for line in open(filename,'r',encoding='utf-8'):
        line = line.strip('\n')
        if not line:
            if(len(sentence)>0):
               data.append((sentence,tag))
               sentence=[]
               tag=[]
        else:
            if(line[0]==" "):
                line = "#"+line[1:]
                line = line.strip()
                word = line.split()
            else:
                word = line.split()
            #print(word)
            sentence.append(word[0])
            tag.append(word[1])
    return data

class DataProcessor(object):
    '''
    创建样例
    '''
    def get_train_examples(self,filename):
        raise NotImplementedError()

    def get_dev_examples(self,filename):
        raise NotImplementedError()

    def get_test_example(self,filename):
        raise NotImplementedError

    @classmethod
    def _read_tsv(cls,input_file,quotechar=None):
        return get_data_list(input_file)

class InputExample(object):
    '''
    创建输入实例
    guid：每个example拥有唯一的id
    text_a:第一个句子的原始文本，一般对于文本分类来说，只需要text_a
    text_b:第二个句子的原始文本，在句子对的任务中才有，分类问题中为None
    label：example对应的标签，对于训练集和验证集应非None，测试集为None
    '''
    def __init__(self,guid,text_a,text_b=None,label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label =label

class NerProc(DataProcessor):
    def _create_examples(self,data,set_type):
        examples = []
        for i,(sentence,label) in enumerate(data):
            guid = "%s-%s"%(set_type,i)
            text_a = ' '.join(sentence)
            label = label
            examples.append(InputExample(guid=guid,text_a=text_a,label=label))
        return  examples

    def get_train_examples(self,filename):
        data = self._read_tsv(filename)
        examples = self._create_examples(data,"train")
        return examples

    def get_dev_examples(self,filename):
        data = self._read_tsv(filename)
        examples = self._create_examples(data,"dev")
        return examples

    def get_test_examples(self,filename):
        data = self._read_tsv(filename)
        examples = self._create_examples(data,"test")
        return examples

#-----------------------得到特征--------------------------------
class InputFature(object):
    def __init__(self,input_ids,input_mask,segment_ids,label_ids,output_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids =label_ids
        self.output_mask = output_mask

def convert_examples_to_features(examples,label_dict,max_seq_length,tokenizer):
    '''
    :param examples: class examples
    :param label_list: example label
    :param max_seq_length: config.max_seq_length
    :param tokenizer: Bert里的tokenizer，获取每个字的id
    :return: [input_ids,input_mask,segment_ids,label_id]
    '''
    # load sub_vocab
    sub_vocab = {}
    with open("vocab.txt", 'r',encoding='utf-8') as fr:
        for line in fr:
            _line = line.strip('\n')
            if "##" in _line and sub_vocab.get(_line) is None:
                sub_vocab[_line] = 1
    #for s in sub_vocab:
     #   print(s)
    features = []
    length = []
    for i,example in enumerate(examples):
        #不是以空格分隔
        textlist = example.text_a.split(' ')
        labellist =example.label
        #print("textlist=",textlist)
        #print("labellist=",labellist)
        length.append(len(labellist))
        tokens = []
        labels = []
        for j,word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            #print("word=",word)
            tokens.extend(token)
            tag = labellist[j]
            #print("tag=",tag)
            for k in range(len(token)):
                if k==0:
                    labels.append(tag)
                else:
                    labels.append("X")
        #print(labels)
        #超出最大长度进行截断
        if len(tokens)>max_seq_length-2:
            tokens = tokens[0:(max_seq_length-2)]
            labels = labels[0:(max_seq_length-2)]
        output_mask = [0 if sub_vocab.get(t) is not None else 1 for t in tokens]
        #句子首尾加入标示符
        tokens = ["[CLS]"]+tokens+["[SEP]"]
        segment_ids = [0]*len(tokens)
        #得到词的id
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        #得到mask
        input_mask = [1]*len(input_ids)
        label_ids = []
        #label不加CLSheSEP
        for i in range(len(labels)):
            label_ids.append(label_dict[labels[i]])
        #处理output_mask
        output_mask = [0] + output_mask + [0]

        # 填充
        padding = [0] * (max_seq_length - len(input_ids))
        label_padding = [-1]*(max_seq_length-len(label_ids))
        input_ids = input_ids + padding
        input_mask = input_mask + padding
        segment_ids = segment_ids + padding
        label_ids = label_ids+label_padding
        output_mask += padding

        features.append(InputFature(input_ids,input_mask,segment_ids,label_ids,output_mask))
    print(max(length))
    return features

def get_pytorch_data(features):
    input_ids = torch.tensor([f.input_ids for f in features],dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features],dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features],dtype=torch.long)
    label_ids = torch.tensor([f.label_ids for f in features],dtype=torch.long)
    output_mask = torch.tensor([f.output_mask for f in features],dtype=torch.long)
    data = TensorDataset(input_ids,input_mask,segment_ids,label_ids,output_mask)
    return data

#调试数据
processor = NerProc()
train_examples = processor.get_train_examples(args.train_file)
dev_examples = processor.get_dev_examples(args.dev_file)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese')
train_features = convert_examples_to_features(train_examples,args.label_dict,args.max_seq_length,tokenizer)
dev_features = convert_examples_to_features(dev_examples,args.label_dict,args.max_seq_length,tokenizer)

#torch_dataset = get_pytorch_data(features)
#for each in torch_dataset:
 #   print(each)
#train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# data_iter = iter(train_dataloader)
# next(data_iter)

# input_ids, input_mask, segment_ids, label_ids = next(data_iter)
# print(input_ids.shape)  # 8 * 500, mini-batch


# model = BertForTokenClassification.from_pretrained(bert_model, num_labels = num_labels)

# loss = model(input_ids, segment_ids, input_mask, label_ids)
# print(loss)



