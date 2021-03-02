from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
from pre_data import NerProc
from pre_data import convert_examples_to_features
from pre_data import get_pytorch_data
import config as args
from transformers import BertTokenizer

def create_batch_iter(mode):
    #构造迭代器
    processor = NerProc()
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    if mode == 'train':
        examples = processor.get_train_examples(args.train_file)
        num_train_step = int(len(examples)/args.train_batch_size/args.gradient_accumulation_steps*args.num_train_epochs)
        batch_size = args.train_batch_size

    elif mode =='dev':
        examples = processor.get_dev_examples(args.dev_file)
        batch_size = args.eval_batch_size
    elif mode == 'test':
        examples = processor.get_dev_examples(args.test_file)
        batch_size = args.test_batch_size
    else:
        raise ValueError("Invalid mode %s" %mode)

    #得到特征
    features =convert_examples_to_features(examples,args.label_dict,args.max_seq_length,tokenizer)
    data = get_pytorch_data(features)
    #数据集
    if mode == 'train':
        sampler = RandomSampler(data)
    elif mode == 'dev':
        sampler = SequentialSampler(data)
    elif mode == 'test':
        sampler = SequentialSampler(data)
    else:
        raise ValueError("Invalid mode %s" % mode)
    #迭代器
    it= DataLoader(dataset=data,sampler=sampler,batch_size=batch_size)

    if mode =='train':
        return it,num_train_step
    elif mode =='dev':
        return it
    elif mode == 'test':
        return it
    else:
        raise ValueError("Invalid mode %s" %mode)

