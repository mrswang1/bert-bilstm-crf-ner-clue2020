from utils.create_batch_iter import create_batch_iter
from utils.progress_util import ProgressBar
import config as args
from model.bert_ner import Bert_CRF
from train.train import fit
if __name__=='__main__':
    #创建迭代数据
    training_iter,num_train_step = create_batch_iter('train')
    eval_iter = create_batch_iter('dev')
    test_iter = create_batch_iter('test')
    #轮数
    epoch_size = num_train_step*args.train_batch_size*args.gradient_accumulation_steps/args.num_train_epochs
    #显示进度
    pbar = ProgressBar(epoch_size=epoch_size,batch_size=args.train_batch_size)
    #模型
    model = Bert_CRF.from_pretrained('bert-base-chinese',num_tag=len(args.label_dict))
    fit(model=model,
        training_iter=training_iter,
        eval_iter=eval_iter,
        test_iter=test_iter,
        num_epoch=args.num_train_epochs,
        pbar=pbar,
        num_train_steps=num_train_step,
        verbose=1)