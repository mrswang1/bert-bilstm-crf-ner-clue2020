#---------------ARGS--------------
#---------Path Parameters---------
import os
root_path = '/home/wangjun/bert_bilstm_crf'+os.sep
train_file = root_path+'data'+os.sep+'train-clue.txt'
dev_file = root_path+'data'+os.sep+'dev-clue.txt'
test_file = root_path+'data'+os.sep+'test.txt'
bert_model = root_path+'model'+os.sep+'pytorch_pretrained_model'
vocab_path = bert_model+os.sep+'vocab.txt'
image_path = root_path+'output'+os.sep+'image'+os.sep+'loss_acc.png'
output_path = root_path+'output'+os.sep+'checkpoint'
#---------------Other Parameters-------
label_dict = {
'B-company':0, 
'I-company':1, 
'B-name':2,
'I-name':3,      
'B-game':4,
'I-game':5, 
'B-organization':6,
'I-organization':7,
'B-movie':8,
'I-movie':9,
'B-position':10,
'I-position':11,
'B-address':12,
'I-address':13,
'B-government':14,
'I-government':15,
'B-scene':16,
'I-scene':17, 
'B-book':18,
'I-book':19,
'O':20,
'START':21,
'STOP':22
}
flag_words = ["[PAD]","[CLS]","[SEP]","[UNK]"]
max_seq_length = 100
train_batch_size = 32
eval_batch_size = 32
test_batch_size = 32
gradient_accumulation_steps = 1
learning_rate = 1e-5
device = "cuda"
num_train_epochs = 30
seed = 2021
warmup_proportion = 0.1
no_cuda = False
num_labels = 7
hidden_dropout_prob = 0.5


