
### Dependencies
Python 3.8

### How to use the code

you need download pretrained bert model.

1. Download the Bert pretrained model from https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin 
2. Download the Bert config file from https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json
3. Download the Bert vocab file from https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt 
4. Rename:

    - `bert-base-uncased-pytorch_model.bin` to `pytorch_model.bin`
    - `bert-base-uncased-config.json` to `config.json`
    - `bert-base-uncased-vocab.txt` to `bert_vocab.txt`
5. Place `model` ,`config` and `vocab` file into  the `/pybert/pretrain/bert/base-uncased` directory.
7. Download kaggle data from https://www.dropbox.com/s/ggl9krhh6dcwhhz/train.csv and https://www.dropbox.com/s/ggl9krhh6dcwhhz/test.csv and place in `pybert/dataset`.
    -  you can modify the `io.task_data.py` to adapt your data.
8. Modify configuration information in `pybert/configs/basic_config.py`(the path of data,...).
9. Run `python run_bert.py --do_data` to preprocess data.
10. Run `python run_bert.py --do_train --save_best --do_lower_case` to fine tuning bert model.
11. Run `run_bert.py --do_test --do_lower_case` to predict new data.

### training 

```text
my best for 37 labels for long text like company description
learning_rate=0.0001
epochs=12

03/17/2023 08:01:36 - INFO - root -   Training/evaluation parameters Namespace(arch='bert', do_data=False, 
do_train=True, do_test=False, save_best=True, do_lower_case=True, data_name='kaggle', mode='min', 
monitor='valid_loss', epochs=12, resume_path='', predict_checkpoints=0, valid_size=0.2, 
local_rank=-1, sorted=1, n_gpu='0', gradient_accumulation_steps=1, train_batch_size=16, 
eval_batch_size=16, train_max_seq_len=512, eval_max_seq_len=512, loss_scale=0, 
warmup_proportion=0.1, weight_decay=0.01, adam_epsilon=1e-08, grad_clip=1.0, learning_rate=0.0001, 
seed=42, fp16=False, fp16_opt_level='O1')

{"loss": [0.34420319427462187, 0.1348500602385577, 0.12623817597826323, 0.11489458929966478, 0.09760856562677551, 
0.08229508871833484, 0.06981758006355342, 0.05964330833989615, 0.05249834025972614, 0.04669221158267236, 
0.04271873243737454, 0.040279789279927224], 
"auc": [0.5228237497319477, 0.6092902968471646, 0.6869881368792663, 0.793798787875583, 0.887606612771258,
 0.937622925397821, 0.9614198096060834, 0.9759980023409672, 0.9825942636046346, 0.9881100251563754, 
 0.9915513265036537, 0.9935599804069732], 
 "valid_loss": [0.14151175320148468, 0.12814728915691376, 0.12152998894453049, 0.11039523780345917, 
 0.10119393467903137, 0.09382440894842148, 0.09327846020460129, 0.08821512013673782, 0.08573216199874878, 
 0.08422341197729111, 0.08194807171821594, 0.08090870082378387], 
 "valid_auc": [0.6567467496245696, 0.6866062799929887, 0.7665290828363123, 0.8198986900199441, 0.8641982308577818, 
 0.8880497628984657, 0.8749221307398232, 0.8994013541553169, 0.9020165621698699, 0.896147573795188, 
 0.9033956326892194, 0.9082661624607397]}

```
```text
origin sample:
id,text,labels
0000997932d777bf,"Celential.ai - AI recruiting that scales with your hiring",hr|ai
```
