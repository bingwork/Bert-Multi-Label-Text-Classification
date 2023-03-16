import torch
from pybert.configs.basic_config import config
from pybert.io.bert_processor import BertProcessor
from pybert.model.bert_for_multi_label import BertForMultiLable

def main(text,arch,max_seq_length,do_lower_case):
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=do_lower_case)
    label_list = processor.get_labels()
    id2label = {i: label for i, label in enumerate(label_list)}
    model = BertForMultiLable.from_pretrained(config['checkpoint_dir'] /f'{arch}', num_labels=len(label_list))
    tokens = processor.tokenizer.tokenize(text)
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:max_seq_length - 2]
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = processor.tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # Batch size 1, 2 choices
    logits = model(input_ids)
    probs = logits.sigmoid()
    probs = probs.cpu().detach().numpy()[0]
    labels_scores = [(id2label.get(i), p) for i, p in enumerate(probs) if p >= 0.5]
    return probs, labels_scores

if __name__ == "__main__":
    text = "Celential.ai - AI recruiting that scales with your hiring!"
    max_seq_length = 256
    do_loer_case = True
    arch = 'bert'
    probs, labels_scores= main(text,arch,max_seq_length,do_loer_case)
    print(probs)
    print(labels_scores)
    
'''
#output
[0.9892476  0.24539666 0.97839487 0.00427242 0.7542925  0.00711112]
[('toxic', 0.9892476), ('obscene', 0.97839487), ('insult', 0.7542925)]
'''
