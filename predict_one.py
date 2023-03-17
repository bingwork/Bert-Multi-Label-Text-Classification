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
    text = """Company
Pixar Animation Studios
Brief
Pixar Animation Studios is a group charged with developing state-of-the-art computer technology for the film industry.
Industry
Animation
Founded
1986
Specialties
Animation
Description
Pixar Animation Studios, a wholly owned subsidiary of The Walt Disney Company, is an Academy Award®-winning film studio with world-renowned technical, creative and production capabilities in the art of computer animation. The Northern California studio has created some of the most successful and beloved animated films of all time, including “Toy Story,” “Monsters, Inc.,” “Cars,” “The Incredibles,” “Ratatouille,” “WALL•E,” “Up,” “Brave,” “Inside Out,” “Coco” and “Turning Red.” Its movies and technology have won 40 Academy Awards® and the films have grossed more than $14 billion at the worldwide box office. ""Lightyear,"" Pixar's 26th feature, released in theaters on June 17, 2022. 

Pixar's objective is to combine groundbreaking technology and world-class creative talent to develop computer-animated films with memorable characters and heartwarming stories that appeal to audiences of all ages.


Careers Page: http://www.pixar.com/careers

Twitter: @PixarRecruiting"""
    max_seq_length = 512
    do_lower_case = True
    arch = 'bert'
    probs, labels_scores= main(text,arch,max_seq_length,do_lower_case)
    print(probs)
    print(labels_scores)
    
'''
#output
[0.9892476  0.24539666 0.97839487 0.00427242 0.7542925  0.00711112]
[('toxic', 0.9892476), ('obscene', 0.97839487), ('insult', 0.7542925)]
'''
