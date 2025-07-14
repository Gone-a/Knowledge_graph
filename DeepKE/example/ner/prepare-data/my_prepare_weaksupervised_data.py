#新增跳过空行和注释行的处理
# -*- coding: utf-8 -*-
import os
import random
import jieba.posseg as psg
import jieba
import csv
import argparse

def add_entity(dict_dir):
    """
    把实体字典加载到jieba里，
    实体作为分词后的词，
    实体标记作为词性
    """
    
    dics = csv.reader(open(os.path.join(os.getcwd(), dict_dir),'r',encoding='utf8'))
    
    for row in dics:
        # 处理前先去除每个字段的前后空格
        processed_row = [field.strip() for field in row]

        # 跳过空行（所有字段均为空）
        if not any(processed_row):
            continue
            
        # 跳过注释行（假设注释以#开头）
        if processed_row[0].startswith('#'):
            continue
            
        # 处理有效行（长度为2且字段非空）
        if len(processed_row) == 2 and processed_row[0] and processed_row[1]:
            jieba.add_word(processed_row[0], tag=processed_row[1])
            """ 保证由多个词组成的实体词不被切分开 """
            jieba.suggest_freq(processed_row[0])
    

def auto_label(input_texts, data_type, end_words=['。','.','?','？','!','！'], mode='cn'):
    
    writer = open(f"example_{mode}_{data_type}.txt", "w", encoding="utf8")
    for input_text in input_texts:
        words = psg.cut(input_text)

        for word,pos in words: 
            word,pos = word.strip(), pos.strip()   
            if not (word and pos):
                continue
            
            """ 如果是英文，需要转换成list，否则会按照Character级别遍历 """
            if mode == 'en':
                word = word.split(' ')

            """ 如果词性不是实体的标记，则打上O标记 """
            if pos not in label_set:
                for char in word:
                    string = char + ' ' + 'O' + '\n'
                    """ 在句子的结尾换行 """
                    if char in end_words:
                        string += '\n'
                    writer.write(string)
            else:
                """ 如果词性是实体的标记，则打上BI标记"""
                for i, char in enumerate(word):
                    if i == 0:
                        string = char + ' ' + 'B-' + pos + '\n'    
                    else:
                        string = char + ' ' + 'I-' + pos + '\n'
                    writer.write(string)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='prepare_weak_supervised_data')
    parser.add_argument('--language', type=str, default='cn')
    parser.add_argument('--source_dir', type=str, default='source_data')
    parser.add_argument('--dict_dir', type=str, default='vocab_dict.csv')
    parser.add_argument('--train_rate', type=float, default=0.8)
    parser.add_argument('--dev_rate', type=float, default=0.1)
    parser.add_argument('--test_rate', type=float, default=0.1)

    args = parser.parse_args()
    print(args, '\n')
    
    mode = args.language
    source_data_dir = args.source_dir
    dict_dir = args.dict_dir
    train_rate = args.train_rate
    dev_rate = args.dev_rate
    test_rate = args.test_rate
    
    dics = csv.reader(open(os.path.join(os.getcwd(), dict_dir),'r',encoding='utf8'))
    label_set = set()
    
    for raw in dics:
        # 处理前先去除每个字段的前后空格
        processed_row = [field.strip() for field in raw]
        
        # 跳过空行（所有字段均为空）
        if not any(processed_row):
            continue
            
        # 跳过注释行（假设注释以#开头）
        if processed_row[0].startswith('#'):
            continue
            
        # 处理有效行（长度为2且字段非空）
        if len(processed_row) == 2 and processed_row[0] and processed_row[1]:
            label_set.add(processed_row[1])
    
    end_words = set(['。','.','?','？','!','！'])
    add_entity(dict_dir)

    lines = list()

    for file in os.listdir(os.path.join(os.getcwd(), source_data_dir)):
        if ('txt' not in file):
            continue

        fp = open(os.path.join(os.getcwd(), source_data_dir, file), 'r', encoding='utf8')
        lines.extend([line.strip().replace('\n', '') for line in fp.readlines() if line.strip().replace('\n', '') != ''])

    print(f'Prepare the weak supervised dataset for langauge({mode}) ...\n')

    print(f'The source directory is {source_data_dir}\n')

    print(f'Total length of corpus is {len(lines)}\n')

    assert len(lines) > 0
    print(f'For example, the first instance is {lines[0]}\n')

    random.seed(42)
    random.shuffle(lines)

    assert((train_rate+dev_rate+test_rate) == 1.0)


    auto_label(lines[: int(train_rate * len(lines))], 'train', mode=mode)
    auto_label(lines[int(train_rate * len(lines)): int((train_rate + dev_rate) * len(lines))], 'dev', mode=mode)
    auto_label(lines[int((train_rate + dev_rate) * len(lines)) : ], 'test', mode=mode)

    print('Building success!!!')