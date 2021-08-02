import os
import re
import json

import numpy as np 
import pandas as pd 
from tqdm import tqdm
from konlpy.tag import Okt

FILTERS = "([~.,!?\"':;])"
PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"

PAD_INDEX = 0
SOS_INDEX = 1
EOS_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD_INDEX, SOS_INDEX, EOS_INDEX, UNK_INDEX]
CHANGE_FILTER = re.compile(FILTERS)

MAX_SEQUENCE = 25

def load_data(path):
    data_df = pd.read_csv(path, header=0)
    question, answer = list(data_df['Q']), list(data_df['A'])

    return question, answer


def data_tokenizer(data):
    words = []
    for sentence in data:
        sentence = re.sub(CHANGE_FILTER, "", sentence)
        for word in sentence.split():
            if word:
                words.append(word)
    return words


def prepro_like_morphlized(data):
    morph_analyzer = Okt()
    result_data = ()
    for seq in tqdm(data):
        morphlized_seq = " ".join(morph_analyzer.morphs(seq))
        result_data.append(morphlized_seq)

    return result_data


def load_vocabulary(path, vocab_path, tokenize_as_morph=False):
    vocabulary_list = []
    if not os.path.exists(vocab_path):
        if os.path.exists(path):
            question, answer = load_data(path)
            if tokenize_as_morph:
                question = prepro_like_morphlized(question)
                answer = prepro_like_morphlized(answer)
            data = []
            data.extend(question)
            data.extend(answer)
            words = data_tokenizer(data)
            words = list(set(words))
            words[:0] = MARKER
        with open(vocab_path, 'w', encoding='utf-8') as vocabulary_file:
            for word in words:
                vocabulary_file.write(word+'\n')

    with open(vocab_path, 'r', encoding='utf-8') as vocabulary_file:
        for line in vocabulary_file:
            vocabulary_list.append(line.strip())

    word2idx, idx2word = make_vocabulary(vocabulary_list)

    return word2idx, idx2word, len(word2idx)


def make_vocabulary(vocabulary_list):
    word2idx = {word : idx for idx, word in enumerate(vocabulary_list)}
    idx2word = {idx : word for idx, word in enumerate(vocabulary_list)}

    return word2idx, idx2word


def enc_processing(data, dictionary, tokenize_as_morph=False):
    sequence_input_index = []
    sequence_length = []

    if tokenize_as_morph:
        data = prepro_like_morphlized(data)

    for sentence in data:
        sentence = re.sub(CHANGE_FILTER, '', sentence)
        sequence_index = []

        for word in sentence.split():
            if dictionary.get(word) is not None:
                sequence_index.extend([dictionary[word]])
            else:
                sequence_index.extend([dictionary[UNK]])

        if len(sequence_length) > MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE]

        sequence_length.append(len(sequence_index))
        sequence_index += (MAX_SEQUENCE - len(sequence_length)) * [dictionary[PAD]]
        sequence_input_index.append(sequence_index)

    return np.asarray(sequence_input_index), sequence_length


def dec_input_processing(data, dictionary, tokenize_as_morph=False):
    sequence_input_index = []
    sequence_length = []

    if tokenize_as_morph:
        data = prepro_like_morphlized(data)

    for sentence in data:
        sentence = re.sub(CHANGE_FILTER, '', sentence)
        sequence_index = []
        sequence_index = [dictionary[SOS]] +[dictionary[word] if word in dictionary else dictionary[UNK] for word in sentence.split()]

        if len(sequence_length) > MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE]

        sequence_length.append(len(sequence_index))
        sequence_index += (MAX_SEQUENCE - len(sequence_length)) * [dictionary[PAD]]
        sequence_input_index.append(sequence_index)

    return np.asarray(sequence_input_index), sequence_length


def dec_target_processing(data, dictionary, tokenize_as_morph=False):
    sequence_target_index = []

    if tokenize_as_morph:
        data = prepro_like_morphlized(data)

    for sentence in data:
        sentence = re.sub(CHANGE_FILTER, "", sentence)
        sequence_index = [dictionary[word] if word in dictionary else dictionary[UNK] for word in sentence.split()]

        if len(sequence_index) >= MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE - 1] + [dictionary[EOS]]
        else:
            sequence_index += [dictionary[EOS]]
        
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        
        sequence_target_index.append(sequence_index)

        return np.asarray(sequence_target_index)