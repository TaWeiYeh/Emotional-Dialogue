import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import numpy as np
from scipy.spatial import distance


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

corpus_name = "dailydialog"
corpus = os.path.join("data", corpus_name)


def print_lines(file, n=10):
    with open(file, 'r', encoding="utf-8") as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

# Splits each line of the file into a dictionary of fields


def load_lines(file_name, split_type='text'):
    conversations = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            if split_type == 'text':
                values = line.split("__eou__")
            elif split_type == 'emotion':
                values = line.split()
            else: 
                raise TypeError(
                    "Please specify splitting type, i.e. text, or emotion. "
                )
            conversations.append(values)
    return conversations


# Extracts pairs of sentences from conversations
def extract_sentence_pairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        # We ignore the last line (no answer for it)
        for i in range(len(conversation) - 1):
            input_line = conversation[i].strip()
            target_line = conversation[i+1].strip()
            # Filter wrong samples (if one of the lists is empty)
            if input_line and target_line:
                qa_pairs.append([input_line, target_line])
    return qa_pairs

    
# Extract sentences from conversations
def extract_each_sentence(conversations):
    sentences = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        # We ignore the last line (no answer for it)
        for i in range(len(conversation)):
            input_line = conversation[i].strip()
            # Filter wrong samples (if one of the lists is empty)
            if input_line:
                sentences.append([input_line])
    return sentences


def create_formatted_dataset(source, target, split_type='text', extract_mode='pairs', type=None):
    # Define path to new file
    datafile = target

    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    print("\nLoading conversations...")
    conversations = load_lines(source, split_type=split_type)

    # Write new csv file
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter,
                            lineterminator='\n')
        if extract_mode == 'pairs':
            for pair in extract_sentence_pairs(conversations):
                writer.writerow(pair)
        elif extract_mode == 'single':
            for sentence in extract_each_sentence(conversations):
                writer.writerow(sentence)
        else:
            raise TypeError(
                "Please specify extract mode, i.e. pairs or single. "
            )

    # Print a sample of lines
    print("\nSample lines from file:")
    print_lines(datafile)




if __name__ == "__main__":
    # create_formatted_dataset(
    #     source="data/train/dialogues_train.txt",
    #     target="data/train/formatted_single_dialogues_train_count.txt",
    #     split_type='text',
    #     extract_mode='single',
    # )
    
    create_formatted_dataset(
        source="data/train/dialogues_train.txt",
        target="data/train/formatted_single_dialogues_train.txt",
        split_type='text',
        extract_mode='single',
    )

    create_formatted_dataset(
        source="data/validation/dialogues_validation.txt",
        target="data/validation/formatted_single_dialogues_validation.txt",
        split_type='text',
        extract_mode='single',
    )

    create_formatted_dataset(
        source="data/test/dialogues_test.txt",
        target="data/test/formatted_single_dialogues_test.txt",
        split_type='text',
        extract_mode='single',
    )
    
    '''
    Format Emotion Conversations
    '''
    # create_formatted_dataset(
    #     source="data/train/dialogues_emotion_train.txt",
    #     target="data/train/formatted_single_dialogues_emotion_train_count.txt",
    #     split_type='emotion',
    #     extract_mode='single',
    # )

    create_formatted_dataset(
        source="data/train/dialogues_emotion_train.txt",
        target="data/train/formatted_single_dialogues_emotion_train.txt",
        split_type='emotion',
        extract_mode='single',
    )

    create_formatted_dataset(
        source="data/validation/dialogues_emotion_validation.txt",
        target="data/validation/formatted_single_dialogues_emotion_validation.txt",
        split_type='emotion',
        extract_mode='single',
    )

    create_formatted_dataset(
        source="data/test/dialogues_emotion_test.txt",
        target="data/test/formatted_single_dialogues_emotion_test.txt",
        split_type='emotion',
        extract_mode='single',
    )
