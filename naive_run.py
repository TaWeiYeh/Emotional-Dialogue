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

from seq2seq import *
from dataloading import *
from rl import dull_responses, load_sentiment_checkpoint
from Sentiment_Classifier import *


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=15):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexes_from_sentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words, tokens


def evaluate_input(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            quit = False
            for sent in dull_responses:
                if normalize_string_without_symbols(sent) == normalize_string_without_symbols(input_sentence): 
                    quit = True
                    break
            if input_sentence == 'q' or input_sentence == 'quit' or quit: break
            # Normalize sentence
            input_sentence = normalize_string(input_sentence)
            # Evaluate sentence
            output_words, _ = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


def get_samples(voc, test_sample_filename):
    # load file
    conversations = list()
    with open(test_sample_filename, 'r', encoding='utf-8') as f:
        for line in f:
            conversations.append(line)

    # eliminate any unknown words if show up in the sentence
    samples = list()
    eliminate_count = 0
    for sentence in conversations:
        is_valid_sentence = True
        for word in normalize_string(sentence).split(' '):
            if word not in voc.word2index.keys() :
                is_valid_sentence = False
                eliminate_count += 1
                break
        if is_valid_sentence:
            samples.append(sentence)
    print(f"Eliminate {eliminate_count} conversations. ")

    return samples 


def get_reward(response, sentiment_net, max_seq_length):
    trimmed_response = trimNPadding([response], max_seq_length, pad_value=0, data_type=torch.int64)
    # print(f"trimmed_response {trimmed_response}")
    inputs = torch.stack(trimmed_response)

    outputs = sentiment_net(inputs)
    _, predicted = torch.max(outputs.to('cpu').data, 1) # multiclass
    reward = predicted.ge(1).sum()

    return reward


def simutlate_turns(encoder, decoder, searcher, voc, samples, sentiment_net, max_seq_length, simualted_turns=100, max_turns=100, verbose=False):
    avg_conversation_turns = 0
    max_conversation_turns = 0
    avg_reward = 0
    max_reward = 0
    for i in range(simualted_turns):
        idx = torch.randint(len(samples), (1,))
        input_sentence = samples[idx]
        if verbose:
            print(f"Bot 0 : {input_sentence}")

        temp_conversation_turns = 0
        temp_reward = 0
        for j in range(max_turns):
            # Normalize sentence
            input_sentence = normalize_string(input_sentence)
            # Evaluate sentence
            output_words, tokens = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # identify if the response is emotional
            temp_reward += get_reward(tokens, sentiment_net, max_seq_length)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            output_sentence = ' '.join(output_words)
            # quit the conversation if the response is dull responses. 
            quit = False
            for dull in dull_responses:
                if normalize_string_without_symbols(dull) == normalize_string_without_symbols(output_sentence): 
                    quit = True
                    break
            if output_sentence == 'q' or output_sentence == 'quit' or quit: break

            if verbose:
                print(f'Bot {(j+1)%2}:', output_sentence)

            input_sentence = output_sentence
            temp_conversation_turns += 1
        
        if verbose:
            print("\n")

        max_conversation_turns = max(max_conversation_turns, temp_conversation_turns)
        avg_conversation_turns += temp_conversation_turns
        
        max_reward = max(max_reward, temp_reward.item())
        avg_reward += temp_reward.item()

    avg_reward /= simualted_turns
    avg_conversation_turns /= simualted_turns
    print(f"Average conversation turns are {avg_conversation_turns:.2f}. Max conversation turns are {max_conversation_turns}")
    print(f"Average rewards are {avg_reward:.2f}. Max rewards are {max_reward}")


if __name__ == '__main__':

    # device choice
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    corpus_name = "train"
    corpus = os.path.join("data", corpus_name)
    datafile = os.path.join(corpus, "formatted_dialogues_train.txt")

    # Load/Assemble voc and pairs   
    filename = os.path.join("data", "save", "MulticlassSentimentClassifier", \
                            "500_checkpoint.tar")  
    sentiment_checkpoint = torch.load(filename, map_location=torch.device('cpu'))
    voc = Voc(corpus_name)
    voc.__dict__ = sentiment_checkpoint['voc_dict']
    voc.trimmed = False
    sentiment_net_weight = sentiment_checkpoint['SentimentNet']
    voc.__dict__ = sentiment_checkpoint['voc_dict']
    vocab_size = sentiment_checkpoint['vocab_size']
    embedding_dim = sentiment_checkpoint['embedding_dim']
    max_seq_length = sentiment_checkpoint['max_seq_length']

    sentiment_net = SentimentNet(vocab_size, embedding_dim, max_seq_length)
    sentiment_net.load_state_dict(sentiment_net_weight)
    sentiment_net.eval()

    save_dir = os.path.join("data", "save")
    voc, pairs = load_prepare_data(corpus, corpus_name, datafile, save_dir, voc)
    # `Print some pairs to validate
    # print("\npairs:")
    # for pair in pairs[:10]:
    #     print(pair)
    pairs = trim_rare_words(voc, pairs, min_count=3)

    # Example for validation
    small_batch_size = 5
    batches = batch_2_train_data(
        voc, [random.choice(pairs) for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = batches

    # print("input_variable:", input_variable)
    # print("lengths:", lengths)
    # print("target_variable:", target_variable)
    # print("mask:", mask)
    # print("max_target_len:", max_target_len)

    # Configure models
    model_name = 'cb_model'
    attn_model = 'dot'
    # attn_model = 'general'
    # attn_model = 'concat'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    # Set checkpoint to load from; set to None if starting from scratch
    # load_file_name = "data/save/cb_model/train/2-2_500/10000_checkpoint.tar"               # Seq2Seq
    # load_file_name = "data/save/RL_model_seq/train/10000_checkpoint.tar"                   # RL with seq2seq with emotion
    load_file_name = "data/save/RL_model_seq_no_emotion/train/10000_checkpoint.tar"        # RL with seq2seq w/o emotion
    # load_file_name = "data/save/RL_model_seq_no_emotion_seq/train/10000_checkpoint.tar"    # RL w/o seq2seq w/o emotion
    # load_file_name = "data/save/RL_model_seq_no_seq2seq/train/10000_checkpoint.tar"        # RL w/o seq2seq but with emotion
    # checkpoint_iter = 10000  # 4000
    # load_file_name = os.path.join(save_dir, model_name, corpus_name,
    #                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
    #                            '{}_checkpoint.tar'.format(checkpoint_iter))
    # print(load_file_name)

    # Load model if a load_file_name is provided
    if load_file_name:
        # If loading on same machine the model was trained on
        #checkpoint = torch.load(load_file_name)
        # If loading a model trained on GPU to CPU
        checkpoint = torch.load(load_file_name, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        # voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    if load_file_name:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(
        attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if load_file_name:
        print("Now loading saved model state dicts")
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    # # Configure training/optimization
    # clip = 50.0
    # teacher_forcing_ratio = 1.0
    # learning_rate = 0.0001
    # decoder_learning_ratio = 5.0
    # n_iteration = 1000  # 4000
    # print_every = 1
    # save_every = 1000

    # # Ensure dropout layers are in train mode
    # encoder.train()
    # decoder.train()

    # # Initialize optimizers
    # print('Building optimizers ...')
    # encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    # decoder_optimizer = optim.Adam(
    #     decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    # if load_file_name:
    #     encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    #     decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # if USE_CUDA:
    #     # If you have cuda, configure cuda to call
    #     for state in encoder_optimizer.state.values():
    #         for k, v in state.items():
    #             if isinstance(v, torch.Tensor):
    #                 state[k] = v.cuda()

    #     for state in decoder_optimizer.state.values():
    #         for k, v in state.items():
    #             if isinstance(v, torch.Tensor):
    #                 state[k] = v.cuda()

    # checkpoint = torch.load(load_file_name, map_location=torch.device(
    #     'cuda') if torch.cuda.is_available() else torch.device('cpu'))
    # encoder_sd = checkpoint['en']
    # decoder_sd = checkpoint['de']
    # encoder_optimizer_sd = checkpoint['en_opt']
    # decoder_optimizer_sd = checkpoint['de_opt']
    # embedding_sd = checkpoint['embedding']
    # voc_dict = checkpoint['voc_dict']

    # # Initialize encoder & decoder models
    # encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    # decoder = LuongAttnDecoderRNN(
    #     attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

    # # Use appropriate device
    # encoder = encoder.to(device)
    # decoder = decoder.to(device)

    # # Set dropout layers to eval mode
    # encoder.eval()
    # decoder.eval()

    # # Initialize search module
    # searcher = GreedySearchDecoder(encoder, decoder)

    # # Begin chatting
    # #evaluateInput(encoder, decoder, searcher, voc)

    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)

    # Begin chatting
    # evaluate_input(encoder, decoder, searcher, voc)

    # simulate turns
    print("RL with seq2seq w/o emotion")
    test_sample_filename = "data/test/formatted_single_dialogues_test.txt"
    samples = get_samples(voc, test_sample_filename)
    simutlate_turns(encoder, decoder, searcher, voc, samples, sentiment_net, max_seq_length, simualted_turns=100, max_turns=10, verbose=False)
