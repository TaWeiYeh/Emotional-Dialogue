import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from dataloading import *



"""
Variables
"""
num_epochs = 500
batch_size = 64
data_type = torch.int64
max_seq_length = 50
save_every = 50
print_every = 1200
save_dir = os.path.join("data", "save")
model_name = "MulticlassSentimentClassifier"

learning_rate = 0.001

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

# device choice
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

voc = Voc("train")


"""
Load Dataset
"""
print("Loading dataset....")
def load_file(file_name):
    conversations = list()
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            conversations.append(line)
    return conversations


def trimNPadding(sequences, max_seq_length, pad_value=0, dtype=data_type):
    output = list()
    id = 0
    for seq in sequences:
        if max_seq_length <= len(seq):
            seq = torch.tensor(seq[:max_seq_length], dtype=data_type)
        else:
            padding_length = max_seq_length - len(seq)
            seq = torch.tensor(seq, dtype=data_type)
            seq = F.pad(seq, (0, padding_length), "constant", pad_value)
        seq[-1] = EOS_token
        output.append(seq)
    return output


class SentimentDataset(Dataset):
    def __init__(self, sentences, sentiments):
        self.sentences = sentences
        self.sentiments = sentiments

    def __len__(self):
        assert len(self.sentences) == len(self.sentiments)
        return len(self.sentences)

    def __getitem__(self, idx):       
        # label = 1 if self.sentiments[idx] else 0  # binary
        # label = torch.tensor(label, dtype=self.sentiments[idx].dtype)
        # sample = (self.sentences[idx], label)
        sample = (self.sentences[idx], self.sentiments[idx]) # multiclass
        return sample


def create_data(corpus, sentiments, max_length_sentence, data_type):
    # print("\nCounting words...")
    # all_words = []
    # for sent in corpus:
    #     tokenize_word = word_tokenize(sent)
    #     for word in tokenize_word:
    #         all_words.append(word)

    # unique_words = set(all_words)
    # word_to_ix = {word: i for i, word in enumerate(unique_words)}
    # vocab_size = len(unique_words)

    # print("Embedding sentences...")
    # embedded_sentences = [[word_to_ix[word] for word in word_tokenize(sent)] for sent in corpus]

    # word_count = lambda input: len(word_tokenize(input[1]))
    # longest_sentence_idx = max(enumerate(corpus), key=word_count)[0]
    # longest_sentence = corpus[longest_sentence_idx]
    # max_length_sentence = len(word_tokenize(longest_sentence))
    # print(f"longest_sentence [{longest_sentence_idx}] is {longest_sentence}")

    print("Building vocab library...")
    for idx, sent in enumerate(corpus):
        corpus[idx] = normalize_string(sent)
        voc.add_sentence(corpus[idx])
    
    print("Embedding sentences...")
    
    embedded_sentences = [[voc.word2index[word] for word in sent.split(' ')] for sent in corpus]

    print("Padding...")
    padded_sentences = trimNPadding(embedded_sentences, max_length_sentence, 0, torch.int64)
    
    padded_sentences = torch.stack(padded_sentences)
    sentiments = np.array(sentiments, dtype=np.int64)
    sentiments = torch.tensor(sentiments, dtype=data_type).unsqueeze(1)
    
    dataset = SentimentDataset(padded_sentences, sentiments)

    return dataset, voc.num_words, voc.word2index

    
corpus_train = load_file("data/train/formatted_single_dialogues_train.txt")
sentiments_train = load_file("data/train/formatted_single_dialogues_emotion_train.txt")
corpus_test = load_file("data/test/formatted_single_dialogues_test.txt")
sentiments_test = load_file("data/test/formatted_single_dialogues_emotion_test.txt")
corpus_valid = load_file("data/validation/formatted_single_dialogues_validation.txt")
sentiments_valid = load_file("data/validation/formatted_single_dialogues_emotion_validation.txt")
assert len(corpus_train) == len(sentiments_train)
assert len(corpus_test) == len(sentiments_test)
assert len(corpus_valid) == len(sentiments_valid)

print("\nLoad training data...")
train_data, train_vocab_size, train_vocab_dict = create_data(corpus_train, sentiments_train, max_seq_length, torch.int64)
print("\nLoad test data...")
test_data, test_vocab_size, test_vocab_dict = create_data(corpus_test, sentiments_test, max_seq_length, torch.int64)
print("\nLoad valid data...")
valid_data, valid_vocab_size, valid_vocab_dict = create_data(corpus_test, sentiments_test, max_seq_length, torch.int64)

print(f"\ntrain_vocab_dict length is {len(train_vocab_dict)}")
print(f"\ntest_vocab_dict length is {len(test_vocab_dict)}")
print(f"\nvalid_vocab_dict length is {len(valid_vocab_dict)}")
vocab_size = max(train_vocab_size, test_vocab_size, valid_vocab_size)
print("\nvocab_size is ", vocab_size)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# # Display image and label.
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Sentence : {train_features}")
# print(f"Sentiment : {train_labels}")


"""
Model
"""
# model.add(Embedding(vocab_length, 20, input_length=length_long_sentence))
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))

class SentimentNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_length):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(max_length * embedding_dim, 7) # mutliclass
        self.fc2 = nn.Linear(7, 1) # binary

    def forward(self, x):
        x = self.embeddings(x)
        x = self.flatten(x)
        x = torch.sigmoid(self.fc1(x)) # mutliclass
        # x = torch.sigmoid(self.fc2(x)) # binary
        return x

embedding_dim = 20
net = SentimentNet(vocab_size, embedding_dim, max_seq_length)
net = net.to(device)
criterion = nn.CrossEntropyLoss() # multiclass 
# criterion = nn.BCELoss() # binary
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()


"""
Training
"""
def evaluate(model: nn.Module, eval_dataloader: DataLoader) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    accuracy = 0.
    with torch.no_grad():
        for i, data in enumerate(eval_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            labels = labels.flatten()

            # forward + backward + optimize
            outputs = net(inputs.to(device))
            loss = criterion(outputs, labels.to(device)) # (binary) BCELoss need float input

            # print statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.to('cpu').data, 1) # multiclass
            # predicted = outputs.cpu().ge(0.5).flatten().type(labels.dtype) #binary
            accuracy += (predicted == labels).sum().item()
        
        accuracy = 100 * accuracy / (labels.size(0) * len(eval_dataloader))
        total_loss /= len(eval_dataloader)

    model.train()  # turn on training mode

    return total_loss, accuracy


print("training...")
for epoch in range(num_epochs):  # loop over the dataset multiple times

    training_loss = 0.0
    correct = 0.0
    train_accuracy = 0.
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        labels = labels.flatten()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.to(device))
        loss = criterion(outputs, labels.to(device)) # (binary) BCELoss need float input
        loss.backward()
        optimizer.step()

        # print statistics
        training_loss += loss.item()
        _, predicted = torch.max(outputs.to('cpu').data, 1) # multiclass
        # predicted = outputs.cpu().ge(0.5).flatten().type(labels.dtype) # binary
        correct += (predicted == labels).sum().item()

        # if i % print_every == print_every - 1:    # print every 2000 mini-batches
        #     train_accuracy = 100 * correct / (labels.size(0) * print_every)
        #     training_loss /= print_every
        #     # print(f'[{epoch + 1}, {i + 1:5d}] loss: {training_loss:.3f} Accuracy: {train_accuracy:.2f}%')
        #     training_loss = 0.0
        #     correct = 0.0
    
    train_accuracy = 100 * correct / (batch_size * len(train_dataloader))
    training_loss /= len(train_dataloader)
    eval_loss, eval_accuracy = evaluate(net, test_dataloader)
    print(f'[{epoch + 1}] Train loss: {training_loss:.3f} Accuracy: {train_accuracy:.2f}% \
            Eval loss: {eval_loss:.3f} Accuracy: {eval_accuracy:.2f}%')
         
    #SAVE CHECKPOINT TO DO
    if (epoch % save_every == save_every - 1):
        directory = os.path.join(save_dir, model_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            'epoch': epoch + 1,
            'SentimentNet': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': training_loss,
            'voc_dict': voc.__dict__,
            'vocab_size' : vocab_size, 
            'embedding_dim' : embedding_dim, 
            'max_seq_length' : max_seq_length,
        }, os.path.join(directory, '{}_{}.tar'.format(epoch + 1, 'checkpoint')))
        print(f"Save checkpoint: {os.path.join(directory, '{}_{}.tar'.format(epoch + 1, 'checkpoint'))} \
                at Train loss: {training_loss:.3f} Accuracy: {train_accuracy:.2f}% \
                Eval loss: {eval_loss:.3f} Accuracy: {eval_accuracy:.2f}%")

print('Finished Training')
