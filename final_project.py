# This code is meant to be run on the GPU. On Midway3, a full run of the code takes ~8 hrs.

# Imports ------------------------------------------------------------------------------------------
import os 
import pathlib
import regex as re 
from glob import glob 
from itertools import chain
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import torch 
from torchtext.vocab import build_vocab_from_iterator
from torch import nn
from torch.nn import functional as F
import math 
import time 

nottingham_path = pathlib.Path() / 'nottingham'
files = nottingham_path.glob('*.abc')

#  Helper functions --------------------------------------------------------------------------------
def abc_preprocess(file):
    """
    Takes in a raw .abc file and returns a list where each item is one bar, in sequence, of 
    the tunes in that .abc file.
    """
    # Remove un-needed fields 
    remove = ['X:.+\n', 'T:.+\n', 'C:.+\n', 'O:.+\n', 'A:.+\n', '^M:.+\n', 'Z:.+\n', 'N:.+\n', 
              'G:.+\n', 'H:.+\n', 'B:.+\n', 'D:.+\n', 'F:.+\n', 'S:.+\n', 'I:.+\n', 'P:.+\n', 
              '%.+', 'R:.+\n']
    remove_regex = '|'.join(remove)

    file = re.sub(remove_regex, '', file)

    file = file.split(sep='\n\n') # split the entire string into a list with the different tunes 
    tunes = [tune for tune in file if tune] # strip whitespace 

    return tunes

def tunes_flatten(tunes):
    """
    Takes a nested list (the output of re.split applied to the list of tokens) and flattens 
    the list to be 1-dimensional again.
    """
    tunes = [content.strip() for bar in tunes for content in bar] # get rid of whitespace 
    tunes = [content for content in tunes if content]
    
    return tunes 

def tunes_tokenize(tunes):
    """
    Takes in a list of tunes corresponding to a .abc file and returns a flat list of tokens,
    in sequence, corresponding to the pieces in the file 
    """
    tunes = [re.split('(?<!\||:)\|{1}(?!\||:)', tune) for tune in tunes] # split on bar 
    tunes = [re.split('(\([0-9])', bar) for tune in tunes for bar in tune]
    # capturing parentheses to keep what's enclosed; get triplets as their own token 

    tunes = tunes_flatten(tunes)
    tunes = [re.split('(\[.+\])', bar) for bar in tunes] # capturing parentheses
    # split on chords which are denoted by []

    tunes = tunes_flatten(tunes)

    tunes = [bar.split(" ") for bar in tunes] # split on the whitespace in each bar 
    tunes = tunes_flatten(tunes)
    
    # Tokenize coda signs 
    tunes = [re.split('(:\||:\||\|:|\|\|)', bar) for bar in tunes]
    tunes = tunes_flatten(tunes)

    # Tokenize the end of parts e.g. first time 1 or second time 2 to be played
    tunes = [re.split('\[[0-9]', bar) for bar in tunes]
    tunes = tunes_flatten(tunes)

    # capture notes and their tempo, eg d2, as separate tokens 
    tunes = [re.split('((?<=[0-9]+)[a-zA-Z])', bar) for bar in tunes]
    tunes = tunes_flatten(tunes)

    # split up fractional tempo delimiters e.g. 3/2  
    tunes = [re.split('([0-9]+/[0-9]+)', bar) for bar in tunes]
    tunes = tunes_flatten(tunes)

    # intermediate step to get rid of new lines 
    tunes = [re.split('\n', bar) for bar in tunes]
    tunes = tunes_flatten(tunes)

    # break any lingering connected triplets, doublets, etc. eg 
    # "A7"GF/2 should be "A7"G, F/2 as separate tokens  
    tunes = [re.split('([a-zA-Z]\/[0-9]{1})', bar) for bar in tunes]
    tunes = tunes_flatten(tunes)

    return tunes  

# Read the data ------------------------------------------------------------------------------------
tunes_list = []
for file in files:
    with open(file, 'rb') as f:
        tunes = f.read().decode('utf-8')
        tunes = re.sub('% Nottingham Music Database', '', tunes)
        tunes = abc_preprocess(tunes)
        tunes_list.append(tunes)

tunes_list = tunes_flatten(tunes_list)
tunes_out = tunes_flatten(tunes_list)

# Setup --------------------------------------------------------------------------------------------
print(f'Is GPU available? {torch.cuda.is_available()}')

vocab = build_vocab_from_iterator(tunes_out)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32 
bptt = 35

####################################################################################################
# NEURAL NETWORK CLASSES 
####################################################################################################
class PositionalEncoding_own(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # 2D tensor of shape (max_len, 1)
        pe = torch.zeros(max_len, 1, d_model)
        # Trick to create the denominator: use torch.exp() and -math.log() for numerical stability
        denom = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0, 0::2] = torch.sin(position * denom)
        pe[:, 0, 1::2] = torch.cos(position * denom)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Expects x to be the output from nn.Embedding, which will be be a Tensor of shape 
        (seq_len, batch_size, embedding_dim)
        """
        x = x + self.pe[:x.size(0)] # the pe tensor will broadcast if seq_len is longer than the 
        # max_len parameter 
        return self.dropout(x)

class GPT_own(nn.Module):

    def __init__(self, vocab_size, d_model, dropout, n_head, d_hid, n_layers):
        super().__init__()
        self.d_model = d_model 
        self.embedder = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding_own(d_model, dropout)
        decoder_only_layer = nn.TransformerEncoderLayer(d_model, n_head, d_hid, dropout=dropout,
                                                        activation='gelu')
        self.decoder_stack = nn.TransformerEncoder(decoder_only_layer, n_layers)
        self.lin_layer = nn.Linear(d_model, vocab_size)
        
        self.apply(self._init_weights) # initialize weights 
    
    # Custom function have GPT-like weight initialization 
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0., std=0.02)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0., std=0.02)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, src, src_mask):
        we = self.embedder(src) * math.sqrt(self.d_model)
        h0 = self.pos_encoder(we)
        hl = self.decoder_stack(h0, src_mask)
        output = self.lin_layer(hl)

        return output 
    
def generate_square_mask(sz):
    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

####################################################################################################
# DATA PROCESSING UTILITY FUNCTIONS 
####################################################################################################
def data_process(raw_text_iter):
    """Converts raw text into a flat Tensor."""
    # tokenizer splits the sentence; vocab grabs the indices in the vocabulary 
    # data will be a list of tensors, with each list item corresponding to a tensor with the 
    # vocabulary indices of the tokens in the sentence 
    data = [torch.tensor(vocab(tunes_tokenize(item)), dtype=torch.long) for item in raw_text_iter]

    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def batchify(data, bsz):
    seq_len = data.size(0) // bsz # floor function; calculates length of each example in batch
    data = data[:seq_len * bsz] # truncates the data to drop incomplete batches at the end 
    # Reshape the data, creating a copy 
    # Output is a tensor of shape (seq_len, batch_size) where seq_len is defined as above 
    # This will make sure we have batches where each example (row) is seq_len 
    data = data.view(bsz, seq_len).t().contiguous()
    
    return data.to(device)

def get_batch(source, i):
    """
    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len] # Grab the correct rows 
    # Number of rows = bptt (batch size; number of rows in batch 0, 1, 2, ...)
    # number of columns = length of one example in one batch 
    target = source[i+1:i+1+seq_len].reshape(-1)
    
    return data, target

####################################################################################################
# TRAINING SETUP/UTILITY FUNCTIONS 
####################################################################################################
# Initialize the model ----------------------------------------------------------------------------- 
vocab_size = len(vocab)
d_model = 10 # embedding size 
dropout = 0.1
n_head = 2
d_hid = 30 
n_layers = 6 

model = GPT_own(vocab_size=vocab_size, d_model=d_model, dropout=dropout, n_head=n_head, 
                d_hid=d_hid, n_layers=n_layers)

model = model.to(device)

crit = nn.CrossEntropyLoss() 
lr = 10e-4 
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Train + test functions ---------------------------------------------------------------------------
def train(model, log_freq, train_data):
    """
    Trains the model for one epoch.
    """
    model.train() 
    total_loss = 0.
    mask = generate_square_mask(bptt).to(device)

    num_batches = len(train_data) // bptt 

    for batch, index in enumerate(range(0, train_data.size(0) - 1, bptt)):
        src, tgts = get_batch(train_data, index)
        seq_len = src.size(0)
        
        if seq_len != bptt:  # On the last batch, we might not have enough examples 
            print(f'Last batch sequence length: {seq_len}')
            mask = mask[:seq_len, :seq_len]

        output = model(src, mask) # Pass through the network 

        loss = crit(output.view(-1, vocab_size), tgts) # output is shape (N, C); compare to flat 
        # target tensor 

        optimizer.zero_grad() # prevents gradient accumulation 
        loss.backward() # computes the gradient wrt each parameter 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step() # updates the parameter from their gradients 

        total_loss += loss.item() # update the running loss 
        
        # If we're at a batch size divisible by the frequency at which we want to log, print some 
        # information 
        if (batch % log_freq == 0) and (batch > 0):
            avg_loss = total_loss / log_freq # average loss on the last log_freq batches 
            print(f'Done training batch {batch:4d}/{num_batches:4d} | '
                  f'Avg. training loss on the last {log_freq} batches: {avg_loss:5.2f}')

            total_loss = 0.

def evaluate(model, eval_data):
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    mask = generate_square_mask(bptt).to(device)
    with torch.no_grad(): # don't want to update parameters 
        for i in range(0, eval_data.size(0) - 1, bptt):
            src, tgts = get_batch(eval_data, i)
            seq_len = src.size(0)
            if seq_len != bptt:
                mask = mask[:seq_len, :seq_len]
            output = model(src, mask)
            output_flat = output.view(-1, vocab_size)
            total_loss += seq_len * crit(output_flat, tgts).item()
    return total_loss / (len(eval_data) - 1)

# Function to predict ------------------------------------------------------------------------------
def generate_tune(model, tune_start, predict_length, topk=3):
    model.eval() 

    with torch.no_grad():
        for _ in range(predict_length):
            # Setup 
            tune_start_use = data_process(tune_start)
            tune_start_use = batchify(tune_start_use, 1)
            mask_predict = generate_square_mask(tune_start_use.size(0)).to(device)
            # Predict 
            output = model(tune_start_use, mask_predict)
            logits = output[-1, 0, :]
            # print(logits)
            # top_token_idx = torch.argmax(logits).item() 
            top_token_idx = torch.multinomial(F.softmax(logits), num_samples=1).item()
            top_token = vocab.lookup_token(top_token_idx)
            # print(f'Top token: {vocab.lookup_token(top_token_idx)}')
            tune_start.append(top_token)

    return tune_start 

####################################################################################################
# TRAINING STARTS 
####################################################################################################
def tunes_tokens_to_tensor(tunes):
    return torch.flatten(torch.tensor(vocab(tunes)))

def run_tr_test():
    epochs = 10000 
    log_freq = 100
    eval_batch_size = 16
    scheduler = torch.optim.AdamW(model.parameters())
    best_val_loss = float('inf')
    save_path = pathlib.Path().cwd() / 'output' / 'music_model_state_dict.pt'

    # Train-test split 
    tunes_list_train, tunes_list_test = train_test_split(tunes_list, test_size=0.2)
    print(f'Length of training set: {len(tunes_list_train)}')
    print(f'Length of test set: {len(tunes_list_test)}')

    # Get validation set 
    test_data = tunes_flatten(tunes_list_test)
    test_data = tunes_tokens_to_tensor(test_data)
    test_data = batchify(test_data, eval_batch_size)

    for epoch in range(epochs):    
        time_start = time.time() 

        tunes_list_train = shuffle(tunes_list_train)
        train_data = tunes_flatten(tunes_list_train)
        train_data = tunes_tokens_to_tensor(train_data)
        train_data = batchify(train_data, batch_size)

        train(model, log_freq, train_data)
        val_loss = evaluate(model, test_data)

        tune_start = ['M']
        print(''.join(generate_tune(model, tune_start, 20)))

        # If validation loss is the best we've seen so far, save the model 
        if val_loss < best_val_loss:
            best_val_loss = val_loss 
            print('Saving model....')
            torch.save(model.state_dict(), save_path)

        print(f'End of epoch {epoch + 1} | test loss: {val_loss:5.2f}')
        print('-' * 80)

        time_used = time.time() - time_start 
        print(f'Time elapsed during epoch: {time_used:5.2f}')

        scheduler.step() 

####################################################################################################
# GENERATE MUSIC 
####################################################################################################
def main():
    path_check = pathlib.Path().absolute() / 'output' / 'music_model_state_dict.pt'
    
    if path_check.is_file():
        print('Model found! Loading...')
        model_use = GPT_own(vocab_size=vocab_size, d_model=d_model, dropout=dropout, n_head=n_head, 
                        d_hid=d_hid, n_layers=n_layers)

        model_use = model_use.to(device)

        model_use.load_state_dict(torch.load(path_check))

        for _ in range(20):
            tune_start = ['M']
            print(''.join(generate_tune(model_use, tune_start, 200)))

            tune_start = ['M', ':', '4', '/', '4', 'K', ':', 'D']
            print(''.join(generate_tune(model_use, tune_start, 200)))
    else:
        print('Model not found! Running train-test loop...')
        run_tr_test()

if __name__ == '__main__':
    main()  