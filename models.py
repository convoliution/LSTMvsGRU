import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable

import time

import utils


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        
        self.lstm1 = nn.LSTM(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.5)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_size, output_size)
        
        self.init_hidden()
        
        self.loss_history = []
        
        self.output_samples = []
    def init_hidden(self):
        with torch.cuda.device(0):
            self.hidden1 = (Variable(torch.randn(1, 1, self.hidden_size).cuda()),
                            Variable(torch.randn(1, 1, self.hidden_size).cuda()))
            self.hidden2 = (Variable(torch.randn(1, 1, self.hidden_size).cuda()),
                            Variable(torch.randn(1, 1, self.hidden_size).cuda()))
    def forward(self, input_tensor):
        output, self.hidden1 = self.lstm1(input_tensor, self.hidden1)
        output               = self.dropout1(output)
        output, self.hidden2 = self.lstm2(output, self.hidden2)
        output               = self.dropout2(output)
        output               = self.linear(output.squeeze(1))
        output               = func.log_softmax(output, dim=-1)
        return output
    def train(self, source, sequence_length, criterion, optimizer, num_epochs=1, iters_per_epoch=64):
        start_time = time.time()

        with torch.cuda.device(0):
            self.cuda()
            for epoch in range(num_epochs):
                epoch_loss_total = 0
                for iteration in range(iters_per_epoch):
                    self.zero_grad()
                    self.init_hidden()

                    input_seq, target_seq = utils.sample_sequence(source, sequence_length)
                    input_tensor = Variable(utils.sequence_to_one_hot(input_seq).float().cuda())
                    target_tensor = Variable(utils.sequence_to_target(target_seq).long().cuda())

                    loss = 0
                    for i_letter in range(sequence_length):
                        output = self(input_tensor[i_letter].unsqueeze(0))
                        loss += criterion(output, target_tensor[i_letter])

                    loss.backward()
                    epoch_loss_total += loss.data[0]/sequence_length
                    optimizer.step()
                    print("Epoch {} Iteration {}/{}".format(epoch+1, iteration+1, iters_per_epoch), end='\r')
                self.loss_history.append(epoch_loss_total/iters_per_epoch)
                print("Epoch {} loss: {}, {} elapsed"
                      .format(epoch+1, self.loss_history[-1], utils.time_since(start_time)))
                self.output_samples.append(self.generate())
    # from http://norvig.com/mayzner.html, 't' is the most common first letter in English words
    def generate(self, output_length=128, seed='T'):
        input_tensor = Variable(utils.sequence_to_one_hot(seed).float().cuda())
        output_sequence = seed
        
        self.init_hidden()
        
        for i_letter in range(output_length):
            pred = self(input_tensor[0].unsqueeze(0))
            _, pred = pred.data.topk(1)
            pred = pred[0][0] # extract element from tensor
            
            letter = utils.all_letters[pred]
            output_sequence += letter
            
            input_tensor = Variable(utils.sequence_to_one_hot(letter).float().cuda())
        return output_sequence
    
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        
        self.gru1 = nn.GRU(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.5)
        self.gru2 = nn.GRU(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_size, output_size)
        
        self.init_hidden()
        
        self.loss_history = []
        
        self.output_samples = []
    def init_hidden(self):
        with torch.cuda.device(0):
            self.hidden1 = (Variable(torch.randn(1, 1, self.hidden_size).cuda()),
                            Variable(torch.randn(1, 1, self.hidden_size).cuda()))
            self.hidden2 = (Variable(torch.randn(1, 1, self.hidden_size).cuda()),
                            Variable(torch.randn(1, 1, self.hidden_size).cuda()))
    def forward(self, input_tensor):
        output, self.hidden1 = self.gru1(input_tensor, self.hidden1)
        output               = self.dropout1(output)
        output, self.hidden2 = self.gru2(output, self.hidden2)
        output               = self.dropout2(output)
        output               = self.linear(output.squeeze(1))
        output               = func.log_softmax(output, dim=-1)
        return output
    def train(self, source, sequence_length, criterion, optimizer, num_epochs=1, iters_per_epoch=64):
        start_time = time.time()

        with torch.cuda.device(0):
            self.cuda()
            for epoch in range(num_epochs):
                epoch_loss_total = 0
                for iteration in range(iters_per_epoch):
                    self.zero_grad()
                    self.init_hidden()

                    input_seq, target_seq = utils.sample_sequence(source, sequence_length)
                    input_tensor = Variable(utils.sequence_to_one_hot(input_seq).float().cuda())
                    target_tensor = Variable(utils.sequence_to_target(target_seq).long().cuda())

                    loss = 0
                    for i_letter in range(sequence_length):
                        output = self(input_tensor[i_letter].unsqueeze(0))
                        loss += criterion(output, target_tensor[i_letter])

                    loss.backward()
                    epoch_loss_total += loss.data[0]/sequence_length
                    optimizer.step()
                    print("Epoch {} Iteration {}/{}".format(epoch+1, iteration+1, iters_per_epoch), end='\r')
                self.loss_history.append(epoch_loss_total/iters_per_epoch)
                print("Epoch {} loss: {}, {} elapsed"
                      .format(epoch+1, self.loss_history[-1], utils.time_since(start_time)))
                self.output_samples.append(self.generate())
    # from http://norvig.com/mayzner.html, 't' is the most common first letter in English words
    def generate(self, output_length=128, seed='T'):
        input_tensor = Variable(utils.sequence_to_one_hot(seed).float().cuda())
        output_sequence = seed
        
        self.init_hidden()
        
        for i_letter in range(output_length):
            pred = self(input_tensor[0].unsqueeze(0))
            _, pred = pred.data.topk(1)
            pred = pred[0][0] # extract element from tensor
            
            letter = utils.all_letters[pred]
            output_sequence += letter
            
            input_tensor = Variable(utils.sequence_to_one_hot(letter).float().cuda())
        return output_sequence