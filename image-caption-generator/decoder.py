import torch
import torch.nn as nn
from attention import Attention

class RNN_Decoder(torch.nn.Module):
  def __init__(self, embedding_dim, fc_units, vocab_size, hidden_size, attention_dim, device):
    super(RNN_Decoder, self).__init__()
    
    self.device = device # cuda:0
    
    self.fc_units = fc_units # fc layer hyperparameter

    # embedding layer
    self.embedding_dim = embedding_dim
    self.gru_hidden_size = hidden_size 
    self.attention_dim = attention_dim

    self.embedding = torch.nn.Embedding(vocab_size, embedding_dim).to(device)

    self.attention = Attention(self.gru_hidden_size, self.attention_dim, device=self.device).to(device)

    self.gru = torch.nn.GRU(input_size=(embedding_dim*2), 
                            hidden_size=self.gru_hidden_size, 
                            batch_first=True).to(device) # GRU

    self.fc1 = nn.Linear(self.gru_hidden_size, self.fc_units).to(device)
    self.fc2 = nn.Linear(self.fc_units, vocab_size).to(device)
    self.softmax = nn.Softmax(dim=1).to(device)

  def forward(self, x, features, hidden):
    is_eval = False
    batch_size = x.shape[0]
    if(x.shape[0] == 1):
      is_eval = True

    # passing features through the attention model
    # shape of context vector = (batch_size, 1, embedding_dim)
    x = x.to(self.device)
    features = features.to(self.device)
    hidden = hidden.to(self.device)
    context_vector, attention_weights = self.attention(features, hidden)

    x = self.embedding(x) # batch_size, max_length, embedding_dim
   
    context_vector = torch.reshape(context_vector, (batch_size, 1, 256))
    
    if is_eval:
      x = torch.unsqueeze(x, 0)

    x = torch.cat((context_vector, x), dim=2)

    # x is BATCH_SIZE, 1, gru_hidden_size
    x, gru_state = self.gru(x)

    x = torch.squeeze(x)
    if is_eval:
      x = torch.unsqueeze(x, 0)
    x = self.fc1(x)

    x = self.fc2(x)
    x = self.softmax(x)
  
    return x, gru_state, attention_weights

  def reset_state(self, batch_size):
    return torch.zeros((batch_size, self.gru_hidden_size))