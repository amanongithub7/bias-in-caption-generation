import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, decoder_dim, attention_dim, encoder_dim=256):
        super(Attention, self).__init__()
        self.decoder_linear = nn.Linear(decoder_dim, attention_dim)
        self.encoder_linear = nn.Linear(encoder_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, img_features, hidden_state):
        D_o = self.decoder_linear(hidden_state).unsqueeze(1)
        E_o = self.encoder_linear(img_features)
        att = self.tanh(D_o + E_o)
        e = self.v(att).squeeze(2)
        alpha = self.softmax(e)
        context = (img_features * alpha.unsqueeze(2)).sum(1)
        return context, alpha