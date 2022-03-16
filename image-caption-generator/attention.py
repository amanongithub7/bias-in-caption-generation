import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, decoder_dim, attention_dim, device, encoder_dim=256):
        super(Attention, self).__init__()
        self.decoder_linear = nn.Linear(decoder_dim, attention_dim).to(device)
        self.encoder_linear = nn.Linear(encoder_dim, attention_dim).to(device)
        self.v = nn.Linear(attention_dim, 1).to(device)
        self.tanh = nn.Tanh().to(device)
        self.softmax = nn.Softmax(1).to(device)
        self.device = device

    def forward(self, img_features, hidden_state):
        img_features = img_features.to(self.device)
        hidden_state = hidden_state.to(self.device)
        D_o = self.decoder_linear(hidden_state).unsqueeze(1)
        E_o = self.encoder_linear(img_features)
        att = self.tanh(D_o + E_o)
        e = self.v(att).squeeze(2)
        alpha = self.softmax(e)
        context = (img_features * alpha.unsqueeze(2)).sum(1)
        return context, alpha