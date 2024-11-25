import torch
from torch.nn import functional as F
import torch.nn as nn
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

class Encoder(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), config.hidden_size)
        self.lstm = nn.LSTM(
            config.hidden_size,
            config.hidden_size,
            bidirectional=config.bidirectional,
            num_layers=config.layer_dim, 
            batch_first=True, 
            dropout=config.dropout
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.lstm(embedded)

        return output, hidden
    
class Decoder(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.MAX_LENGTH = vocab.max_sentence_length
        self.vocab = vocab
        
        self.embedding = nn.Embedding(len(vocab), config.hidden_size)
        self.lstm = nn.LSTM(
            config.hidden_size,
            config.hidden_size,
            bidirectional=config.bidirectional,
            num_layers=config.layer_dim, 
            batch_first=True, 
            dropout=config.dropout
        )
        self.out = nn.Linear(config.hidden_size, len(vocab))

    def forward(self, encoder_outputs: torch.Tensor, encoder_hidden: torch.Tensor, target_tensor: torch.Tensor):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=encoder_outputs.device).fill_(self.vocab)
        
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        for i in range(self.MAX_LENGTH):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            # Teacher forcing: Feed the target as the next input
            decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        
        return decoder_outputs, decoder_hidden

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output)
        
        return output, hidden
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        pass

@META_ARCHITECTURE.register()
class LSTM_Model(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        self.vocab_size = len(vocab)
        
        self.encoder = Encoder(config.encoder, vocab)
        self.decoder = Decoder(config.decoder, vocab)
        
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        encoder_outs, hidden_states = self.encoder(x)

        outs, _ = self.decoder(encoder_outs, hidden_states, labels)
        
        print(outs.shape)
        print(labels.shape)
        raise

        loss = self.loss(outs.reshape(-1, self.vocab_size), labels.reshape(-1))
    
