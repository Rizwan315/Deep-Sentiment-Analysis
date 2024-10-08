
import torch
import torch.nn as nn
from transformers import RobertaModel
from torchcrf import CRF

class SentimentModel(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(SentimentModel, self).__init__()
        # Load RoBERTa for word embeddings
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        # ID-CNN for feature extraction
        self.cnn = nn.Conv1d(in_channels=768, out_channels=hidden_units, kernel_size=3, padding=1)
        # BiLSTM for sequence modeling
        self.bilstm = nn.LSTM(input_size=hidden_units, hidden_size=hidden_units, num_layers=1, batch_first=True, bidirectional=True)
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        # Linear layer for classification
        self.fc = nn.Linear(hidden_units * 2, 3)  # Output: 3 classes (positive, negative, neutral)
        # CRF for sequence tagging
        self.crf = CRF(3, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        # Pass through RoBERTa
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state.permute(0, 2, 1)
        # CNN for feature extraction
        cnn_output = self.cnn(sequence_output).permute(0, 2, 1)
        # BiLSTM for sequence modeling
        lstm_output, _ = self.bilstm(cnn_output)
        lstm_output = self.dropout(lstm_output)
        # Linear layer for emission scores
        emissions = self.fc(lstm_output)
        # If labels provided, compute the loss using CRF
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
            return loss
        else:
            return self.crf.decode(emissions, mask=attention_mask.byte())

print("Model architecture updated with RoBERTa, ID-CNN, BiLSTM, and CRF.")
