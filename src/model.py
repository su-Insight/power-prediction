# src/model.py

import torch
import torch.nn as nn
import math


class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMForecast, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        prediction = self.linear(last_time_step_out)
        return prediction


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
class TransformerForecast(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, output_size, dropout=0.1):
        super(TransformerForecast, self).__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.output_projection = nn.Linear(d_model, output_size)

    def forward(self, src):
        src = self.input_projection(src) * math.sqrt(self.d_model)
        output = self.transformer_encoder(src)
        output = output[:, -1, :]
        prediction = self.output_projection(output)
        return prediction


class Autoformer(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_heads=4, num_layers=2, dropout_rate=0.2):
        super(Autoformer, self).__init__()
        self.input_embedding = nn.Linear(input_size, hidden_size)
        self.decomposition = DecompositionLayer(kernel_size=25)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout_rate,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size, seq_len, num_features = x.size()
        assert num_features == self.input_embedding.in_features, \
            f"Input features ({num_features}) do not match expected features ({self.input_embedding.in_features})."
        x = self.input_embedding(x)
        trend, seasonal = self.decomposition(x)
        seasonal = seasonal.permute(1, 0, 2)
        seasonal_encoded = self.encoder(seasonal)
        seasonal_encoded = seasonal_encoded.permute(1, 0, 2)
        combined = trend + seasonal_encoded
        output = combined[:, -1, :]
        output = self.fc(self.dropout(output))
        return output

class DecompositionLayer(nn.Module):
    def __init__(self, kernel_size):
        super(DecompositionLayer, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        trend = self.moving_avg(x)
        trend = trend.permute(0, 2, 1)
        seasonal = x.permute(0, 2, 1) - trend
        return trend, seasonal