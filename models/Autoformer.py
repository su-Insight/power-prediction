from torch import nn

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