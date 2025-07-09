import os

# --- Path Settings ---
DATA_DIR = './data'
TRAIN_FILE = os.path.join(DATA_DIR, 'train_daily.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test_daily.csv')
RESULT_DIR = 'save_results'

# --- General Parameters ---
SEQUENCE_LENGTH = 90
TARGET_FEATURE = 'Global_active_power'
FEATURES = [
    'Global_active_power', 'Voltage',
    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3','RR','sub_metering_remainder','NBJRR1','NBJRR5','NBJRR10','NBJBROU'
]
NUM_RUNS = 5

# =================== Short-Term Forecast Config (90 days) ===================
SHORT_TERM_HORIZON = 90
SHORT_TERM_SCALER_NAME = 'scaler_short_term.pkl'
SHORT_TERM_PLOT_NAME = 'prediction_short_term_90_days'

# --- LSTM Short-Term ---
LSTM_SHORT_TERM_MODEL_NAME = f'lstm_short_term_{SHORT_TERM_HORIZON}d.pt'
LSTM_SHORT_TERM_PARAMS = {
    'model': {
        'input_size': len(FEATURES),
        'hidden_size': 128,
        'num_layers': 3,
        'dropout': 0.05,
    },
    'train': {
        'learning_rate': 0.00001,
        'epochs': 40,
        'batch_size': 32,
    }
}

# --- Transformer Short-Term ---
TRANSFORMER_SHORT_TERM_MODEL_NAME = f'transformer_short_term_{SHORT_TERM_HORIZON}d.pt'
TRANSFORMER_SHORT_TERM_PARAMS = {
    'model': {
        'input_size': len(FEATURES),
        'd_model': 64,
        'nhead': 4,
        'num_encoder_layers': 4,
        'dim_feedforward': 128,
        'dropout': 0.2
    },
    'train': {
        'learning_rate': 0.01,
        'epochs': 40,
        'batch_size': 16,
    }
}

# --- NEW: Autoformer Short-Term ---
AUTOFORMER_SHORT_TERM_MODEL_NAME = f'autoformer_short_term_{SHORT_TERM_HORIZON}d.pt'
AUTOFORMER_SHORT_TERM_PARAMS = {
    'model': {
        'input_size': len(FEATURES),
        'hidden_size': 64,      # Embedding dimension
        'num_heads': 4,         # Number of attention heads
        'num_layers': 2,        # Number of encoder layers
        'dropout_rate': 0.1,
    },
    'train': {
        'learning_rate': 0.0005,
        'epochs': 40,
        'batch_size': 32,
    }
}


# =================== Long-Term Forecast Config (365 days) ===================
LONG_TERM_HORIZON = 365
LONG_TERM_SCALER_NAME = 'scaler_long_term.pkl'
LONG_TERM_PLOT_NAME = 'prediction_long_term_365_days'

# --- LSTM Long-Term ---
LSTM_LONG_TERM_MODEL_NAME = f'lstm_long_term_{LONG_TERM_HORIZON}d.pt'
LSTM_LONG_TERM_PARAMS = {
    'model': {
        'input_size': len(FEATURES),
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2,
    },
    'train': {
        'learning_rate': 0.0001,
        'epochs': 40,
        'batch_size': 16,
    }
}

# --- Transformer Long-Term ---
TRANSFORMER_LONG_TERM_MODEL_NAME = f'transformer_long_term_{LONG_TERM_HORIZON}d.pt'
TRANSFORMER_LONG_TERM_PARAMS = {
    'model': {
        'input_size': len(FEATURES),
        'd_model': 128,
        'nhead': 8,
        'num_encoder_layers': 4,
        'dim_feedforward': 256,
        'dropout': 0.2
    },
    'train': {
        'learning_rate': 0.0001,
        'epochs': 60,
        'batch_size': 64,
    }
}

# --- NEW: Autoformer Long-Term ---
AUTOFORMER_LONG_TERM_MODEL_NAME = f'autoformer_long_term_{LONG_TERM_HORIZON}d.pt'
AUTOFORMER_LONG_TERM_PARAMS = {
    'model': {
        'input_size': len(FEATURES),
        'hidden_size': 128,     # Embedding dimension
        'num_heads': 8,         # Number of attention heads
        'num_layers': 3,        # Number of encoder layers
        'dropout_rate': 0.2,
    },
    'train': {
        'learning_rate': 0.0001,
        'epochs': 60,
        'batch_size': 64,
    }
}