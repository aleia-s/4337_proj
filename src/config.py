# src/config.py
import json
from pathlib import Path

_cfg = json.loads(
    (Path(__file__).parent.parent / "data" / "config.json").read_text()
)

JSON_KEYS = (
    "start_year", "end_year",
    "api_key", "series_codes_file",
    "data_dir", "models_dir", "output_csv",
)
FETCH_CONFIG = {k: _cfg[k] for k in JSON_KEYS}

BLS_PARAMS  = FETCH_CONFIG
DATA_CONFIG = {"data_dir": FETCH_CONFIG["data_dir"],
               "models_dir": FETCH_CONFIG["models_dir"],
               "default_data_file": FETCH_CONFIG.get("output_csv")}


# LSTNet / model‐training parameters 
MODEL_CONFIG = {
    'num_features':       5,
    'conv_out_channels':  32,
    'gru_hidden_size':    64,
    'skip_lengths':       [4, 24],
    'skip_hidden_size':   16,
    'ar_window':          7,
    'dropout':            0.2
}

TRAINING_CONFIG = {
    'epochs':           100,
    'batch_size':       32,
    'learning_rate':    1e-3,
    'sequence_length':  12,
    'test_size':        0.2,
    'val_size':         0.2
}

VISUALIZATION_CONFIG = {
    'figure_size':        (10,5),
    'dpi':                 300,
    'font_family':        'Times New Roman',
    'font_size':           10,
    'title_size':          12,
    'label_size':          10,
    'tick_size':            8,
    'legend_size':          8,
    'colors': {
      'actual':   'black',
      'pred':     'red',
      'mape_bar': '#f0f0f0',
      'mae_bar':   'black',
      'mse_bar':   'white',
    },
    'line_styles': {
      'actual':   '-',
      'pred':     '--',
    },
    'line_width':       1.5,
    'grid_alpha':       0.7,
    'grid_style':      '--',
    'bar_width':       0.25,
    'bar_patterns': {
      'mape': '',
      'mae':  '',
      'mse':  'xxx',
    },
    'show_titles':        True,
    'show_value_labels':  True,
    'value_label_decimals': {
      'mape': 1,
      'mae':  3,
      'mse':  3
    },
    'label_offset':      0.1,
    'margin_top':        0.2,
    'x_rotation':        45,
    'legend_loc':      'upper right',
}
