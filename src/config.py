import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_cfg = json.loads((PROJECT_ROOT / "data" / "config.json").read_text())

BLS_PARAMS = {
    "start_year": _cfg["start_year"],
    "end_year":   _cfg["end_year"],
    "api_key":    _cfg["api_key"],
    "series_codes_file": PROJECT_ROOT / _cfg["series_codes_file"],
    "output_csv":        PROJECT_ROOT / _cfg["data_dir"] / _cfg["output_csv"],
}

DATA_CONFIG = {
    "data_dir":       PROJECT_ROOT / _cfg["data_dir"],
    "models_dir":     PROJECT_ROOT / _cfg["models_dir"],
    "default_data_file": PROJECT_ROOT / _cfg["data_dir"] / _cfg["output_csv"],
}


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
    'sequence_length':  4,      # using past 4 weeks to predict week + 1
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
