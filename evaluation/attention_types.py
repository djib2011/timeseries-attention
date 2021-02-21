import os
import sys

sys.path.append(os.getcwd())

import evaluation


experiment_name = 'attention_types'

columns = ['input_len', 'output_len', 'layer_size', 'encoder_type', 'encoder_layers', 'decoder_type', 'decoder_layers',
           'attention_type', 'attention_scale', 'attention_dropout', 'attention_causal', 'input_length',
           'output_length', 'base_layer_size']

res, tracked = evaluation.run_evaluation(experiment_name, columns=columns, return_results=True)