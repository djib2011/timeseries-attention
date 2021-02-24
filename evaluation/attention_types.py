import argparse
import os
import sys

sys.path.append(os.getcwd())

import evaluation


parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', help='Run script in debug mode')

args = parser.parse_args()


experiment_name = 'attention_types'

columns = ['input_len', 'output_len', 'layer_size', 'encoder_type', 'encoder_layers', 'decoder_type', 'decoder_layers',
           'attention_type', 'attention_scale', 'attention_dropout', 'attention_causal', 'input_length',
           'output_length', 'base_layer_size']

evaluation.run_evaluation(experiment_name, columns=columns, debug=args.debug)
