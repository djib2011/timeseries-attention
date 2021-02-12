import argparse
import sys
import os

sys.path.append(os.getcwd())

import networks
import datasets
import training
import config

experiment_name = 'attention_types'

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', help='Run in debug mode: Print lots of diagnostic messages.')
args = parser.parse_args()

data = datasets.seq2seq_generator(config.data.path, batch_size=config.training.BATCH_SIZE)

hp_comb_dict = {'input_seq_length': [18],
                'output_seq_length': [6],
                'base_layer_size': [128],
                'encoder_type': ['bi'],
                'decoder_type': ['uni'],
                'num_encoder_layers': [2],
                'num_decoder_layers': [1],
                'attention_type': ['none', 'mul', 'add', 'self-br', 'self-ar'],
                'attention_scale': [False],
                'attention_dropout': [0.],
                'attention_causal': [False]}

hp_generator = training.make_runs(hp_comb_dict)

for hp in hp_generator:

    model_gen, hp = networks.get(**hp)

    run_name = '__'.join(['{}_{}'.format(k, v) for k, v in hp.items()])

    run_id = training.register_experiment(experiment_name, run_name, debug=args.debug)

    print('run name:', run_name)
    print('run id:', run_id)

    training.run_training(model_gen, hp, data, run_id, num_runs=config.training.NUM_RUNS, debug=args.debug,
                          batch_size=config.training.BATCH_SIZE, epochs=config.training.EPOCHS,
                          warmup=config.training.WARMUP, patience=config.training.PATIENCE)
