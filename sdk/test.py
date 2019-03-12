from config.params import Config
from utils.prepare_data import semi_to_dataset

# from utils.tokenization import Tokenizer

from utils.data_helper import Dataset

params = Config(project_name='douban',
				model_type='rnn',
				is_tgt_label=True,
				n_observations=10).params

# t=Tokenizer(params)


semi_to_dataset(params)



Dataset(params['dataset_file'])