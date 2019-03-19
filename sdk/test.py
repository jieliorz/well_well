from config.params import Config
from utils.prepare_data import semi_to_dataset,produce_semi

from utils.tokenization import Tokenizer

from utils.data_helper import Dataset

params = Config(project_name='douban',
				model_type='rnn',
				max_length=30,
				is_tgt_label=True,
				n_observations=10).params

t=Tokenizer(params)
produce_semi(params)

semi_to_dataset(params)



Dataset(params)