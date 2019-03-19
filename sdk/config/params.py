from collections import defaultdict
import os
import yaml


class Config:
    """
    main config for all the model
    """
    def __init__(self, 
                project_name,
                model_type,
                is_tgt_label=False,
                update_semi=False,
                update_vocab=False,
                update_dataset=False,
                n_observations=None
                                ):


        self.init_params={

            # file name 也会作为模型的名字
            'project_name': project_name,

            # data
            'update_semi': update_semi,
            'update_vocab': update_vocab,
            'update_dataset': update_dataset,
            'n_observations': n_observations,
            'is_tgt_label':is_tgt_label
            }


        # dirs
        # raw_data 原始文件 ， src=source， tgt=target

        #关于模型，生成的dir路径
        raw_data_dir = os.path.join('../data',project_name,'raw') 
        semi_dir = os.path.join('../data',project_name,'semi') 
        dataset_dir = os.path.join('../data',project_name,'dataset') 
        vocab_dir = os.path.join('../release',project_name,'vocab')
        save_dir = os.path.join('../release',project_name,'model',model_type)

        # 模型文件存储地
        model_dir = os.path.join('model',model_type)

        self._dir_create(semi_dir)
        self._dir_create(dataset_dir)
        self._dir_create(vocab_dir)
        self._dir_create(save_dir)
        self._dir_create(model_dir)

        # semi_dir 下有行行对应的两个文件
        src_file = os.path.join(semi_dir,'src.txt')
        tgt_file = os.path.join(semi_dir,'tgt.txt')
        dataset_file = os.path.join(dataset_dir,'dataset.tfrecord')

        vocab_file = os.path.join(vocab_dir,'vocab.txt')
        save_file = os.path.join(save_dir,'model.ckpt')


        self.init_params['src_file'] = src_file
        self.init_params['tgt_file'] = tgt_file

        self.init_params['vocab_file'] = vocab_file
        self.init_params['save_file'] = save_file

        
        self.init_params['raw_data_dir'] = raw_data_dir
        self.init_params['model_dir'] = model_dir
        self.init_params['dataset_file'] = dataset_file





    def _dir_create(self,file_dir):
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)


    @property
    def params(self):
        model_params_file = os.path.join(self.init_params['model_dir'],'model_params.yml')
        with open(model_params_file,'r') as f:
            model_params = yaml.load(f)
        self.init_params.update(model_params)
        return self.init_params
        

