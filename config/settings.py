import os
import logging


num_classes = 1
num_stacks = 1
num_blocks = 4

lr = 1e-6
batch_size = 1

data_dir = '/tiger_dataset'
test_dir = '/home/miga/downloads/CODE/Face_Detection(human_tiger)/tiger_ssd/config/input'
log_dir = '/home/miga/downloads/CODE/Face_Detection(human_tiger)/tiger_ssd/logdir'
model_dir = '/home/miga/downloads/CODE/Face_Detection(human_tiger)/tiger_ssd/model'
config_dir = '/home/miga/downloads/CODE/Face_Detection(human_tiger)/tiger_ssd/config'

log_level = 'info'
model_path = os.path.join(model_dir, 'latest')
save_steps = 1000

num_workers = 4
num_GPU = 1
device_id = 1

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
