import torch
import logging


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(
    filename='training.log',
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO
)
logger = logging.getLogger('__file__')

truth_path = './result/ground_truths.txt'
prediction_path = './result/predictions.txt'
perl_path = 'data/SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2/semeval2010_task8_scorer-v1.2.pl'

