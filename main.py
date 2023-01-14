import logging
from trainer import train, infer
from argparse import ArgumentParser
import config
import subprocess


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--train_data", type=str,
                        default='./data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT')
    parser.add_argument("--test_data", type=str,
                        default='./data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT')
    parser.add_argument("--num_classes", type=int, default=19, help='number of relation classes')
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--gradient_acc_steps", type=int, default=2, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--num_epochs", type=int, default=30, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.00007, help="learning rate")
    parser.add_argument("--model_size", type=str, default='bert-base-uncased')
    parser.add_argument("--train", type=int, default=1, help="0: Don't train, 1: train")
    parser.add_argument("--infer", type=int, default=1, help="0: Don't infer, 1: Infer")
    args = parser.parse_args()

    Rmodel, test_loader = train(args)
    infer(Rmodel, test_loader, save=True)
    '''
    # perl语言文件的源程序
    process = subprocess.Popen([
        "perl",
        config.perl_path,
        config.prediction_path,
        config.truth_path
    ], stdout=subprocess.PIPE)
    for line in str(process.communicate()[0].decode("utf-8")).split("\\n"):
        print(line)
    '''
