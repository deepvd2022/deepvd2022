
import numpy as np
import pandas as pd
import logging
from embeddings.longpath import Longpath
from embeddings.ns import NaturalSeq
import os
import tempfile
import argparse
import time
from pathlib import Path
from data_loader import MyLargeDataset
# from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ndcg_score
from model import DLVP, DLVP_nocc
from tqdm import tqdm
try:
    import cPickle as pickle
except:
    import pickle

from config import *
import nni
pd.set_option('display.max_columns', None)

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Additional Info when using cuda
if device.type == 'cuda':
    print('#', torch.cuda.get_device_name(0))
    # print('# Memory Usage:')
    # print('# Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    # print('# Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))





def get_params():
    # args
    parser = argparse.ArgumentParser(description='Test for argparse')


    parser.add_argument('--input_path', help='input_path', type=str,
                        default=INPUT_PATH)
    parser.add_argument('--output_path', help='output_path', type=str,
                        default=OUTPUT_PATH)

    parser.add_argument('--input_dim', help='input_dim', type=int, default=128)
    parser.add_argument('--output_dim', help='output_dim', type=int, default=2)

    parser.add_argument('--epoch', help='epoch', type=int, default=50)
    parser.add_argument('--hop', help='epoch', type=int, default=1)

    parser.add_argument('--lp_dim', help='lp_dim', type=int, default=128)
    parser.add_argument('--ns_dim', help='ns_dim', type=int, default=128)

    args, _ = parser.parse_known_args()
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    return args

def train_socre(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    res = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }
    return res

def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    y_true = []
    y_pred = []
    for data in loader:
        with torch.no_grad():
            pred = model(data)
            # pred = pred.argmax(dim=1)
            label = data[-1]

        _, pred = torch.max(pred.data, 1)

        y_true.append(label.cpu())
        y_pred.append(pred.cpu())
        # correct += pred.eq(label).sum().item()

    total = len(loader.dataset)

    # return correct / total
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)

    y_true_np = np.asarray([np.asarray(y_true)])
    y_pred_np = np.asarray([np.asarray(y_pred)])
    ndcg = ndcg_score(y_true_np, y_pred_np)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    res = {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        'ndcg': ndcg,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }
    # if res['f1'] < 0.01:
    #     logging.info("y_true: {}".format(y_true))
    #     logging.info("y_pred: {}".format(y_pred))
    return res

def print_result(phase, score, epoch = -1):
    if phase in ['train', 'vali']:
        score['phase'] = phase
        score['epoch'] = epoch
        df = pd.DataFrame([score])
        print(df.head())
    else:
        score['phase'] = "res==== {}".format(phase)
        df = pd.DataFrame([score])
        print(df.head())


def collate_batch(batch):
    # _data = batch[0]
    # return _data
    y = [ data.y for data in batch ]
    y = torch.tensor(y , dtype=torch.long)

    return (batch, y)



def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def count_params2(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_figures(history, trail_id, mode='train'):
    figure_save_path = "./figures"
    Path(figure_save_path).mkdir(parents=True, exist_ok=True)

    accuracy = [a['accuracy'] for a in history]
    recall = [a['recall'] for a in history]
    precision = [a['precision'] for a in history]
    f1 = [a['f1'] for a in history]

    plt.figure()
    plt.title("dlvp_{}_{}".format(trail_id, mode))

    if mode == 'loss':
        loss = [a['loss'] for a in history]
        plt.plot(loss, label="Loss")
    else:
        plt.plot(accuracy, label="Accuracy")
        plt.plot(recall, label="Recall")
        plt.plot(precision, label="Precision")
        plt.plot(f1, label="F1")

    plt.xlabel("# Epoch")
    plt.ylabel("Score")
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig(figure_save_path + "/dlvp_{}_{}.pdf".format(trail_id, mode))


def train(params, tuner_params, trail_id, train_dataset, vali_dataset, test_dataset, model_save_path, writer, plot=False, print_grad=False, max_patience=20):
    learning_rate = tuner_params['learning_rate']
    hid_size = tuner_params['hidden_dim']
    batch_size = tuner_params['batch_size']
    dropout = tuner_params['dropout']


    params = vars(params)
    params['hidden_dim'] = hid_size
    params['batch_size'] = batch_size
    params['dropout'] = dropout


    # logging.info("train: {}, vali: {}, test: {}".format(len(train_dataset), len(vali_dataset), len(test_dataset)))
    print("train: {}, vali: {}, test: {}".format(len(train_dataset), len(vali_dataset), len(test_dataset)))

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    vali_loader = DataLoader(vali_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    # for batch in loader:
    #     logging.info("batch.y: {}".format(batch.y))


    # build model


    # gpu_tracker = MemTracker()  # define a GPU tracker
    # gpu_tracker.track()
    # print(model)
    # exit()


    lp_obj = Longpath(args.input_path, args.output_path)
    lp_obj.load_model()
    lp_weight_matrix = torch.tensor( lp_obj.vectors, dtype=torch.float)
    ns_obj = NaturalSeq(args.input_path, args.output_path)
    ns_obj.load_model()
    ns_weight_matrix = torch.tensor( ns_obj.vectors, dtype=torch.float)

    model = DLVP_nocc(params, lp_weight_matrix, ns_weight_matrix)
    model.to(device)

    total_params_1 = count_params(model)
    total_params_2 = count_params2(model)
    print("=== Total Parameters: 1: {} 2: {}".format(total_params_1, total_params_2))

    # gpu_tracker.track()

    # print("len(model.convs):", len(model.convs))

    opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    # opt = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    min_valid_loss = np.inf
    best_f1_score = -1
    best_model_path = ""
    best_model_list = [] # 存储最好的 model_path 及其 F1-score，排序后，取 top5 个 model，进行测试
    patience_counter = 0

    train_accuracies, vali_accuracies = list(), list()
    train_history = []
    valid_history = []

    # train
    for epoch in range(params['epoch']):
        logging.info("=== now epoch: %d" % epoch)
        total_loss = 0
        model.train()
        bb = 0
        for batch in tqdm(loader):

            opt.zero_grad() # 清空梯度
            pred = model(batch).to(device)

            label = batch[-1].to(device)


            loss = model.loss(pred, label)

            # gpu_tracker.track()

            loss.backward() # 反向计算梯度，累加到之前梯度上
            opt.step() # 更新参数
            total_loss += loss.item()

            # gpu_tracker.track()

            # delete caches
            del pred, loss
            # torch.cuda.empty_cache()

            # gpu_tracker.track()

        total_loss /= len(loader.dataset)
        # writer.add_scalar("loss", total_loss, epoch)

        # validate
        train_score = test(loader, model)
        train_score['loss'] = total_loss
        vali_score = test(vali_loader, model)
        train_history.append(train_score)
        valid_history.append(vali_score)


        # train_accuracies.append(train_score['accuracy'])
        vali_accuracies.append(vali_score['accuracy'])


        print("Epoch: {}, loss: {:.6f}".format(epoch, total_loss))
        if total_loss < min_valid_loss:
            print("Training Loss Decreased: {:.6f} --> {:.6f}.".format(min_valid_loss, total_loss))
            # logging.info("Training Loss Decreased: {:.6f} --> {:.6f}.".format(min_valid_loss, total_loss))
            min_valid_loss = total_loss

        if vali_score['f1'] > best_f1_score:
            patience_counter = 0
            # Saving State Dict
            best_model_path = model_save_path + "/dlvp_model_{}_epoch{}.pth".format(trail_id, epoch)

            torch.save(model.state_dict(), best_model_path)
            best_model_list.append( [vali_score['f1'], best_model_path] )

            print("New best F1: {:.4f} --> {:.4f}. Saved model: {}".format(best_f1_score, vali_score['f1'], best_model_path))
            # logging.info("New best F1: {:.4f} --> {:.4f}. Saved model: {}".format(best_f1_score, vali_score['f1'], best_model_path))
            best_f1_score = vali_score['f1']
        else:
            patience_counter += 1

        # report intermediate result
        nni.report_intermediate_result(vali_score['f1'])


        print_result("train", train_score, epoch)
        print_result("vali", vali_score, epoch)
        # if patience_counter == max_patience:
        #     break


        # writer.add_scalar("test_accuracy", vali_score['accuracy'], epoch)

    # report final result
    nni.report_final_result(vali_score['f1'])

    # Test
    logging.info("=== Test ===")
    best_model_list = sorted(best_model_list, key=lambda x: x[0], reverse=True)
    for _, best_model_path in best_model_list[:5]:
        if os.path.exists(best_model_path):
            print("loading the best model: %s" % best_model_path)
            model.load_state_dict(torch.load(best_model_path))
            test_score = test(test_loader, model)
            print_result("test", test_score)



    if plot:
        plt.plot(train_accuracies, label="Train accuracy")
        plt.plot(vali_accuracies, label="Validation accuracy")
        plt.xlabel("# Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc='upper right')
        # plt.show()
        plt.savefig(model_save_path + "/dlvp_model_accuracy.pdf")

    plot_figures(train_history, trail_id, 'train')
    plot_figures(train_history, trail_id, 'loss')
    plot_figures(valid_history, trail_id, 'vali')
    print("=== Total Trainable Parameters: {}".format(total_params_1))
    print("=== Tuner Parameters: {}".format(tuner_params))
    return model

if __name__ == '__main__':
    args = get_params()

    # log file
    now_time = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
    log_file = "{}/logs/{}_final.log".format(BASE_DIR, now_time)
    Path('logs').mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(filename)s line: %(lineno)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=log_file)

    logging.info("args: {}".format(args))

    try:
        i = 0 # index of random_dataset

        tuner_params = nni.get_next_parameter()
        print("tuner_params", tuner_params)

        # 1. data_loader
        rand_key_file_path = RAND_DATA_PATH + "/lv0_func_keys_{}_{}.txt".format(i, 'train')
        dataset_savepath = args.output_path + "/datasets_{}_{}".format(i, 'train')
        Path(dataset_savepath).mkdir(parents=True, exist_ok=True)
        train_dataset = MyLargeDataset(dataset_savepath, args, rand_key_file_path)

        rand_key_file_path = RAND_DATA_PATH + "/lv0_func_keys_{}_{}.txt".format(i, 'vali')
        dataset_savepath = args.output_path + "/datasets_{}_{}".format(i, 'vali')
        Path(dataset_savepath).mkdir(parents=True, exist_ok=True)
        vali_dataset = MyLargeDataset(dataset_savepath, args, rand_key_file_path)

        rand_key_file_path = RAND_DATA_PATH + "/lv0_func_keys_{}_{}.txt".format(i, 'test')
        dataset_savepath = args.output_path + "/datasets_{}_{}".format(i, 'test')
        Path(dataset_savepath).mkdir(parents=True, exist_ok=True)
        test_dataset = MyLargeDataset(dataset_savepath, args, rand_key_file_path)

        # 2. train & test
        # writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

        trail_id = nni.get_trial_id()

        model_save_path = args.output_path + '/models_{}'.format(i)
        Path(model_save_path).mkdir(parents=True, exist_ok=True)
        model = train(args, tuner_params, trail_id, train_dataset, vali_dataset, test_dataset, model_save_path, writer=None, plot=False, print_grad=False)


    except Exception as exception:
        logging.exception(exception)
        raise
    logging.info("done")