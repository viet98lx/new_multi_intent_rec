import model
import model_utils
import utils
import data_utils
import loss
import check_point

import argparse
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import scipy.sparse as sp
import random
import os
from torch.utils.tensorboard import SummaryWriter

torch.set_printoptions(precision=8)
parser = argparse.ArgumentParser(description='Train model')

parser.add_argument('--batch_size', type=int, help='batch size of data set (default:32)', default=32)
parser.add_argument('--rnn_units', type=int, help='number units of hidden size lstm', default=16)
parser.add_argument('--rnn_layers', type=int, help='number layers of RNN', default=1)
parser.add_argument('--alpha', type=float, help='coefficient of C matrix in predict item score', default=0.4)
parser.add_argument('--lr', type=float, help='learning rate of optimizer', default=0.001)
parser.add_argument('--dropout', type=float, help='drop out after linear model', default= 0.2)
parser.add_argument('--embed_dim', type=int, help='dimension of linear layers', default=8)
parser.add_argument('--device', type=str, help='device for train and predict', default='cpu')
parser.add_argument('--top_k', type=int, help='top k predict', default=10)
parser.add_argument('--nb_hop', type=int, help='level of correlation matrix', default=1)
parser.add_argument('--epoch', type=int, help='epoch to train', default=30)
parser.add_argument('--epsilon', type=float, help='different between loss of two consecutive epoch ', default=0.00000001)
parser.add_argument('--model_name', type=str, help='name of model', required=True)
parser.add_argument('--data_dir', type=str, help='folder contains data', required=True)
parser.add_argument('--output_dir', type=str, help='folder to save model', required=True)
parser.add_argument('--seed', type=int, help='seed for random', default=1)

args = parser.parse_args()
seed = args.seed
torch.manual_seed(seed)

config_param={}
config_param['rnn_units'] = args.rnn_units
config_param['rnn_layers'] = args.rnn_layers
config_param['dropout'] = args.dropout
config_param['embedding_dim'] = args.embed_dim
config_param['batch_size'] = args.batch_size
config_param['top_k'] = args.top_k
config_param['alpha'] = args.alpha

data_dir = args.data_dir
output_dir = args.output_dir
nb_hop = args.nb_hop

np.random.seed(seed)
random.seed(seed)

train_data_path = data_dir + 'train.txt'
train_instances = utils.read_instances_lines_from_file(train_data_path)
nb_train = len(train_instances)
print(nb_train)

validate_data_path = data_dir + 'validate.txt'
validate_instances = utils.read_instances_lines_from_file(validate_data_path)
nb_validate = len(validate_instances)
print(nb_validate)

test_data_path = data_dir + 'test.txt'
test_instances = utils.read_instances_lines_from_file(test_data_path)
nb_test = len(test_instances)
print(nb_test)

### build knowledge ###

print("---------------------@Build knowledge-------------------------------")
MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs = utils.build_knowledge(train_instances, validate_instances, test_instances)

print("#Statistic")
NB_ITEMS = len(item_dict)
print(" + Maximum sequence length: ", MAX_SEQ_LENGTH)
print(" + Total items: ", NB_ITEMS)

print('---------------------Load correlation matrix-------------------')

if (os.path.isfile(data_dir + 'adj_matrix/r_matrix_' +str(nb_hop)+ 'w.npz')):
    real_adj_matrix = sp.load_npz(data_dir + 'adj_matrix/r_matrix_' + str(nb_hop)+ 'w.npz')
else:
    real_adj_matrix = sp.csr_matrix((NB_ITEMS, NB_ITEMS), dtype="float32")
print('Density of correlation matrix: %.6f' % (real_adj_matrix.nnz * 1.0 / NB_ITEMS / NB_ITEMS))

print('---------------------Create data loader--------------------')
train_loader = data_utils.generate_data_loader(train_instances, config_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=True)
valid_loader = data_utils.generate_data_loader(validate_instances, config_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=False)
test_loader = data_utils.generate_data_loader(test_instances, config_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=False)

print('---------------------Create model------------------------')
print(args.device)
exec_device = torch.device('cuda:{}'.format(args.device[-1]) if (args.device != 'cpu' and torch.cuda.is_available()) else 'cpu')
data_type = torch.float32
rec_sys_model = model.RecSysModel(config_param, MAX_SEQ_LENGTH, item_probs, real_adj_matrix.todense(), exec_device, data_type)
rec_sys_model.to(exec_device, dtype= data_type)

print('-----------------SUMMARY MODEL----------------')
pytorch_total_params = sum(p.numel() for p in rec_sys_model.parameters() if p.requires_grad)
print('number params: %d' % pytorch_total_params)
print(rec_sys_model)
for param in rec_sys_model.parameters():
  print(param.shape)

loss_func = loss.Weighted_BCE_Loss()
optimizer = torch.optim.RMSprop(rec_sys_model.parameters(), lr= args.lr, weight_decay= 1e-6)

try:
    os.makedirs(output_dir, exist_ok = True)
    print("Directory '%s' created successfully" % output_dir)
except OSError as error:
    print("Directory '%s' can not be created" % output_dir)

# checkpoint_dir = output_dir + '/check_point/'
best_model_dir = output_dir + '/best_model/'
try:
    os.makedirs(best_model_dir, exist_ok = True)
    print("Directory '%s' created successfully" % best_model_dir)
except OSError as error:
    print("Directory '%s' can not be created" % best_model_dir)
model_name = args.model_name


top_k = config_param['top_k']
train_display_step = 300
val_display_step = 60
test_display_step = 10
epoch = args.epoch

loss_min = 1000
f1_max = 0
epsilon = args.epsilon

train_losses = []
val_losses = []
train_recalls = []
val_recalls = []
test_losses = []
test_recalls = []

log_dir = 'seed_{}_{}'.format(seed, args.model_name)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir='runs/'+log_dir)
print('-------------------Start Training Model---------------------')

############################ Train Model #############################

# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

for ep in range(epoch):

    rec_sys_model, optimizer, avg_train_loss, avg_train_recall, avg_train_prec, avg_train_f1 = model_utils.train_model(rec_sys_model, loss_func, optimizer, train_loader,
                                                                                         ep, top_k, train_display_step)

    writer.add_scalar("Loss/train", avg_train_loss, ep)
    writer.add_scalar("Recall/train", avg_train_recall, ep)
    writer.add_scalar("Precision/train", avg_train_prec, ep)
    writer.add_scalar("F1/train", avg_train_f1, ep)

    avg_val_loss, avg_val_recall, avg_val_prec, avg_val_f1 = model_utils.validate_model(rec_sys_model, loss_func, valid_loader,
                                                              ep, top_k, val_display_step)

    writer.add_scalar("Loss/val", avg_val_loss, ep)
    writer.add_scalar("Recall/val", avg_val_recall, ep)
    writer.add_scalar("Precision/val", avg_val_prec, ep)
    writer.add_scalar("F1/val", avg_val_f1, ep)

    avg_test_loss, avg_test_recall, avg_test_prec, avg_test_f1 = model_utils.test_model(rec_sys_model, loss_func, test_loader,
                                                            ep, top_k, test_display_step)

    writer.add_scalar("Loss/test", avg_test_loss, ep)
    writer.add_scalar("Recall/test", avg_test_recall, ep)
    writer.add_scalar("Precision/test", avg_test_prec, ep)
    writer.add_scalar("F1/test", avg_test_f1, ep)

    if (avg_test_f1 > f1_max):
        score_matrix = []
        print('Test loss decrease from ({:.6f} --> {:.6f}) '.format(loss_min, avg_test_loss))
        print('Test f1 increase from {:.6f} --> {:.6f}'.format(f1_max, avg_test_f1))
        # check_point.save_ckpt(checkpoint, True, model_name, checkpoint_dir, best_model_dir, ep)
        check_point.save_config_param(output_dir, args.model_name, config_param)
        loss_min = avg_test_loss
        f1_max = avg_test_f1
        torch.save(rec_sys_model, best_model_dir + '/best_' + args.model_name + '.pt')
        print('Can save model')

    print('-' * 100)
    # ckpt_path = checkpoint_dir+model_name+'/epoch_'+str(ep)+'/'
    # utils.plot_loss(train_losses, val_losses, ckpt_path)
    # utils.plot_recall(train_recalls, val_recalls, ckpt_path)
