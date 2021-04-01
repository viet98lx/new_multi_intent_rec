import os
import torch
import utils
import argparse
import check_point
import model
import scipy.sparse as sp
import data_utils

def generate_predict(model, data_loader, result_file, reversed_item_dict, number_predict, batch_size):
    device = model.device
    nb_test_batch = len(data_loader.dataset) // batch_size
    if len(data_loader.dataset) % model.batch_size == 0:
        total_batch = nb_test_batch
    else :
        total_batch = nb_test_batch + 1
    print("Total Batch in data set %d" % total_batch)
    model.eval()
    with open(result_file, 'w') as f:
        # f.write('Predict result: ')
        for i, data_pack in enumerate(data_loader,0):
            data_x, data_seq_len, data_y = data_pack
            x_ = data_x.to_dense().to(dtype = model.d_type, device = device)
            real_batch_size = x_.size()[0]
            hidden = model.init_hidden(real_batch_size)
            y_ = data_y.to(dtype = model.d_type, device = device)
            predict_ = model(x_, data_seq_len, hidden)
            sigmoid_pred = torch.sigmoid(predict_)
            topk_result = sigmoid_pred.topk(dim=-1, k= number_predict, sorted=True)
            indices = topk_result.indices
            # print(indices)
            values = topk_result.values

            for row in range(0, indices.size()[0]):
                f.write('ground_truth | ')
                ground_truth = y_[row].nonzero().squeeze(dim=-1)
                for idx_key in range(0, ground_truth.size()[0]):
                    f.write(str(reversed_item_dict[ground_truth[idx_key].item()]) + " ")
                f.write('\n')
                f.write('predicted_items ')
                for col in range(0, indices.size()[1]):
                    f.write('| ' + str(reversed_item_dict[indices[row][col].item()]) + ':%.8f' % (values[row][col].item()) + ' ')
                f.write('\n')

parser = argparse.ArgumentParser(description='Generate predict')
parser.add_argument('--ckpt_dir', type=str, help='folder contains check point', required=True)
parser.add_argument('--model_name', type=str, help='name of model', required=True)
# parser.add_argument('--epoch', type=int, help='last epoch before interrupt', required=True)
parser.add_argument('--data_dir', type=str, help='folder contains data', required=True)
parser.add_argument('--device', type=str, help='device for train and predict', default='cpu')
parser.add_argument('--nb_hop', type=int, help='level of correlation matrix', default=1)
parser.add_argument('--batch_size', type=int, help='batch size predict', default=8)
parser.add_argument('--nb_predict', type=int, help='number items predicted', default=30)
parser.add_argument('--log_result_dir', type=str, help='folder to save result', required=True)

args = parser.parse_args()

prefix_model_name = args.model_name
ckpt_dir = args.ckpt_dir
data_dir = args.data_dir
real_adj_matrix = sp.load_npz(data_dir + 'adj_matrix/r_matrix_'+ str(args.nb_hop) + 'w.npz')

exec_device = torch.device('cuda:{}'.format(args.device[-1]) if ('gpu' in args.device and torch.cuda.is_available()) else 'cpu')
data_type = torch.float

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

print("@Build knowledge")
MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs = utils.build_knowledge(train_instances, validate_instances, test_instances)
# print("first item in dict ", reversed_item_dict[0])
print("#Statistic")
NB_ITEMS = len(item_dict)
print(" + Maximum sequence length: ", MAX_SEQ_LENGTH)
print(" + Total items: ", NB_ITEMS)

batch_size = args.batch_size
# train_loader = data_utils.generate_data_loader(train_instances, load_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=True)
# valid_loader = data_utils.generate_data_loader(validate_instances, load_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=False)
test_loader = data_utils.generate_data_loader(test_instances, batch_size, item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=False)

load_model = torch.load(ckpt_dir+'/'+prefix_model_name+'.pt',  map_location='cpu')
load_model = load_model.to(device = exec_device, dtype = data_type)
load_model.device = exec_device

log_folder = os.path.join(args.log_result_dir, prefix_model_name)
if(not os.path.exists(log_folder)):
  try:
    os.makedirs(log_folder, exist_ok = True)
    print("Directory '%s' created successfully" % log_folder)
  except OSError as error:
      print("OS folder error")

nb_predict = args.nb_predict
result_file = log_folder + '/' + prefix_model_name + '_predict_top_' + str(nb_predict) + '.txt'
generate_predict(load_model, test_loader, result_file, reversed_item_dict, nb_predict, batch_size)