import chardet

import os
import random
import chardet
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.nn import Conv2D, Linear
# 引入VisualDL库，并设定保存作图数据的文件位置
from visualdl import LogWriter

# log_writer = LogWriter(logdir="./log")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
BATCH_SIZE = 256
EPOCH_NUM = 1

def load_text(file_name):
    try:
        with open(file_name, "rb") as f:
            f_read = f.read()
            f_cha_info = chardet.detect(f_read)
            final_data = f_read.decode(f_cha_info['encoding'])
            return final_data, True
    except FileNotFoundError:
        return str(None), False


def load_data(mode='train', string=None):
    # 从文件导入数据
    file_name = ''
    # 根据输入mode参数决定使用训练集，验证集还是测试
    if string:
        if string == 'train':
            file_name = './train.npy'
        elif string == 'eval':
            file_name = './test.npy'
        else:
            file_name = './jsp_result_slice/' + string
    else:
        if mode == 'train':
            file_name = './train.npy'
        elif mode == 'eval':
            file_name = './test.npy'

    data = np.load(file_name)

    feature_num = data.shape[1] - 1
    # 计算训练集的最大值，最小值
    maximums = np.max(data[:, :feature_num - 1], 1)
    # 对数据进行归一化处理
    for _ in range(data.shape[0]):
        data[_, :feature_num] = data[_, :feature_num] / maximums[_]

    # 定义数据集每个数据的序号，根据序号读取数据
    index_list = list(range(len(data)))

    # 定义数据生成器t
    def data_generator():
        random.shuffle(index_list)  # 打乱顺序
        datas_list = []
        labels_list = []
        for i_ in index_list:
            datas = np.array(data[i_][:-1])
            sss = np.array(datas.reshape([3, 8, 8])).astype('float32')
            label = np.array(data[i_][-1:]).astype('int64') + 1

            datas_list.append(sss)
            labels_list.append(label)
            if len(datas_list) == BATCH_SIZE:
                yield np.array(datas_list), np.array(labels_list)
                datas_list = []
                labels_list = []
        if len(datas_list) > 0:
            yield np.array(datas_list), np.array(labels_list)

    return data_generator


class PCCNN_2(paddle.nn.Layer):
    def __init__(self):
        super(PCCNN_2, self).__init__()

        self.conv1 = Conv2D(in_channels=3, out_channels=32, kernel_size=1)
        self.conv2 = Conv2D(in_channels=32, out_channels=48, kernel_size=3)
        self.conv3 = Conv2D(in_channels=48, out_channels=64, kernel_size=3)
        self.conv4 = Conv2D(in_channels=64, out_channels=128, kernel_size=3)
        self.conv5 = Conv2D(in_channels=128, out_channels=160, kernel_size=2)

        self.fc1 = Linear(in_features=160, out_features=64)
        # self.fc2 = Linear(in_features=64, out_features=32)
        self.fc3 = Linear(in_features=64, out_features=3)

    def forward(self, inputs, label=None):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc3(x)
        if label is not None:
            acc = paddle.metric.accuracy(input=x, label=label)
            return x, acc
        else:
            return x


def train(model_):
    print('start train .......')
    model_.train()
    # 调用加载数据的函数，获得训练数据集
    train_loader = load_data()

    opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model_.parameters())
    log_writer = LogWriter(logdir="./log")

    loss_list = []
    iter_ = 0
    iters = []
    acc_list = []
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            # 准备数据
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)

            # 前向计算的过程
            predicts, acc = model_(images, labels)

            # 计算损失，取一个批次样本损失的平均值
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)

            if batch_id % 1000 == 0:
                loss = avg_loss.item(0)
                loss_list.append(loss)
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.item(),
                                                                            acc.item()))
                log_writer.add_scalar(tag='acc', step=iter_, value=acc.item())
                log_writer.add_scalar(tag='loss', step=iter_, value=avg_loss.item())
                iters.append(iter_)
                acc_list.append(acc.item())
                iter_ = iter_ + 1000
            # 后向传播，更新参数的过程
            avg_loss.backward()
            # 最小化loss,更新参数
            opt.step()
            # 清除梯度
            opt.clear_grad()

            # 保存模型参数
        paddle.save(model_.state_dict(), './model.pdparams')
        return iters, acc_list


def evaluation(model_, string=None):
    print('start evaluation .......')

    model_.eval()
    eval_loader = load_data('eval', string)

    acc_set = []
    avg_loss_set = []
    for batch_id, data in enumerate(eval_loader()):
        images, labels = data
        images = paddle.to_tensor(images)
        # print(images)
        labels = paddle.to_tensor(labels)
        predicts, acc = model_(images, labels)
        predicts = model_(images)
        loss = F.cross_entropy(input=predicts, label=labels)
        avg_loss = paddle.mean(loss)
        acc_set.append(float(acc.item()))
        avg_loss_set.append(float(avg_loss.item()))
        # 计算多个batch的平均损失和准确率
        acc_val_mean = np.array(acc_set).mean()
        avg_loss_val_mean = np.array(avg_loss_set).mean()
        if acc_val_mean > 0.9:
            print('loss={},************ acc={}'.format(avg_loss_val_mean, acc_val_mean))
            return 'evaluation: loss={},**************** acc={}'.format(avg_loss_val_mean, acc_val_mean)
        else:
            print('loss={}, acc={}'.format(avg_loss_val_mean, acc_val_mean))
            return 'evaluation: loss={}, acc={}'.format(avg_loss_val_mean, acc_val_mean)


def get_files_in_folder(folder_path, file_extension):
    files = [folder_path + '/' + file for file in os.listdir(folder_path) if file.endswith(file_extension)]
    return files


def merge_npy_files(file_paths, output_path):
    # 初始化一个空列表，用于存储每个文件的数据
    data_list = []

    # 逐个加载每个文件，并将数据添加到列表中
    for file_path in file_paths:
        data = np.load(file_path)
        data_list.append(data)

    # 使用 numpy.concatenate() 函数将数据进行合并
    merged_data = np.concatenate(data_list)

    # 保存合并后的数据为新的 .npy 文件
    np.save(output_path, merged_data)


def train_data(train_list=None):
    if train_list is None:
        folder_path = "./jsp_result_slice"
        file_extension = ".npy"
        filelist = get_files_in_folder(folder_path, file_extension)
        # print(filelist)
    else:
        filelist = train_list
    train_size = len(filelist)
    output_path = './train.npy'
    merge_npy_files(filelist, output_path)
    return train_size


def deeplearning():
    model = PCCNN_2()

    string = 'train data' + '\nBATBATCH_SIZE:' + str(BATCH_SIZE) + '\nEPOCH_NUM:' + str(EPOCH_NUM) + '\nLEARN_RATE:0.001' + '\n'
    train_iter, train_acc = train(model)
    for i, j in zip(train_iter, train_acc):
        string += 'loss={}, acc={}'.format(i, j) + '\n'

    result_string = './learning_data' + '.txt'
    file = open(result_string, 'w')
    file.write(string)
    file.close()


def model_test(evaluation_list: list):
    """
    对给定的测试问题集进行测试，并记录每个问题对应的准确率以及整体的准确率
    :param evaluation_list:
    :return:
    """
    model = PCCNN_2()
    params_file_path = './model.pdparams'
    param_dict = paddle.load(params_file_path)
    model.load_dict(param_dict)
    string = 'evaluation data' + evaluation(model) + '\n'
    for i in evaluation_list:
        string += i + '\n' + evaluation(model, i + '.npy') + '\n'

    result_string = './learning_data' + '.txt'
    with open(result_string, 'a', encoding='utf-8') as file:
        file.write(string)


if __name__ == '__main__':
    model = PCCNN_2()
    # params_file_path = './model.pdparams'
    # param_dict = paddle.load(params_file_path)
    # model.load_dict(param_dict)
    train(model)
    evaluation(model, 'eval')

    new_list = []
    for i in range(200):
        temp = 'test8_' + str(i+2600)
        new_list.append(temp)
    model_test(new_list)

