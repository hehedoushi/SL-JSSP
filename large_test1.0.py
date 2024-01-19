from jsp_deeplearn import deeplearning
from jsp_result_manage import get_result_slice
from jsp_deeplearn import model_test
from jsp_deeplearn import train_data
from datetime import datetime
import shutil
import os


if __name__ == '__main__':
    print('PyCharm')
    for _ in range(1):
        # 获取当前时间
        current_time = datetime.now()
        # 将时间格式化为字符串
        current_time_str = current_time.strftime("%m%d%H%M")

        new_list = []
        for i in range(1200 + 400*_):
            temp = './jsp_result_slice/test8_' + str(i + 0) + '.npy'
            new_list.append(temp)
        # 统计已有的切片，建立训练数据集
        num = train_data(new_list)

        # new_data = get_result_slice()
        deeplearning()
        new_list = []
        for i in range(10):
            temp = 'test8_' + str(i+2700)
            new_list.append(temp)
        model_test(new_list)