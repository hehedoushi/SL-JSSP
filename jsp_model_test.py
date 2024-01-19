import json
import paddle
import numpy as np
import chardet
import copy
import collections
from jsp_deeplearn import PCCNN_2
from jsp_result_manage import getdata_str


def nestedList(check):
    returnValue = 0
    for _ in range(len(check)):
        if isinstance(check[_], list):
            returnValue += nestedList(check[_])
        else:
            returnValue += 1
    return returnValue


def load_text(file_name):
    try:
        with open(file_name, "rb") as f:
            f_read = f.read()
            f_cha_info = chardet.detect(f_read)
            final_data = f_read.decode(f_cha_info['encoding'])
            return final_data, True
    except FileNotFoundError:
        return str(None), False


def slice_manage(jobs_data, jobs_data_slice):  # 将得到的时间切片
    jobs_load, machine_occupy = [], []
    for iter1, iter2 in zip(iter(jobs_data), iter(jobs_data_slice)):
        jobs_load.append([(), ()])
        for opera1, opera2 in zip(iter(iter1), iter(iter2)):
            if opera2 != (-1, 0):
                if opera1 != opera2:
                    jobs_load[-1] = [opera2, ()]
                    machine_occupy.append(opera2[0])
                    set(machine_occupy)
                    break
                else:
                    jobs_load[-1] = [(), opera2]
                    break
    choice_list = []
    for choice_id, choice in enumerate(jobs_load):
        if len(choice[0]) == 0 and len(choice[1]) != 0:
            if choice[1][0] not in machine_occupy:
                choice_list.append((choice_id, jobs_data[choice_id].index(choice[1])))
    compete_objects = []
    for choice in choice_list:  # 主主判断
        data = jobs_data[choice[0]][choice[1]]
        for i, load in enumerate(jobs_load):
            if i == choice[0] and data != load[1]:
                print('false in job_load')  #
                return 0
            elif i != choice[0] and load[1] != ():
                if data[0] == load[1][0]:
                    for k, temp in enumerate(jobs_data[i]):
                        if temp == load[1]:
                            compeobject = [choice, (i, k)]
                            compete_objects.append(compeobject)
        # 主与可能竞争
        for i, jobs_data_list in enumerate(jobs_data_slice):
            for data_slice in jobs_data_list:
                if data_slice != (-1, 0):
                    if i != choice[0] and data_slice[0] != data[0] and data_slice[1] < data[1]:
                        sum_time, j = 0, 0
                        for add_data in jobs_data_list:
                            if add_data[0] != data[0]:
                                sum_time += add_data[1]
                                j += 1
                            elif add_data[0] == data[0]:
                                if sum_time < data[1]:
                                    double = [choice, (i, j)]
                                    compete_objects.append(double)
                                break
                    break
    return choice_list, compete_objects


def softmax_max(x):
    # 将输入向量取指数
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    # 对指数化后的向量进行归一化计算
    softmax_x = exp_x / sum_exp_x
    return max(softmax_x)


def model_solve(path, string='test8_0', way='SPT'):
    # 准备模型
    model = PCCNN_2()
    params_file_path = './model.pdparams'
    # 加载模型参数
    param_dict = paddle.load(params_file_path)
    model.load_dict(param_dict)
    model.eval()

    file_name = path + string + '.txt'  # './jsp_data/FT06.txt'
    data, check = load_text(file_name)  # data读取的是文本中全部的数据
    if data is None:
        quit()
    a = list(map(int, data.split()))  # 创建一个列表，使用 split() 函数进行分割 # map() 函数根据提供的函数对指定序列做映射
    jobs_count, machines_count = a[0], a[1]
    jobs_data, job = [], []
    for _, (j, k) in enumerate(zip(a[2::2], a[3::2])):  # enumerate枚举  zip聚合多个对象
        job.append((j, k))
        if (_ + 1) % machines_count == 0:
            jobs_data.append(job)
            job = []
    result_machine, result_time = [[] for _ in range(machines_count)], [[] for _ in range(machines_count)]
    machines_time_note = [0 for _ in range(machines_count)]
    while nestedList(result_machine) != nestedList(jobs_data):
        # 确定节点 获得切片
        for time_i, time_list in enumerate(result_time):
            if len(time_list) != 0 and time_list[-1][1] > machines_time_note[time_i]:
                machines_time_note[time_i] = time_list[-1][1]
        time_note = 0
        if sum(machines_time_note) != 0:
            time_note = max(machines_time_note)
            for _ in machines_time_note:
                if _ != min(machines_time_note) and _ < time_note:
                    time_note = _
            for i_id, _ in enumerate(machines_time_note):
                if _ == min(machines_time_note):
                    machines_time_note[i_id] = time_note
        jobs_data_slice = copy.deepcopy(jobs_data)
        for i_i, _ in enumerate(result_time):
            for j_j, j in enumerate(_):
                if j[1] <= time_note:
                    jobs_data_slice[result_machine[i_i][j_j][0]][result_machine[i_i][j_j][1]] = (-1, 0)
                else:
                    new = j[1] - time_note
                    jobs_data_slice[result_machine[i_i][j_j][0]][result_machine[i_i][j_j][1]] \
                        = (jobs_data_slice[result_machine[i_i][j_j][0]][result_machine[i_i][j_j][1]][0], new)
        choice_list_jobs, compete_objects = slice_manage(jobs_data, jobs_data_slice)
        choice_slice_dict, compete_slice_dict = {}, collections.defaultdict(list)
        for _ in choice_list_jobs:
            # choice_slice_dict[jobs_data[i[0]][i[1]][1]] = i
            choice_slice_dict.setdefault(_, jobs_data[_[0]][_[1]][1])
        choice_slice_dict = sorted(choice_slice_dict.items(), key=lambda x: x[1], reverse=True)  # 会形成列表 排序True为降序
        for _ in compete_objects:
            compete_slice_dict[_[0]].append(_)
        # print('\n time: ', time_note, '\n choice_slice_dict:', choice_slice_dict, '\n compete_objects:', compete_slice_dict)
        note_choices = []
        if way == 'SPT':
            machine_choice_dict = {}
            for choice in choice_list_jobs:
                machine = jobs_data[choice[0]][choice[1]][0]
                if machine not in machine_choice_dict:
                    machine_choice_dict[machine] = choice
                else:
                    # if jobs_data[machine_choice_dict[machine][0]][machine_choice_dict[machine][1]][1] < \
                    #         jobs_data[choice[0]][choice[1]][1]:
                    #     machine_choice_dict[machine] = choice
                    if jobs_data[machine_choice_dict[machine][0]][machine_choice_dict[machine][1]][1] > \
                            jobs_data[choice[0]][choice[1]][1]:
                        machine_choice_dict[machine] = choice
            note_choices = list(machine_choice_dict.values())
        elif way == 'MWKR':
            machine_choice_dict = {}
            for choice in choice_list_jobs:
                machine = jobs_data[choice[0]][choice[1]][0]
                if machine not in machine_choice_dict:
                    machine_choice_dict[machine] = choice
                else:
                    sum1 = sum([x[1] for x in jobs_data[machine_choice_dict[machine][0]][machine_choice_dict[machine][1]:]])
                    sum2 = sum([x[1] for x in jobs_data[choice[0]][choice[1]:]])
                    if sum1 < sum2:
                        machine_choice_dict[machine] = choice
                    # if sum1 > sum2:
                    #     machine_choice_dict[machine] = choice
            note_choices = list(machine_choice_dict.values())
        elif way == 'MOPNR':
            # 最少操作剩余(LOR)最多操作剩余(MOR)
            machine_choice_dict = {}
            for choice in choice_list_jobs:
                machine = jobs_data[choice[0]][choice[1]][0]
                if machine not in machine_choice_dict:
                    machine_choice_dict[machine] = choice
                else:
                    if machine_choice_dict[machine][1] > choice[1]:
                        machine_choice_dict[machine] = choice
                    # if machine_choice_dict[machine][1] < choice[1]:
                    #     machine_choice_dict[machine] = choice
            note_choices = list(machine_choice_dict.values())

        elif way == 'OURS':
            false_list = []
            while len(choice_slice_dict) != 0:
                choice = choice_slice_dict[0]
                choice_slice_dict.remove(choice)
                choice = choice[0]
                output = ''
                for jobs_data_list in jobs_data_slice:
                    for write_data in jobs_data_list:
                        output += '%i %i  ' % (write_data[0], write_data[1])
                    output += '-2 '
                output += '#'
                label = True
                for compete_objects in compete_slice_dict[choice]:
                    if (compete_objects[1][0], compete_objects[1][1]) not in false_list:
                        temp = getdata_str(output + '%i %i %i %i #' %
                                           (compete_objects[0][0], compete_objects[0][1], compete_objects[1][0],
                                            compete_objects[1][1]))

                        temp_temp = getdata_str(output + '%i %i %i %i #' %
                                                (compete_objects[1][0], compete_objects[1][1], compete_objects[0][0],
                                                 compete_objects[0][1]))
                        # data = [np.array([data.reshape(-1, 8)]).astype('float32')]
                        temp = np.array(temp)
                        # 计算训练集的最大值，最小值
                        maximums = np.max(temp)
                        # 对数据进行归一化处理
                        for _ in range(temp.shape[0]):
                            temp[_] = temp[_] / maximums
                        temp = [np.array(temp.reshape([3, 8, 8])).astype('float32')]
                        _data = paddle.to_tensor(temp)
                        result = model(_data)
                        # 取概率最大的标签作为预测输出
                        lab = np.argsort(result.numpy())
                        # print('%i %i %i %i #' % (compete_objects[0][0], compete_objects[0][1], compete_objects[1][0], compete_objects[1][1]), lab[0][-1])

                        temp = np.array(temp_temp)
                        # 计算训练集的最大值，最小值
                        maximums = np.max(temp)
                        # 对数据进行归一化处理
                        for _ in range(temp.shape[0]):
                            temp[_] = temp[_] / maximums
                        temp = [np.array(temp.reshape([3, 8, 8])).astype('float32')]
                        _data = paddle.to_tensor(temp)
                        result_temp = model(_data)
                        # 取概率最大的标签作为预测输出
                        lab_temp = np.argsort(result_temp.numpy())
                        # print('%i %i %i %i #' % (compete_objects[1][0], compete_objects[1][1], compete_objects[0][0], compete_objects[0][1]),lab[0][-1])

                        if lab[0][-1] + lab_temp[0][-1] == 4 or lab[0][-1] + lab_temp[0][-1] == 0:
                            x = softmax_max(result.numpy())
                            y = softmax_max(result_temp.numpy())
                            # print(x, y)
                            if max(x) > max(y):
                                if lab[0][-1] == 0:
                                    label = False
                                    # break
                        elif lab[0][-1] == 0:
                            label = False
                            # break

                del compete_slice_dict[choice]
                if label:  # 这里定义谁最终被选择 因为是有序的判断 所以多个可选的情况会优先选大的
                    note_choices.append(choice)
                    # print('True:  ', note_choices)
                    temp_ = copy.deepcopy(choice_slice_dict)
                    for _ in iter(temp_):
                        if jobs_data[_[0][0]][_[0][1]][0] == jobs_data[choice[0]][choice[1]][0]:
                            choice_slice_dict.remove(_)
                else:
                    false_list.append(choice)
                    # print('false_list', false_list)
                # print(choice_slice_dict)
        # print('note_choices:   ', note_choices)
        for _ in note_choices:
            machine = jobs_data[_[0]][_[1]][0]
            result_machine[machine].append(_)
            result_time[machine].append([time_note, time_note + jobs_data[_[0]][_[1]][1]])
    time = 0
    for _ in result_time:
        if len(_) != 0 and _[-1][1] > time:
            time = _[-1][1]
    # print(string, 'Completion time  ', time)

    file_path = "./jsp_makespan.json"  # JSON 文件路径
    # 读取 JSON 文件
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    if string in json_data:
        best_result = json_data[string]
        gap = time / best_result - 1
        print(string, 'Completion time  ', time, 'best  ', best_result, '  gap', gap)
        return gap
    else:
        print(string, 'Completion time  ', time, 'no know best')

    # output = ''
    # for i_, j_ in zip(result_machine, result_time):
    #     sol_line_tasks = ''
    #     sol_line = ''
    #     for i, j in zip(i_, j_):
    #         name = '%i %i' % (i[0], i[1])
    #         # Add spaces to output to align columns.
    #         sol_line_tasks += '%-15s' % name
    #
    #         sol_tmp = '%i %i' % (j[0], j[1])
    #         # Add spaces to output to align columns.
    #         sol_line += '%-15s' % sol_tmp
    #     sol_line += '\n'
    #     sol_line_tasks += '\n'
    #     output += sol_line_tasks
    #     output += sol_line
    # print(output)


if __name__ == '__main__':
    problem = ['./tai_problem/', 'ta1515', 'OURS']
    for num in range(10):
        temp = problem[1] + str(num + 0)
        model_solve(problem[0], temp, problem[2]) * 100

