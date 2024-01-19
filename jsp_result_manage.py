import chardet
import copy
import collections
import numpy as np
from ORtools_Optimal_Schedule import MinimalJobshopSat
from jsp_problem_set import problem_set


def load_text(file_name):
    try:
        with open(file_name, "rb") as f:
            f_read = f.read()
            f_cha_info = chardet.detect(f_read)
            final_data = f_read.decode(f_cha_info['encoding'])
            return final_data, True
    except FileNotFoundError:
        return str(None), False


#   用于实现字典中通过值找键  value-key
def get_key(dct, value):
    k = [k for k, v in dct.items() if v == value]
    return k


def change_machines_num(array_1_1, array_1_2, array_2):
    """
    :param array_1_2: 工件信息
    :param array_1_1: 工件信息
    :param array_2:环境信息矩阵
    :return:机器顺序调整后的信息矩阵

    """
    # array1 = np.array(array_1)
    array_1_1 = np.array(array_1_1)
    array_1_2 = np.array(array_1_2)
    array2 = np.array(array_2)
    # print(array2)
    columns_dict = {}
    for index, column in enumerate(array2.T):
        a, b, c = 9, 0, 0
        for _, j in enumerate(column):
            if j != 0.0:
                a = _
                break
        for _, j in enumerate(column):
            if j != 0.0:
                b = _
            c = c + j
        columns_dict.setdefault(index, 100 * a + 10 * b + c)
    columns_dict = sorted(columns_dict.items(), key=lambda x: x[1])
    change = [x[0] for x in columns_dict]
    array2 = array2[:, change]
    array_1_1 = array_1_1[:, change]
    array_1_2 = array_1_2[:, change]
    return array_1_1, array_1_2, array2


def average_pooling(matrix, pool_size, strides=(1, 1)):
    height, width = matrix.shape

    pool_height, pool_width = pool_size
    stride_height, stride_width = strides
    output_height = (height - pool_height) // stride_height + 1
    output_width = (width - pool_width) // stride_width + 1

    output = np.zeros((output_height, output_width))

    for i in range(0, output_height):
        for j in range(0, output_width):
            region = matrix[i * stride_height:i * stride_height + pool_height, j * stride_width:j * stride_width + pool_width]
            output[i, j] = np.max(region)

    return output


def reduce_dimension_10(matrix):
    # 直接裁剪
    temp_array = matrix[:8, :8]
    return temp_array


def reduce_dimension_15(matrix):
    np.set_printoptions(linewidth=np.inf)
    # 特定池化
    array = np.zeros((8, 8))
    padded_matrix = np.pad(matrix, ((0, 1), (0, 1)), mode='constant')
    # print(padded_matrix)
    output_matrix = average_pooling(padded_matrix, (4, 4), strides=(4, 4))
    array[-2:, -2:] = output_matrix[-2:, -2:]
    output_matrix = average_pooling(padded_matrix, (2, 4), strides=(2, 4))
    array[4:6, -2:] = output_matrix[2:4, -2:]
    output_matrix = average_pooling(padded_matrix, (4, 2), strides=(4, 2))
    array[-2:, 4:6] = output_matrix[-2:, 2:4]
    output_matrix = average_pooling(padded_matrix, (1, 4), strides=(1, 4))
    array[0:4, -2:] = output_matrix[0:4, -2:]
    output_matrix = average_pooling(padded_matrix, (1, 4), strides=(1, 4))
    array[-2:, 0:4] = output_matrix[-2:, 0:4]
    output_matrix = average_pooling(padded_matrix, (2, 2), strides=(2, 2))
    array[4:6, 4:6] = output_matrix[2:4, 2:4]
    output_matrix = average_pooling(padded_matrix, (1, 2), strides=(1, 2))
    array[0:4, 4:6] = output_matrix[0:4, 2:4]
    output_matrix = average_pooling(padded_matrix, (2, 1), strides=(2, 1))
    array[4:6, 0:4] = output_matrix[2:4, 0:4]
    array[0:4, 0:4] = padded_matrix[0:4, 0:4]
    temp_array = array

    # # 对输入矩阵进行填充---平均池化
    # padded_matrix = np.pad(matrix, ((0, 1), (0, 1)), mode='constant')
    # # 自适应池化，子区域大小为2x2
    # output_matrix = average_pooling(padded_matrix, (2, 2), strides=(2, 2))
    # temp_array = output_matrix

    return temp_array


def getdata_str(line):  # 获得可被训练的数据形式
    """
    line:'0 51 1 36 2 40 3 28 4 66 5 10 6 19 7 11 -2 2 83 1 77 0 20 4 46 6 72 7 24 5 32 3 39 -2 -1 0 2 30 1 61 6 72 7
    92 4 32 3 42 5 69 -2 -1 0 0 85 4 53 3 56 1 28 7 67 6 85 5 55 -2 -1 0 2 18 7 21 4 99 6 41 0 68 3 40 5 59 -2 1 6 2
    67 4 24 5 69 0 19 6 30 3 89 7 34 -2 0 59 2 58 7 62 3 36 1 57 5 80 4 3 6 31 -2 -1 0 0 70 6 71 5 34 1 59 4 52 7 69
    3 70 -2 #3 1 7 1#1'
    :param line: 问题信息以及竞争对象 完成的工序被-1 0代替
    :return:  可以输入网络的数据以及标签  一个行向量
    :type line: string
    """

    job_load_str = list(map(int, line.split('#')[0].split()))
    double_list = list(map(int, line.split('#')[1].split()))
    job_load, temp = [], []
    for _ in iter(job_load_str):
        if _ != -2:
            temp.append(_)
        else:
            job_load.append(temp)
            temp = []
    # print(job_load)
    jobs_data, job, double_object, max_time = [], [], [], 0
    for temp_1 in job_load:
        for j, k in zip(temp_1[0::2], temp_1[1::2]):  # enumerate枚举  zip聚合多个对象
            job.append([j, k])
            if max_time < k:
                max_time = k
        jobs_data.append(job)
        job = []
    # print(jobs_data, max_time)

    for job_list in jobs_data:
        for temp in job_list:
            temp[1] = round(temp[1] / max_time, 4)

    array1_1 = [[0.0 for _ in range(len(jobs_data[0]))] for _ in range(len(jobs_data[0]))]
    array1_2 = [[0.0 for _ in range(len(jobs_data[0]))] for _ in range(len(jobs_data[0]))]
    for j, k in zip(double_list[0::2], double_list[1::2]):
        # double_object.append([j, k])
        object_temp = []
        for s, temp in enumerate(jobs_data[j]):
            if temp[0] != -1:
                if s >= k:
                    object_temp.append(temp)
                else:
                    object_temp.append([0, 0])
        double_object.append(object_temp)
    # print(double_object)
    for _, item in enumerate(double_object[0]):
        array1_1[_][item[0]] = item[1]
    for _, item in enumerate(double_object[1]):
        array1_2[_][item[0]] = item[1]

    # print(array_1)
    array2 = [[0.0 for _ in range(len(jobs_data[0]))] for _ in range(len(jobs_data[0]))]
    for job_list in jobs_data:
        size = len(job_list)
        for temp in range(size):
            if job_list[0][0] < 0:
                del job_list[0]
        # print(job_list)
        for _, job in enumerate(job_list):
            array2[_][job[0]] = job[1] + array2[_][job[0]]
    array1_1, array1_2, array2 = change_machines_num(array1_1, array1_2, array2)

    rows, cols = array1_1.shape

    if cols == 15:
        # my_array = np.array(array1_2)
        # np.set_printoptions(linewidth=np.inf)
        # print(my_array)

        array1_1 = reduce_dimension_15(array1_1)
        array1_2 = reduce_dimension_15(array1_2)
        array2 = reduce_dimension_15(array2)

        # my_array = np.array(array2)
        # np.set_printoptions(linewidth=np.inf)
        # print(my_array)
        # quit()

    elif cols == 10:
        array1_1 = reduce_dimension_10(array1_1)
        array1_2 = reduce_dimension_10(array1_2)
        array2 = reduce_dimension_10(array2)

    rows, cols = array1_1.shape
    if cols < 8:
        # 填充零向量
        expanded_matrix = np.zeros((8, 8), dtype=array1_1.dtype)
        expanded_matrix[:rows, :cols] = array1_1
        array1_1 = copy.deepcopy(expanded_matrix)
        expanded_matrix[:rows, :cols] = array1_2
        array1_2 = copy.deepcopy(expanded_matrix)
        expanded_matrix[:rows, :cols] = array2
        array2 = copy.deepcopy(expanded_matrix)

    output_array = []
    for list_ in array1_1:
        for _ in list_:
            output_array.append(round(_, 6))
    for list_ in array1_2:
        for _ in list_:
            output_array.append(round(_, 6))
    for list_ in array2:
        for _ in list_:
            output_array.append(round(_, 6))

    if len(line.split('#')[-1]) != 0:
        label = line.split('#')[-1]
        output_array.append(int(label))
    # print(output)
    # return output_str
    return output_array


def slice_manage(jobs_data, jobs_data_slice, timeline_choices, next_choices):
    """
    # 将得到的时间切片 处理为竞争之间的数据关系
    # 需要明确各个工件的加工进度 进而判断竞争关系
    # [[(2, 1), (0, 3), (1, 6), (3, 7), (5, 3), (4, 6)],
    # [(1, 8), (2, 5), (4, 10), (5, 10), (0, 10), (3, 4)],
    # [(2, 5), (3, 4), (5, 8), (0, 9), (1, 1), (4, 7)],
    # [(1, 5), (0, 5), (2, 5), (3, 3), (4, 8), (5, 9)],
    # [(2, 9), (1, 3), (4, 5), (5, 4), (0, 3), (3, 1)],
    # [(1, 3), (3, 3), (5, 9), (0, 10), (4, 4), (2, 1)]]
    #
    # jobs_load :[[(), (2, 1)], [(), (1, 8)], [(), (2, 5)], [(), (1, 5)], [(), (2, 9)], [(), (1, 3)]]
    :param next_choices: 当前未做出决策的机器之后的选择
    :param jobs_data:问题信息
    :param jobs_data_slice:决策点的剩余问题信息
    :param timeline_choices:**** [(1, 0), (0, 0)]当前决策点做出的选择
    :return:竞争对以及相应的关系（包含有主主竞争 主从竞争 这里的主主一半是赢一半是输  主从都是输的  主从就不一定了
    """
    # print(jobs_data_slice, timeline_choices)
    # print(next_choices)
    jobs_load = []
    for iter1, iter2 in zip(iter(jobs_data), iter(jobs_data_slice)):
        jobs_load.append([(), ()])
        for opera1, opera2 in zip(iter(iter1), iter(iter2)):
            if opera2 != (-1, 0):
                if opera1 != opera2:
                    jobs_load[-1] = [opera2, ()]
                    break
                else:
                    jobs_load[-1] = [(), opera2]
                    break
    # 判断竞争 分两种竞争--主对主 以及主对可能存在的竞争
    # 关于平等竞争的情况 需要在获得的方案中进行辨识 进而将竞争条进行对比修正竞争结果
    compete_objects = []
    for choice in timeline_choices:  # 主主判断
        data = jobs_data[choice[0]][choice[1]]
        # print(choice, jobs_load, data)
        for i_, load in enumerate(jobs_load):
            if i_ == choice[0] and data != load[1]:
                print('false in job_load')
                return 0
            elif i_ != choice[0] and load[1] != ():
                if data[0] == load[1][0]:
                    for k, temp in enumerate(jobs_data[i_]):
                        if temp == load[1]:
                            complected = [choice, (i_, k), 1]
                            compete_objects.append(complected)
                            complected = [(i_, k), choice, -1]
                            compete_objects.append(complected)
        for i_, jobs_data_list in enumerate(jobs_data_slice):
            for data_slice in jobs_data_list:
                if data_slice != (-1, 0):
                    if i_ != choice[0] and data_slice[0] != data[0] and data_slice[1] < data[1]:
                        sum_time, j = 0, 0
                        for add_data in jobs_data_list:
                            if add_data[0] != data[0]:
                                sum_time += add_data[1]
                                j += 1
                            elif add_data[0] == data[0]:
                                if sum_time < data[1]:
                                    double = [choice, (i_, j), 1]
                                    compete_objects.append(double)
                                break
                    break
    # 从优于主的判断
    for next_ in next_choices:
        data = jobs_data[next_[0]][next_[1]]
        sum_time = 0
        for i_ in range(next_[1]):
            sum_time += jobs_data_slice[next_[0]][i_][1]
        for index, i_ in enumerate(jobs_load):
            if i_[1] != ():
                if i_[1][0] == data[0]:
                    if i_[1][1] > sum_time:
                        k = jobs_data_slice[index].index(i_[1])
                        complected = [(index, k), next_, -1]
                        compete_objects.append(complected)
    # print(compete_objects)
    return compete_objects


def result_manage(string):
    """
    :param string: 问题名称 test8_0类似
    :return: 在文件slice中写入数据文件
    """
    file_name = './jsp_data_8/' + string + '.txt'  # './jsp_data/FT06.txt'
    data, check = load_text(file_name)  # data读取的是文本中全部的数据
    if data is not None:
        a = list(map(int, data.split()))  # 创建一个列表，使用 split() 函数进行分割 # map() 函数根据提供的函数对指定序列做映射
        n, machines_count = a[0], a[1]
        jobs_data = []
        job = []
        for i_, (j, k) in enumerate(zip(a[2::2], a[3::2])):  # enumerate枚举带索引！！  zip聚合多个对象
            job.append((j, k))
            if (i_ + 1) % machines_count == 0:
                jobs_data.append(job)
                job = []
        # print(jobs_data)

        file_name = './jsp_result/' + string + '.txt'  # './jsp_data/FT06.txt'
        string_file, check = load_text(file_name)  # data读取的是文本中全部的数据
        write_list, slice_dict = [], collections.defaultdict(set)
        data_list = string_file.split('*')
        # print(len(data_list))
        for data in data_list:
            data_ = list(map(int, data.split()))
            machine_test = []
            machine_time = []
            machine_timeline = []
            temp_start_time = []
            temp_end_time = []
            machine = []
            for i_, (j, k) in enumerate(zip(data_[0::2], data_[1::2])):  # enumerate枚举  zip聚合多个对象
                machine.append((j, k))
                temp_start_time.append(j)
                temp_end_time.append(k)
                if (i_ + 1) % machines_count == 0:
                    if ((i_ + 1) / machines_count) % 2 == 1:
                        machine_test.append(machine)
                    else:
                        machine_time.append(machine)
                        machine_timeline = machine_timeline + temp_start_time + temp_end_time
                    machine = []
                    temp_start_time = []
            machine_timeline = list(set(machine_timeline))  # set只能建立无重复的集合 不能去排序 FT06之前是巧合
            machine_timeline.sort()
            # print(machine_timeline)
            # 接着沿着时间 获取当前每个工件的进度 对原问题的信息进行操作 并记录当前的选择
            # 获得的timeline为标准 标记加工进度来得到每个工件的当前进度 进而获得每个工件的当前矩阵 然后获得总的矩阵
            map_dict = {}  # 方案中工序与加工时段的对应
            for value1, value2 in zip(iter(machine_test), iter(machine_time)):
                for j, k in zip(iter(value1), iter(value2)):
                    map_dict[j] = k
            jobs_data_slice = copy.deepcopy(jobs_data)

            for timeline in machine_timeline:
                timeline_choices = []
                next_choices = []
                # print(timeline, type(timeline))
                for x, machine_time_list in enumerate(machine_time):
                    for a in range(len(machine_time_list)):
                        this_time = machine_time_list[a]
                        if timeline > this_time[0]:
                            index_list = get_key(map_dict, this_time)
                            for index in index_list:
                                if timeline > this_time[1]:
                                    jobs_data_slice[index[0]][index[1]] = (-1, 0)
                                    if a < len(machine_time_list) - 1:
                                        next_time = machine_time_list[a + 1]
                                        if next_time[0] > timeline:
                                            next_choices.append(machine_test[x][a + 1])
                                elif timeline == this_time[1]:
                                    jobs_data_slice[index[0]][index[1]] = (-1, 0)
                                    if a < len(machine_time_list) - 1:
                                        next_time = machine_time_list[a + 1]
                                        if next_time[0] > timeline:
                                            next_choices.append(machine_test[x][a + 1])
                                else:
                                    new = this_time[1] - timeline
                                    jobs_data_slice[index[0]][index[1]] = (jobs_data_slice[index[0]][index[1]][0], new)
                        elif timeline == this_time[0]:
                            index_list = get_key(map_dict, this_time)
                            timeline_choices += index_list
                # print(jobs_data_slice, '****', timeline_choices, '*******', next_choices)
                compete_objects = slice_manage(jobs_data, jobs_data_slice, set(timeline_choices), next_choices)

                output = ''
                for jobs_data_list in jobs_data_slice:
                    for write_data in jobs_data_list:
                        output += '%i %i' % (write_data[0], write_data[1])
                        output += ' '
                    output += '-2 '
                output += '#'
                for item in compete_objects:
                    temp = (item[0][0], item[0][1], item[1][0], item[1][1])
                    value_ = output + '%i' % item[2]
                    slice_dict[temp].add(value_)
        write_array = []
        for item_list in slice_dict.items():
            index = (item_list[0][2], item_list[0][3], item_list[0][0], item_list[0][1])
            if index in slice_dict:
                for value in item_list[1]:
                    if value in slice_dict[index]:
                        information = value.split('#')[0]
                        temp = information + '#%i %i %i %i#' % \
                               (item_list[0][0], item_list[0][1], item_list[0][2], item_list[0][3]) + '0'
                        write_array.append(getdata_str(temp))
                    else:
                        information = value.split('#')[0]
                        label = value.split('#')[1]
                        temp = information + '# %i %i %i %i#' % \
                               (item_list[0][0], item_list[0][1], item_list[0][2], item_list[0][3]) + label
                        write_array.append(getdata_str(temp))
                        temp = information + '# %i %i %i %i#' % \
                               (item_list[0][2], item_list[0][3], item_list[0][0], item_list[0][1]) + str(-int(label))
                        write_array.append(getdata_str(temp))
            else:
                for value in iter(slice_dict[item_list[0]]):
                    information = value.split('#')[0]
                    label = value.split('#')[1]
                    temp = information + '#%i %i %i %i#' % \
                           (item_list[0][0], item_list[0][1], item_list[0][2], item_list[0][3]) + label
                    write_array.append(getdata_str(temp))
                    temp = information + '# %i %i %i %i#' % \
                           (item_list[0][2], item_list[0][3], item_list[0][0], item_list[0][1]) + str(-int(label))
                    write_array.append(getdata_str(temp))

        new_list = [tuple(x) for x in write_array]
        unique_list = list(set(new_list))
        write_array = [list(x) for x in unique_list]
        result_string = './jsp_result_slice/' + string + '.npy'
        np.save(result_string, write_array)


def get_result_slice():
    """
    创建固定数量的新问题作为测试集
    :return:
    """
    num = 100

    data_list = []
    temp = './jsp_data_8/'
    new_problem = problem_set(8, 8, num)
    for problem in new_problem:
        MinimalJobshopSat(temp, problem)
        result_manage(problem)
        data = np.load('./jsp_result_slice/' + problem + '.npy')
        data_list.append(data)
        # 使用 numpy.concatenate() 函数将数据进行合并
    merged_data = np.concatenate(data_list)
    output_path = './test.npy'
    # 保存合并后的数据为新的 .npy 文件
    np.save(output_path, merged_data)
    return new_problem


if __name__ == '__main__':
    for i in range(20):
        str_ = 'test8_%i' % (i + 0)
        result_manage(str_)

    # data_list = []
    # for i in range(2000):
    #     data = np.load('./jsp_result_slice/test8_' + str(i + 0) + '.npy')
    #     data_list.append(data)
    # merged_data = np.concatenate(data_list)
    # output_path = './train.npy'
    # # 保存合并后的数据为新的 .npy 文件
    # np.save(output_path, merged_data)

