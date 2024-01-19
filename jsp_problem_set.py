import os
import random
import numpy as np


def problem_set(problem_size_j, problem_size_m, problem_num):
    """

    :param problem_size_j:
    :param problem_size_m:
    :param problem_num:
    :return:
    """
    merge_filedir = './jsp_data_%i-%i' % (problem_size_m, problem_size_j)
    filenames = os.listdir(merge_filedir)
    new_set = []
    for num in range(problem_num):
        choice = [list(range(problem_size_m))]
        for _ in range(problem_size_j-1):
            part_start = random.sample(range(0, 3), 2)
            temp = np.delete(np.arange(problem_size_m), part_start)
            ix_j = np.random.sample(temp.shape[0]).argsort()
            sequence = part_start + list(temp[ix_j])
            choice.append(sequence)

        output = '%i  %i\n' % (problem_size_j, problem_size_m)
        for i in range(problem_size_j):
            list_ = []
            while True:
                if len(list_) < problem_size_m:
                    a = random.randint(1, 100)
                    if 0 < a < 100:
                        list_.append(a)
                else:
                    break
            for j, k in zip(choice[i], list_):
                output += '%i   %i    ' % (j, k)
            output += '\n'

        result_string = merge_filedir + '/test%i-%i_%i.txt' % (problem_size_j, problem_size_m, num + len(filenames))
        new_set.append('/test%i-%i_%i.txt' % (problem_size_j, problem_size_m, num + len(filenames)))
        file = open(result_string, 'w')
        file.write(output)
        file.close()
    return new_set


if __name__ == '__main__':
    print("jsp_problem_set")
    problem_set(12, 8, 199)
