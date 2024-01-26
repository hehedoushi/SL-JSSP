from __future__ import print_function

import json
import chardet
import datetime
import collections
import copy
from ortools.sat.python import cp_model

dt = datetime.datetime
time_delta = datetime.timedelta


# 求解最佳方案的OR-TOOL工具
def load_text(file_name):
    try:
        with open(file_name, "rb") as f:
            f_read = f.read()
            f_cha_info = chardet.detect(f_read)
            final_data = f_read.decode(f_cha_info['encoding'])
            return final_data, True
    except FileNotFoundError:
        return str(None), False


def MinimalJobshopSat(path, string):
    file_name = path + string + '.txt'  # './jsp_data/FT06.txt'
    # file_name = string
    data, check = load_text(file_name)  # data读取的是文本中全部的数据
    result = 0

    if data is not None:
        solve_list, while_count = [], 0

        a = list(map(int, data.split()))  # 创建一个列表，使用 split() 函数进行分割 # map() 函数根据提供的函数对指定序列做映射
        n, machines_count = a[0], a[1]
        all_machines = range(machines_count)
        jobs_data = []
        job = []

        for _, (j, k) in enumerate(zip(a[2::2], a[3::2])):  # enumerate枚举  zip聚合多个对象
            job.append((j, k))
            if (_ + 1) % machines_count == 0:
                jobs_data.append(job)
                job = []

        """Minimal jobshop problem."""
        # Create the model.
        model = cp_model.CpModel()

        # Computes horizon dynamically as the sum of all durations.
        horizon = sum(task[1] for job in jobs_data for task in job)
        # Named tuple to store information about created variables.
        task_type = collections.namedtuple('task_type', 'start end interval')
        # Named tuple to manipulate solution information.
        assigned_task_type = collections.namedtuple('assigned_task_type',
                                                    'start job index duration')

        # Creates job intervals and add to the corresponding machine lists.
        all_tasks = {}
        machine_to_intervals = collections.defaultdict(list)

        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                duration = task[1]
                suffix = '_%i_%i' % (job_id, task_id)
                start_var = model.NewIntVar(0, horizon, 'start' + suffix)
                end_var = model.NewIntVar(0, horizon, 'end' + suffix)
                interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                    'interval' + suffix)
                all_tasks[job_id, task_id] = task_type(
                    start=start_var, end=end_var, interval=interval_var)
                machine_to_intervals[machine].append(interval_var)

        # Create and add disjunctive constraints.
        for machine in all_machines:
            model.AddNoOverlap(machine_to_intervals[machine])

        # Precedences inside a job.
        for job_id, job in enumerate(jobs_data):
            for task_id in range(len(job) - 1):
                model.Add(all_tasks[job_id, task_id +
                                    1].start >= all_tasks[job_id, task_id].end)

        # Makespan objective.
        obj_var = model.NewIntVar(0, horizon, 'makespan')
        model.AddMaxEquality(obj_var, [all_tasks[job_id, len(job) - 1].end for job_id, job in enumerate(jobs_data)])
        model.Minimize(obj_var)
        while True:
            # Solve model.
            solver = cp_model.CpSolver()
            status = solver.Solve(model)
            while_count += 1

            if status == cp_model.OPTIMAL:
                # # Create one list of assigned tasks per machine.
                assigned_jobs = collections.defaultdict(list)
                for job_id, job in enumerate(jobs_data):
                    for task_id, task in enumerate(job):
                        machine = task[0]
                        assigned_jobs[machine].append(
                            assigned_task_type(
                                start=solver.Value(all_tasks[job_id, task_id].start),
                                job=job_id,
                                index=task_id,
                                duration=task[1]))
                if len(solve_list) == 0:
                    solve_list.append(assigned_jobs)
                    while_count = 0
                else:
                    for _, ss in enumerate(solve_list):
                        label = False
                        a_list, b_list, c, d = [], [], 0, 0
                        for j in range(len(ss)):
                            ss[j].sort()
                            assigned_jobs[j].sort()
                            for a, b in zip(ss[j], assigned_jobs[j]):
                                a_list.append(a.job)
                                b_list.append(b.job)
                        if a_list == b_list:
                            label = True
                        # print(a_list,'\n',b_list,label)
                        if label:
                            break
                        elif _ == len(solve_list) - 1:
                            solve_list.append(assigned_jobs)
                            # print('solve_list:  %i' % len(solve_list))

                # Create per machine output lines.
                print('User time: %.2fs' % solver.UserTime())
                # print('Wall time: %.2fs' % solver.WallTime())
                print('Optimal Schedule Length: %i' % solver.ObjectiveValue())
                result = solver.ObjectiveValue()
            else:
                print('No solution found.')
            if while_count == 20:
                break
        # print(len(solve_list))
        print(string, '    ', result)
        # print(type(result))
        arr = {string: result}
        with open('jsp_makespan.json', 'r') as file:
            data = json.load(file)
        data.update(arr)
        with open('jsp_makespan.json', 'w') as file:
            json.dump(data, file)
        # 调整时间 尽可能往前
        output = ''
        for assigned_jobs in solve_list:
            for _ in range(machines_count):
                new_assigned_jobs = collections.defaultdict(list)
                operation_dict = {}
                for machine in all_machines:
                    assigned_jobs[machine].sort()
                    last_overtime = 0
                    for assigned_task in assigned_jobs[machine]:
                        start = assigned_task.start
                        duration = assigned_task.duration
                        operation_dict.setdefault((assigned_task.job, assigned_task.index),
                                                  (last_overtime, 0, start, start + duration))
                        last_overtime = start + duration
                for opera in iter(operation_dict):
                    if opera[1] != 0:
                        temp_ = operation_dict[opera]
                        change = operation_dict[(opera[0], opera[1] - 1)][-1]
                        operation_dict[opera] = (temp_[0], change, temp_[2], temp_[3])
                # print(operation_dict)
                for machine in all_machines:
                    for assigned_task in assigned_jobs[machine]:
                        opera_ = (assigned_task.job, assigned_task.index)
                        if operation_dict[opera_][1] < operation_dict[opera_][0]:
                            time_ = operation_dict[opera_][0]
                        else:
                            time_ = operation_dict[opera_][1]
                        if assigned_task.start > time_:
                            new_assigned_jobs[machine].append(
                                assigned_task_type(
                                    start=time_,
                                    job=opera_[0],
                                    index=opera_[1],
                                    duration=assigned_task.duration))
                        else:
                            new_assigned_jobs[machine].append(assigned_task)
                assigned_jobs = copy.deepcopy(new_assigned_jobs)
            # print(new_assigned_jobs)
            for machine in all_machines:
                # Sort by starting time.
                assigned_jobs[machine].sort()
                sol_line_tasks = ' '
                sol_line = ' '
                for assigned_task in assigned_jobs[machine]:
                    name = '%i %i' % (assigned_task.job,
                                      assigned_task.index)
                    # Add spaces to output to align columns.
                    sol_line_tasks += '%-15s' % name

                    start = assigned_task.start
                    duration = assigned_task.duration
                    sol_tmp = '%i %i' % (start, start + duration)
                    # Add spaces to output to align columns.
                    sol_line += '%-15s' % sol_tmp

                sol_line += '\n'
                sol_line_tasks += '\n'
                output += sol_line_tasks
                output += sol_line
            output += '*\n'
        result_string = './jsp_result/' + string + '.txt'
        file = open(result_string, 'w')
        file.write(output)
        file.close()


if __name__ == '__main__':
    for i in range(200):
        str_ = 'test14-8_%i' % (i + 134)
        temp = './jsp_data_8-12/'
        MinimalJobshopSat(temp, str_)
