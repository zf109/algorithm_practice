
import pytest
import numpy as np
from alloc_scenario import TaskFunction, ResourceAllocScenario, simple_discount_length_func, chop_list

def test_chop_list():
    a_list = list(range(10))
    choped_list = chop_list(a_list=a_list, chunk=2)
    assert choped_list == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]


def test_simple_discount_length_func():
    assert simple_discount_length_func(3, 3, 0) == 1
    assert simple_discount_length_func(3, 3, 0.1) == 1.111111111111111
    assert simple_discount_length_func(3, 3, 0.2) == 1.25
    assert simple_discount_length_func(3, 10, 0.2) == 1.0  # resource saturated
    assert simple_discount_length_func(3, 100, 0.2) == 1.0


def test_taskfunction_length():
    task = TaskFunction(size=3, length_func=simple_discount_length_func)
    assert task.length(num_resource=1) == 3
    assert task.length(num_resource=2) == 1.5789473684210527
    assert task.length(num_resource=3) == 1.111111111111111
    assert task.length(num_resource=10) == 0.5454545454545454


def test_resource_alloc_scenario__validate():
    tasks = [
        TaskFunction(size=2, length_func=simple_discount_length_func),
        TaskFunction(size=2, length_func=simple_discount_length_func),
        TaskFunction(size=2, length_func=simple_discount_length_func),
    ]
    scenario = ResourceAllocScenario(n_resources=3, tasks=tasks)
    scenario._validate(resource_to_task_array=np.array([[0, 1, 1,], [0, 0, 1], [1, 0 ,1]]))

    with pytest.raises(ValueError) as e:
        scenario._validate(resource_to_task_array=np.array([[0, 1], [1, 0]]))
    assert str(e.value) == "expect alloc array to have n_row, n_col=(3, 3), got (2, 2)"


def test_resource_alloc_scenario_calc_all_durations():
    tasks = [
        TaskFunction(size=2, length_func=simple_discount_length_func),
        TaskFunction(size=2, length_func=simple_discount_length_func),
        TaskFunction(size=2, length_func=simple_discount_length_func),
    ]
    scenario = ResourceAllocScenario(n_resources=3, tasks=tasks)
    resource_to_task_array = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1 ,1]
    ])
    length_per_task = scenario.calc_all_durations(resource_to_task_array)
    # first 2 tasks just got 1 resource so take 2 for total length
    # 3rd one get all resources so according to simple_discount_length_func (with default .1 discount)
    # we have 2 / (1 + 0.9 + 0.8) = 0.7407407407407407
    assert np.allclose(length_per_task, [2.0, 2.0, 0.7407407407407407])


def test_resource_alloc_scenario_calc_all_durations_2():
    tasks = [
        TaskFunction(size=2, length_func=simple_discount_length_func),
        TaskFunction(size=2, length_func=simple_discount_length_func),
        TaskFunction(size=2, length_func=simple_discount_length_func),
    ]
    scenario = ResourceAllocScenario(n_resources=3, tasks=tasks)
    resource_to_task_array = np.array([
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 1 ,1, 1]
    ])

    length_per_task = scenario.calc_all_durations(resource_to_task_array)
    # first task are 2 / (1 + 0.9) = 1.0526315789
    # second one got 2 / (1 + 0.9 + 0.8) = 0.7407407407407407
    # last one got 2 / (1 + 0.9 + 0.8 + 0.7) = 0.588235294117647
    assert np.allclose(length_per_task, [1.0526315789, 0.7407407407407407, 0.588235294117647])


def test_resource_alloc_scenario_calc_all_durations_raise():
    tasks = [
        TaskFunction(size=2, length_func=simple_discount_length_func),
        TaskFunction(size=2, length_func=simple_discount_length_func),
        TaskFunction(size=2, length_func=simple_discount_length_func),
    ]
    scenario = ResourceAllocScenario(n_resources=3, tasks=tasks)
    resource_to_task_array = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [1, 0 ,1],
        [1, 0 ,1]
    ])

    with pytest.raises(ValueError) as e:
        scenario.calc_all_durations(resource_to_task_array)
    assert str(e.value) == "len_n_resource_per_task=4 not the same as n_tasks=3"



def test_resource_alloc_scenario_duration_for_all_resources():
    tasks = [
        TaskFunction(size=2, length_func=simple_discount_length_func),
        TaskFunction(size=2, length_func=simple_discount_length_func),
        TaskFunction(size=2, length_func=simple_discount_length_func),
    ]
    scenario = ResourceAllocScenario(n_resources=3, tasks=tasks)
    resource_to_task_array = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1 ,1]
    ])
    length_per_task = scenario.duration_for_all_resources(resource_to_task_array)
    # the tasks durations are [2.0, 2.0, 0.7407407407407407]
    # first resource only worked for task 3, so 0.7407407407407407
    # second resource  worked for task 1 and 3, so 2.7407407407407407
    # third resource worked for task 2 and 3, so 2.7407407407407407
    assert np.allclose(length_per_task, [0.7407407407407407, 2.7407407407407407, 2.7407407407407407])


def test_resource_alloc_scenario_duration_for_all_resources_2():
    tasks = [
        TaskFunction(size=2, length_func=simple_discount_length_func),
        TaskFunction(size=2, length_func=simple_discount_length_func),
        TaskFunction(size=2, length_func=simple_discount_length_func),
    ]
    scenario = ResourceAllocScenario(n_resources=3, tasks=tasks)
    resource_to_task_array = np.array([
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 1 ,1, 1]
    ])

    length_per_task = scenario.duration_for_all_resources(resource_to_task_array)
    # the tasks durations are [1.0526315789, 0.7407407407407407, 0.588235294117647]
    # first resource only worked for task 3, so 0.588235294117647
    # second resource  worked for all tasks, so 2.381607613758388
    # third resource worked for all tasks, so 2.381607613758388
    # fourth resource worked on task 2 and 3, so 1.3289760348583877
    assert np.allclose(length_per_task, [0.588235294117647, 2.381607613758388, 2.381607613758388, 1.3289760348583877])


def test_resource_alloc_scenario_total_task_set_duration():
    tasks = [
        TaskFunction(size=2, length_func=simple_discount_length_func),
        TaskFunction(size=2, length_func=simple_discount_length_func),
        TaskFunction(size=2, length_func=simple_discount_length_func),
    ]
    scenario = ResourceAllocScenario(n_resources=4, tasks=tasks)
    resource_to_task_array = np.array([
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 1 ,1, 1],
    ])
    total_duration = scenario.total_task_set_duration(resource_to_task_array)
    assert pytest.approx(total_duration, 2.381607613758388)


