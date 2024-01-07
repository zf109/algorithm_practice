from typing import List, Callable, Tuple
import numpy as np

def simple_discount_length_func(size: int, num_resource: int, discount: float = .1) -> float:
    """
    a sumple function, as resource been put to one task, we have dinimishing return, each added resource
    will be discounted.
    """
    discounted_num_resource = sum( max(1 - n * discount, 0) for n in range(num_resource))
    return size / (discounted_num_resource) if discounted_num_resource != 0 else float("inf")


def simple_robust_n_discount_length_func(
    size: int,
    num_resource: int,
    discount: float = .1,
    max_robust_penalty: float = 1.5
) -> float:
    """
    take into account of robustness, it has two parts:
    - robustness penalty: fewer resource, the longer will be estimated duration as it is less robust
    - discount penalty: the more resource added the less efficienty it can be.
    """
    robustness_penalty_func = {1: max_robust_penalty, 2: max_robust_penalty / 2}  # very simple mapping
    robustness_penalty = robustness_penalty_func[num_resource] if num_resource in (1, 2) else 1
    discounted_num_resource = sum( max(1 - n * discount, 0) for n in range(num_resource))
    return size / (discounted_num_resource) * robustness_penalty if discounted_num_resource != 0 else float("inf")


def chop_list(a_list: List, chunk: int):
    return [a_list[n:n + chunk] for n in a_list[::chunk]]


class TaskFunction:
    """
    different task can have different task function.
    """
    def __init__(self, size: int, length_func: Callable[[int, int], int]):
        """
        size: size of the task
        length_func: function that describe how task duraction reduce as number of resource put in
        """
        self.size = size
        self.length_func = length_func

    def length(self, num_resource: int) -> int:
        """
        take a list of 0, 1 which indicates number of resource
        """
        return self.length_func(self.size, num_resource)


class ResourceAllocScenario:
    def __init__(self, tasks: List[TaskFunction], n_resources: int) -> None:
        self.tasks = tasks
        self.n_resources = n_resources

    @property    
    def n_tasks(self):
        return len(self.tasks)


    def _validate(self, resource_to_task_array: np.ndarray):
        n_given_task, n_given_resource = resource_to_task_array.shape

        if n_given_task != self.n_tasks or n_given_resource != self.n_resources:
            raise ValueError(f"expect alloc array to have n_row, n_col={(self.n_tasks, self.n_resources)}, "
                        f"got ({n_given_task}, {n_given_resource})")

    def calc_all_durations(self, resource_to_task_array: np.ndarray) -> List[float]:
        """
        expect to have resource_to_task_array with shape n_tasks x n_resources
        returns a list of each tasks length taken, given n_resources allocates to it
        """
        n_resource_per_task = np.sum(resource_to_task_array, axis=1).astype(int)
        if (len_n_resource_per_task := len(n_resource_per_task)) != (n_tasks := self.n_tasks):
            raise ValueError(f"{len_n_resource_per_task=} not the same as {n_tasks=}")
        return [t.length(n_resource) for t, n_resource in zip(self.tasks, n_resource_per_task)]

    def duration_for_all_resources(self, resource_to_task_array: np.ndarray) -> List[float]:
        """
        expect to have resource_to_task_array with shape n_tasks x n_resources
        returns a list of each resource's total working duration
        """
        n_resources = resource_to_task_array.shape[1]
        all_durations = self.calc_all_durations(resource_to_task_array=resource_to_task_array)
        resource_duration_array = np.tile(np.c_[all_durations], n_resources)
        return list(np.sum(resource_to_task_array * resource_duration_array, axis=0))


    def total_task_set_duration(self, resource_to_task_array: np.ndarray) -> float:
        """
        resource_alloc_vector: a list of 0 and 1 with length n_resources x n_tasks
                    an indicator list  where the kth (k=ixj) element indicates
                     resource i is assigned to task j
        returns the total time take to finish a task, that is, the duration of the resource who took longest.
        """
        self._validate(resource_to_task_array)
        duration_for_all_resources = self.duration_for_all_resources(resource_to_task_array=resource_to_task_array)
        return max(duration_for_all_resources)


def get_max_duration(resource_alloc_vector: List[int], scenario: ResourceAllocScenario) -> float:
    scenario.n_tasks, scenario.n_resources
    resource_to_task_array = np.reshape(resource_alloc_vector, (scenario.n_tasks, scenario.n_resources))
    return scenario.total_task_set_duration(resource_to_task_array)
