import numpy as np

from ..args import args
from ..entities.action import Action


def pick_first(obs, key=None, reverse=False, n_workers='max'):
    '''sorts the frontier stages by `key`, if provided,
    then selects the first stage in the sorted frontier 
    for which there is at least one available, compatible
    worker.
    '''
    frontier_stages = obs.get_frontier_stages()
    if key is not None:
        frontier_stages.sort(key=key, reverse=reverse)

    avail_workers = obs.find_available_workers()
    
    first_stage = None
    for stage in frontier_stages:
        if first_stage is not None:
            break
        for worker in avail_workers:
            if worker.compatible_with(stage):
                first_stage = stage
                break

    if first_stage is None:
        return Action()

    max_workers = int(first_stage.n_remaining_tasks)
    if n_workers == 'max':
        n_workers = max_workers
    elif n_workers == 'one':
        n_workers = 1
    elif n_workers == 'random':
        n_workers = np.random.randint(1, max_workers+1)

    action = Action(
        job_id=first_stage.job_id,
        stage_id=first_stage.id_,
        n_workers=n_workers
    )
    return action


def fcfs(obs):
    '''selects a frontier stage whose job arrived first'''
    return pick_first(obs)


def frugal_fcfs(obs):
    return pick_first(obs, greedy=False)


def shortest_task_first(obs):
    '''selects a frontier stage whose task duration is shortest'''
    def key(stage):
        durations = stage.task_duration_per_worker_type
        durations = durations[durations<np.inf]
        return durations.mean()
    return pick_first(obs, key)


def longest_task_first(obs):
    '''selects a frontier stage whose task duration is longest'''
    def key(stage):
        durations = stage.task_duration_per_worker_type
        durations = durations[durations<np.inf]
        return durations.mean()
    return pick_first(obs, key, reverse=True)