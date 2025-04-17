# Copyright (c) Meta Platforms, Inc. and affiliates.

from pprint import pprint

from . import input_prediction, output_prediction

TASK_REGISTRY = {
    "input_prediction": input_prediction.InputPrediction,
    "output_prediction": output_prediction.OutputPrediction,
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name, cot = False):
    try:
        return TASK_REGISTRY[task_name](cot = cot)
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
