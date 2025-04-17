# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
import json
import random
import fnmatch

import torch
import datasets
import numpy as np
import transformers
from vllm import LLM
from transformers import HfArgumentParser, AutoTokenizer

from generator_temp import Generator
from generation_arguments import EvalArguments

from eval_plus.model_temp import OllamaDecoder # It is Ollama Model

from tasks import ALL_TASKS


class MultiChoice:

    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_args():
    parser = HfArgumentParser(EvalArguments)

    parser.add_argument(
        "--model_name",
        default="codeparrot/codeparrot-small",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Model revision to use",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        choices=MultiChoice(ALL_TASKS),
        help=f"Evaluation tasks from {ALL_TASKS}",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset before evaluation (useful for distributed inference)",
    )
    parser.add_argument(
        "--max_length_generation",
        type=int,
        default=1024,
        help="Maximum length of generated sequence (prompt+generation)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Ending index of samples in the benchmark to solve",
    )
    parser.add_argument(
        "--save_generations",
        action="store_true",
        help="Whether to save code generations",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only solve the first limit samples in the benchmark (useful with randomize dataset)",
    )
    parser.add_argument(
        "--save_generations_path",
        type=str,
        default="generations.json",
        help="Path for saving the code generations",
    )
    parser.add_argument(
        "--save_references",
        action="store_true",
        help="Whether to save reference solutions/tests",
    )
    parser.add_argument(
        "--save_references_path",
        type=str,
        default="references.json",
        help="Path for saving the reference solutions/tests",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="11434",
        help="Path for saving the reference solutions/tests",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Path for saving the reference solutions/tests",
    )

    args = parser.parse_args()

    args.tasks = pattern_match(args.tasks.split(","), ALL_TASKS)
    assert (len(args.tasks) == 1), f"Only one task is supported at the moment, you gave {args.tasks}"
    args.task_name = args.tasks[0]

    assert args.instruction_tokens is None, "Instruction tokens are not supported yet"
    return args


def pattern_match(patterns, source_list):
    """Returns a list containing all values of the source_list that
    match at least one of the patterns"""
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()

    model = OllamaDecoder(
        model_name=args.model_name,
        temperature=args.temperature,
        dataset=args.dataset,
        host=args.host,
        port=args.port,
        max_new_tokens=args.max_length_generation,
    )

    generator = Generator(model, args)
    generations, generations_raw, references = generator.generate(args.task_name)

    with open(args.save_generations_path, "w") as fp:
        json.dump(generations, fp, indent=4)
        print(f"generations were saved at {args.save_generations_path}")

    path = args.save_generations_path
    path = path.split(".json")[0] + "_raw" + ".json"
    with open(path, "w") as fp:
        json.dump(generations_raw, fp, indent=4)
        print(f"generations were saved at {path}")
    if args.save_references:
        with open(args.save_generations_path, "w") as fp:
            json.dump(references, fp, indent=4)
            print("references were saved")


if __name__ == "__main__":
    main()
