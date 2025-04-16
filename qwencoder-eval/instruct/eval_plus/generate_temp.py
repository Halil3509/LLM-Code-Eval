import argparse
import os
from os import PathLike
import sys
eval_plus_path = os.path.dirname(os.path.abspath(__file__)) + "/evalplus/"
sys.path = [eval_plus_path] + sys.path
from model_temp import DecoderBase, make_model
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)


MODEL_MAPPING = {
    #  Can be either repo's name or /path/to/model
    "codeqwen": {
        "base": "Qwen/CodeQwen1.5-7B",
        "chat": "Qwen/CodeQwen1.5-7B-Chat",
        "chat-awq": "Qwen/CodeQwen1.5-7B-Chat-AWQ",
    },
    "qwen2": {
        "chat": "Qwen/CodeQwen1.5-7B-Chat",
    },
}


def construct_contract_prompt(prompt: str, contract_type: str, contract: str) -> str:
    """
    Constructs a modified prompt by embedding a contract into it based on the specified contract type.

    Args:
        prompt (str): The original prompt string.
        contract_type (str): The type of contract embedding. 
            - "none": Returns the original prompt without modification.
            - "docstring": Embeds the contract within the docstring of the prompt.
            - "code": Appends the contract at the beginning of the function.
        contract (str): The contract string to embed, with comments stripped.

    Returns:
        str: The modified prompt with the contract embedded based on the specified type.

    Raises:
        AssertionError: If the prompt does not contain a valid docstring delimiter when 
                        `contract_type` is "docstring".
    """
    if contract_type == "none":
        return prompt
    elif contract_type == "docstring":
        # embed within the docstring
        sep = ""
        if '"""' in prompt:
            sep = '"""'
        elif "'''" in prompt:
            sep = "'''"
        assert sep != ""
        l = prompt.split(sep)
        contract = "\n".join([x.split("#")[0] for x in contract.splitlines()])
        l[1] = l[1] + contract + "\n" + " " * (len(contract) - len(contract.lstrip()) - 1)
        return sep.join(l)
    elif contract_type == "code":
        # at the beginning of the function
        contract = "\n".join([x.split("#")[0] for x in contract.splitlines()])
        return prompt + contract


def code_generate(args, workdir: PathLike, model: DecoderBase, id_range=None):
    """
    Generates code samples for a given dataset using a specified model and saves the outputs to a specified directory.

    Args:
        args: An object containing various arguments and configurations for code generation, including:
            - dataset (str): The name of the dataset to use (e.g., "humaneval", "mbpp").
            - contract_type (str): The type of contract to use in the prompt (e.g., "none").
            - n_samples (int): The number of samples to generate per task.
            - greedy (bool): Whether to use greedy decoding for generation.
            - resume (bool): Whether to resume from previously generated samples.
        workdir (PathLike): The directory where generated code samples will be saved.
        model (DecoderBase): The model used for code generation. Must implement a `codegen` method.
        id_range (tuple, optional): A tuple specifying the range of task IDs to process (low, high). 
            Tasks outside this range will be skipped. Defaults to None.

    Raises:
        AssertionError: If the model does not produce any outputs during code generation.
        UnicodeEncodeError: If there is an encoding issue while saving generated code to a file.

    Behavior:
        - Loads the specified dataset and iterates through its tasks.
        - Skips tasks outside the specified `id_range` if provided.
        - Skips tasks with empty contracts if `args.contract_type` is not "none".
        - Resumes from previously generated samples if `args.resume` is True.
        - Generates code samples using the model and saves them as `.py` files in the specified `workdir`.
        - Handles special cases such as removing markdown code block delimiters (```).

    Note:
        - The function uses a progress bar to display the generation progress.
        - The generated code is saved in subdirectories named after the task IDs, with filenames corresponding to the sample index.
    """
    with Progress(
        TextColumn(f"{args.dataset} •" + "[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ) as p:
        if args.dataset == "humaneval":
            from evalplus.data import get_human_eval_plus
            dataset = get_human_eval_plus()
        elif args.dataset == "mbpp":
            from evalplus.data import get_mbpp_plus
            dataset = get_mbpp_plus()

        for task_id, task in p.track(dataset.items()):
            if id_range is not None:
                id_num = int(task_id.split("/")[1])
                low, high = id_range
                if id_num < low or id_num >= high:
                    p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                    continue

            p_name = task_id.replace("/", "_")
            if args.contract_type != "none" and task["contract"] == "":
                continue
            os.makedirs(os.path.join(workdir, p_name), exist_ok=True)
            log = f"Codegen: {p_name} @ {model}"
            n_existing = 0
            if args.resume:
                # count existing .py files
                n_existing = len([f for f in os.listdir(os.path.join(workdir, p_name)) if f.endswith(".py")])
                if n_existing > 0:
                    log += f" (resuming from {n_existing})"

            nsamples = args.n_samples - n_existing
            p.console.print(log)

            sidx = args.n_samples - nsamples
            while sidx < args.n_samples:
                model.dataset = args.dataset
                outputs = model.codegen(
                    construct_contract_prompt(task["prompt"], args.contract_type, task["contract"]).strip(),
                    do_sample=not args.greedy,
                    num_samples=args.n_samples - sidx,
                )
                assert outputs, "No outputs from model!"
                for impl in outputs:
                    if "```" in impl:
                        impl = impl.split("```")[0]
                        print("``` exist in generation. Please check the generation results.")

                    try:
                        with open(
                            os.path.join(workdir, p_name, f"{sidx}.py"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            if model.direct_completion:
                                f.write(task["prompt"] + impl)
                            else:
                                f.write(impl)
                    except UnicodeEncodeError:
                        continue
                    sidx += 1


def main():
    """
    Main function to parse command-line arguments, configure the model, and generate code.

    This script is designed to load a specified model, configure its parameters, and generate
    code samples based on the provided dataset and settings. It supports various configurations
    such as model type, size, batch size, temperature, and decoding strategy.

    Command-line Arguments:
        --model_type (str, required): The type of model to use. Must be one of the keys in MODEL_MAPPING.
        --model_path (str, optional): Path to the model checkpoint. Defaults to the path in MODEL_MAPPING.
        --model_size (str, required): The size of the model to use.
        --bs (int, optional): Batch size for inference. Defaults to 1.
        --temperature (float, optional): Sampling temperature for generation. Defaults to 0.0.
        --dataset (str, required): The dataset to use for evaluation. Choices are "humaneval" or "mbpp".
        --root (str, required): Root directory for saving outputs.
        --n_samples (int, optional): Number of samples to generate. Defaults to 1.
        --resume (flag): If set, resumes from a previous run.
        --output (str, optional): Path to save the output.
        --tensor-parallel-size (int, optional): Number of tensor parallel processes. Defaults to 1.
        --contract-type (str, optional): Type of contract to use. Choices are "none", "code", or "docstring". Defaults to "none".
        --greedy (flag): If set, enables greedy decoding and overrides batch size, temperature, and sample count.
        --id-range (list of int, optional): A list of two integers specifying the range of IDs to process.

    Workflow:
        1. Parses and validates command-line arguments.
        2. Loads the specified model and configures its parameters.
        3. Creates necessary directories for saving outputs.
        4. Writes the parsed arguments to a file for reference.
        5. Calls the `code_generate` function to generate code samples.

    Raises:
        AssertionError: If `model_size` is not valid for the given `model_type`.
        AssertionError: If `id_range` is not a list of two integers or is not in increasing order.

    Outputs:
        - Generated code samples saved in the specified working directory.
        - A file containing the parsed arguments for reference.

    Example:
        python generate.py --model_type gpt --model_size large --dataset humaneval --root ./outputs
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--dataset", required=True, type=str, choices=["humaneval", "mbpp"])
    parser.add_argument("--n_samples", default=1, type=int)
    parser.add_argument("--root", type=str, required=True) # root folder_name
    parser.add_argument("--output", type=str)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", default="11434", type=str)
    parser.add_argument(
        "--contract-type",
        default="none",
        type=str,
        choices=["none", "code", "docstring"],
    )
    parser.add_argument("--greedy", action="store_true")
    # id_range is list
    parser.add_argument("--id-range", default=None, nargs="+", type=int)
    args = parser.parse_args()

    print(args)



    if args.greedy and (args.temperature != 0 or args.n_samples != 1):
        args.temperature = 0
        args.n_samples = 1
        print("Greedy decoding ON (--greedy): setting , n_samples=1, temperature=0")

    if args.id_range is not None:
        assert len(args.id_range) == 2, "id_range must be a list of length 2"
        assert args.id_range[0] < args.id_range[1], "id_range must be increasing"
        args.id_range = tuple(args.id_range)

    # Make project dir
    os.makedirs(args.root, exist_ok=True)
    # Make dataset dir
    os.makedirs(os.path.join(args.root, args.dataset), exist_ok=True)
    # Make dir for codes generated by each model


    model = make_model(
        model_type=args.model_type,
        temperature=args.temperature,
        dataset=args.dataset,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
    )

    workdir = os.path.join(
        args.root,
        args.dataset,
        args.model_type
        + f"_{args.model_size}"
        + f"_temp_{args.temperature}"
        + ("" if args.contract_type == "none" else f"-contract-{args.contract_type}"),
    )
    os.makedirs(workdir, exist_ok=True)
    print(f"Working dir: {workdir}")

    with open(os.path.join(workdir, "args.txt"), "w") as f:
        f.write(str(args))

    print(f"Model cls: {model.__class__}")
    print(f"EOS tokens: {model.eos}")
    code_generate(args, workdir=workdir, model=model, id_range=args.id_range)


if __name__ == "__main__":
    main()
