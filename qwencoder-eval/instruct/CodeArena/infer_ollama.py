import argparse
import json
from tqdm import tqdm
from utils import utils
from eval_plus.model_temp import OllamaDecoder, make_model


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Code generation parameters")
    parser.add_argument("--model_name", type=str, default="", help="Path to the model")
    parser.add_argument("--model_type", type=str, default="ollama", help="Type of the model")
    parser.add_argument("--host", type=str, default="localhost", help="Model host address")
    parser.add_argument("--port", type=str, default="11434", help="Model host port")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--input_path", type=str, default="./CodeArena_v1.jsonl", help="Path to input JSONL file")
    parser.add_argument("--output_path", type=str, default="./results/yi-lightning/results.jsonl", help="Path to output JSONL file")
    parser.add_argument("--model_max_len", type=int, default=8192 * 2, help="Maximum model input length")
    return parser.parse_args()


def prepare_inputs(test_data):
    """Prepare inputs for the model."""
    codellama_template = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction} [/INST]"
    prepared_data = []

    for obj in tqdm(test_data, desc="Preparing inputs"):
        obj["input"] = codellama_template.format_map({
            "system_prompt": "",
            "instruction": obj["messages"][-1]["content"]
        })
        prepared_data.append(obj)

    return prepared_data


def generate_responses(model, prompts):
    """Generate responses using the model."""
    outputs = []
    for prompt in tqdm(prompts, desc="Generating responses"):
        try:
            output = model.codegen(prompt)
            outputs.append(output[0])
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            raise ValueError(f"Generation error: {str(e)}")
    return outputs


def main():
    args = parse_args()
    print(f"Arguments: {args}")

    # Load test data
    test_data = utils.read_jsonl_file(args.input_path)
    if not test_data:
        raise ValueError("Input file is empty or invalid.")

    # Prepare inputs
    objs = prepare_inputs(test_data)

    # Initialize model
    model = make_model(
        model_name=args.model_name,
        model_type=args.model_type,
        host=args.host,
        port=args.port
    )

    # Generate responses
    prompts = [obj["input"] for obj in objs]
    outputs = generate_responses(model, prompts)

    # Combine results
    for obj, output in zip(objs, outputs):
        obj["model"] = args.model_name
        obj["response"] = output

    # Write results to output file
    utils.write_jsonl_file(objs, args.output_path)
    print(f"Code generation completed. Results saved to {args.output_path}")


if __name__ == "__main__":
    main()