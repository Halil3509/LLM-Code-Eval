import os
from abc import ABC, abstractmethod
from typing import List

# Set the Hugging Face cache directory
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "./hf_home")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import ollama

# Define end-of-sequence tokens
EOS_TOKENS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
    "\n#"
]

class DecoderBase(ABC):
    """
    Abstract base class for decoder models. Provides a common interface for code generation.
    """
    def __init__(
        self,
        model_name: str,
        host: str = "localhost",
        port: str = "11434",
        temperature: float = 0.8,
        max_new_tokens: int = 512,
        dataset: str = None,
    ) -> None:
        self.model_name = model_name
        self.host = host
        self.port = port
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.dataset = dataset
        self.eos = EOS_TOKENS

        print(f"Initializing decoder model: {model_name}...")

    @abstractmethod
    def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 200) -> List[str]:
        """
        Abstract method for generating code. Must be implemented by subclasses.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.model_name})"

    def __str__(self) -> str:
        return self.__repr__()

class OllamaDecoder(DecoderBase):
    """
    Decoder implementation for the Ollama model.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(f"Initializing OllamaDecoder with model: {self.model_name}, host: {self.host}, port: {self.port}")
        
        # Initialize the Ollama client
        self.client = ollama.Client(host=f"http://{self.host}:{self.port}")
        print("Ollama client initialized successfully.")

    def codegen(self, prompt: str, num_samples: int = 1) -> List[str]:
        """
        Generate code based on the given prompt using the Ollama model.
        """
        print(f"Starting code generation with prompt: {prompt[:50]}... (truncated for display)")
        print(f"Number of samples: {num_samples}, Temperature: {self.temperature}, Max tokens: {self.max_new_tokens}")
        
        outputs = []
        for i in range(num_samples):
            try:
                response = self.client.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                    stop=self.eos if isinstance(self.eos, list) else [self.eos]
                )
                # Extract and clean the generated content
                content = response['message']['content'].strip()
                outputs.append(content)
            except Exception as e:
                print(f"Error during generation of sample {i + 1}: {str(e)}")
                raise ValueError(f"Generation error: {str(e)}")
        
        print(f"Code generation completed. Total samples generated: {len(outputs)}")
        return outputs

def make_model(
    model_name: str,
    model_type: str = "ollama",
    host: str = "localhost",
    port: str = "11434",
    temperature: float = 0.8,
    dataset: str = None
):
    """
    Factory function to create a decoder model instance based on the specified type.
    """
    if model_type == "ollama":
        return OllamaDecoder(
            model_name=model_name,
            temperature=temperature,
            dataset=dataset,
            host=host,
            port=port
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types are: 'ollama'.")