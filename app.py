from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os

class InferlessPythonModel:

    def initialize(self):
        model_id = "TheBloke/Llama-2-7B-chat-AWQ"  # Specify the model repository ID
        HF_TOKEN = os.getenv("HF_TOKEN") # Access Hugging Face token from environment variable
        tokenizer_model = "meta-llama/Llama-2-7b-chat-hf"
        # Define sampling parameters for model generation
        # You can set max_tokens to 1024 for complete answer to your question
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.1,
            repetition_penalty=1.18,
            top_k=40,
            max_tokens=512,
        )

        # Initialize the LLM object with the downloaded model directory
        self.llm = LLM(model=model_id, enforce_eager=True, quantization="AWQ")

        # Load the tokenizer associated with the pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, token=HF_TOKEN)

    def infer(self, inputs):
        prompts = inputs["prompt"]  # Extract the prompt from the input

        # Apply the chat template and convert to a list of strings (without tokenization)
        input_text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompts}], tokenize=False)

        # Generate text using the LLM with the specified sampling parameters
        result = self.llm.generate(input_text, self.sampling_params)

        # Extract the generated text from the result object
        result_output = [output.outputs[0].text for output in result]

        # Return a dictionary containing the generated text
        return {"generated_result": result_output[0]}

    def finalize(self):
        self.llm = None
        self.tokenizer = None
