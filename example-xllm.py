import os

from transformers import AutoTokenizer

from xllm.llm import LLM
from xllm.sampling_params import SamplingParams



def single_prompt_inference(llm: LLM, tokenizer: AutoTokenizer, sampling_params: SamplingParams):
    prompt_strs = [
        "introduce yourself",
        "tell me a joke",
        "give me a list of 10 things to do today",
        "what is the capatical of China?"
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_str}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        for prompt_str in prompt_strs
    ]
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(output["text"])


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=100)
    single_prompt_inference(llm, tokenizer, sampling_params)


if __name__ == "__main__":
    main()
