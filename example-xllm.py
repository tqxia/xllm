import os

from transformers import AutoTokenizer

from xllm.llm import LLM
from xllm.sampling_params import SamplingParams



def single_prompt_inference(llm: LLM, tokenizer: AutoTokenizer, sampling_params: SamplingParams):
    prompt_str = "introduce yourself"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_str}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    output = llm.generate(prompt, sampling_params)
    print(output["text"])


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=64)
    single_prompt_inference(llm, tokenizer, sampling_params)


if __name__ == "__main__":
    main()
