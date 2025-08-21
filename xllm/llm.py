from typing import List, Union

from xllm.engine.llm_engine import LLMEngine
from xllm.sampling_params import SamplingParams


class LLM(LLMEngine):

    def generate(
        self,
        prompts: Union[List[str], List[List[int]]],
        sampling_params: Union[SamplingParams, List[SamplingParams]],
    ):
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        for prompt, sampling_params in zip(prompts, sampling_params):
            self.add_request(prompt, sampling_params)

        outputs = []
        while self.has_unfinished_requests():
            step_outputs = self.step()
            outputs.extend(step_outputs)

        return outputs
