#! /usr/bin/env python

import time
from collections import namedtuple
from typing import List

import torch
import vllm
from loguru import logger as log

# Customize these.
MODEL_NAME = "teknium/OpenHermes-7B"
PROMPT_FILE = "prompts.txt"
BATCH_SIZE = 100
PROMPT_FORMAT = "### Instruction:\n{prompt}\nResponse:\n"
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_TOKENS = 2000
PRESENCE_PENALTY = 0.0
FREQUENCY_PENALTY = 0.0
SEED = 42
# End customize.

GPU_NAME = (
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
)

InferenceResult = namedtuple("InferenceResult", ["prompt", "response"])


def load_prompts(path: str) -> List[str]:
    with open(path, "r") as f:
        return f.read().splitlines()


def format_prompts(prompts: List[str]) -> List[str]:
    return [PROMPT_FORMAT.format(prompt=prompt) for prompt in prompts]


def batch_prompts(prompts: List[str], batch_size: int) -> List[List[str]]:
    return [
        prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)
    ]


# You can modify this to write to a file, make your HTML, whatever.
def display_results(results: List[InferenceResult]):
    for result in results:
        print("-" * 10)
        print(result.prompt)
        print(result.response)
    print("-" * 10)


if __name__ == "__main__":
    log.info(f"GPU: {GPU_NAME}")
    log.info(f"model: {MODEL_NAME}")
    log.info(f"batch_size: {BATCH_SIZE}")
    log.info(f"temperature: {TEMPERATURE}")
    log.info(f"top_p: {TOP_P}")
    log.info(f"max_tokens: {MAX_TOKENS}")
    log.info(f"presence_penalty: {PRESENCE_PENALTY}")
    log.info(f"frequency_penalty: {FREQUENCY_PENALTY}")
    log.info(f"seed: {SEED}")

    start_time = time.time()

    # prep prompts and batches
    prompts = load_prompts(PROMPT_FILE)
    formatted_prompts = format_prompts(prompts)
    batches = batch_prompts(formatted_prompts, BATCH_SIZE)
    log.info(f"prompt count: {len(prompts)}")
    log.info(f"batch count: {len(batches)}")

    # load model and inference
    llm = vllm.LLM(
        MODEL_NAME,
        seed=SEED,
    )
    params = vllm.SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        presence_penalty=PRESENCE_PENALTY,
        frequency_penalty=FREQUENCY_PENALTY,
    )

    results: List[InferenceResult] = []

    for batch in batches:
        generated = llm.generate(batch, params)
        for output in generated:
            results.append(
                InferenceResult(
                    prompt=output.prompt, response=output.outputs[0].text
                )
            )

    elapsed = time.time() - start_time
    print(f"time to generate: {elapsed:.2f}s")

    # display results
    display_results(results)
