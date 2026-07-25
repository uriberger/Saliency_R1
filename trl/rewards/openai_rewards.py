# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import ast
import openai
from retrying import retry
import os

# LLM-as-judge via the NVIDIA inference API. Key is supplied at run time through
# NVIDIA_API_KEY (falls back to OPENAI_API_KEY). Endpoint/model overridable via env.
client = openai.OpenAI(
    api_key=os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY", "xxx"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://inference-api.nvidia.com"),
)
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "azure/openai/gpt-4o-mini")

@retry(wait_exponential_multiplier=200, wait_exponential_max=2000, retry_on_exception=lambda e: isinstance(e, openai.RateLimitError) or isinstance(e, openai.APIConnectionError))
def openai_reward(completions, solution, problem, **kwargs):
    ground_truth_list = solution
    problem_list = problem
    contents = [completion[0]["content"] for completion in completions]
    #prediction_list = [i.split("</think>")[-1] for i in contents]
    prediction_list = [re.search(r"</think>\s*([^<]*(?:(?!<think>|</think>).)*?)\s*$", i, re.DOTALL | re.MULTILINE) for i in contents]
    #prediction_list = [re.search(r"<answer>\s*(.*?)\s*</answer>", i, re.DOTALL | re.MULTILINE) for i in contents]
    prediction_list = [i.group(1) if i else None for i in prediction_list]
    # Only retry transient failures. A 400 (e.g. content-policy violation on a
    # generated completion) is deterministic, so retrying is pointless and, if it
    # escapes, it crashes the whole distributed run.
    @retry(stop_max_attempt_number=5, wait_exponential_multiplier=200,
           retry_on_exception=lambda e: isinstance(e, (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)))
    def query_gpt4o(question, ground_truth, prediction):
        # Compute the correctness score
        chat_completion = client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            max_tokens=512,
            messages=[
                {
                    "role": "system",
                    "content": "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                               "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                               "------"
                               "##INSTRUCTIONS: "
                               "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                               "- Consider synonyms or paraphrases as valid matches.\n"
                               "- Evaluate the correctness of the prediction compared to the answer.",
                },
                {
                    "role": "user",
                    "content": f"I will give you an image and the following text as inputs:\n\n"
                               f"1. **Question Related to the Image**: {question}\n"
                               f"2. **Ground Truth Answer**: {ground_truth}\n"
                               f"3. **Model Predicted Answer**: {prediction}\n\n"
                               "Your task is to evaluate the model's predicted answer against the ground truth answer, based on the context provided by the image and the question. Consider the following criteria for evaluation:"
                               "- **Relevance**: Does the predicted answer directly address the question posed, considering the information provided in the image?"
                               "- **Accuracy**: Compare the predicted answer to the ground truth answer. Does the prediction accurately reflect the information given in the ground truth answer without introducing factual inaccuracies?"
                               "**Output Format**:"
                               "Score: <a integer score of quality from 1-5>",
                },
            ], timeout=120)

        response_message = chat_completion.choices[0].message.content
        # print(f"Response Message: {response_message}")
        score_match = re.search(r'Score:\s*(\d+)', response_message)
        if score_match:
            score = (int(score_match.group(1)) - 1.0) / 4.0
            return score

    def safe_query(question, ground_truth, prediction):
        if not prediction:
            return 0
        try:
            score = query_gpt4o(question, ground_truth, prediction)
        except Exception as e:
            # Judge failed on this sample (content filter, malformed request, etc.).
            # Score it 0 rather than tearing down the entire training run.
            print(f"[openai_reward] judge failed, scoring 0: {type(e).__name__}: {e}")
            return 0
        # query_gpt4o returns None when no "Score: N" is present in the response.
        return score if score is not None else 0

    return [safe_query(question, ground_truth, prediction) for question, ground_truth, prediction in zip(problem_list, ground_truth_list, prediction_list)]
