""" This file contains the code for calling all LLM APIs. """

import os
from functools import partial
import tiktoken
from .schema import TooLongPromptError, LLMError

enc = tiktoken.get_encoding("cl100k_base")

import litellm
import torch

    
def log_to_file(log_file: str, prompt: str, completion: str, model: str, max_tokens_to_sample: int) -> None:
    """ Log the prompt and completion to a file.
    
    Args:
        log_file (str): Path to the log file.
        prompt (str): The input prompt to be logged.
        completion (str): The generated completion text to be logged.
        model (str): The model used for generating the completion.
        max_tokens_to_sample (int): Maximum number of tokens that were sampled.
    """
    with open(log_file, "a") as f:
        # Write the prompt to the log file
        f.write("\n=================== Prompt =====================\n")
        f.write(f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}")
        
        # Calculate and log the number of prompt tokens
        num_prompt_tokens = len(enc.encode(f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}"))
        
        # Write the model response to the log file
        f.write(f"\n==================={model} response ({max_tokens_to_sample})=====================\n")
        f.write(completion)
        
        # Calculate and log the number of sampled tokens
        num_sample_tokens = len(enc.encode(completion))
        f.write("\n=================== Tokens =====================\n")
        f.write(f"Number of prompt tokens: {num_prompt_tokens}\n")
        f.write(f"Number of sampled tokens: {num_sample_tokens}\n")
        f.write("\n\n")

def complete_text_openai(prompt: str, stop_sequences: list = None, model: str = "gpt-4o-mini", max_tokens_to_sample: int = 500, temperature: float = 0.2, log_file: str = None, **kwargs) -> str:
    """ Call the LiteLLM API to complete a prompt using the OpenAI API.
    
    Args:
        prompt (str): The input prompt to be completed.
        stop_sequences (list, optional): List of stop sequences for the completion. Defaults to [].
        model (str, optional): The model to be used for completion. Defaults to "gpt-4o-mini".
        max_tokens_to_sample (int, optional): Maximum number of tokens to generate. Defaults to 500.
        temperature (float, optional): Sampling temperature. Defaults to 0.2.
        log_file (str, optional): Path to the log file. Defaults to None.
        **kwargs: Additional keyword arguments for the API call.
    
    Returns:
        str: The generated completion text.
    """

    # TODO(mononito): This can likely support other modeling APIs too
    
    # Call the LiteLLM API to generate the completion
    response = litellm.completion(
        model=f"openai/{model}",
        messages=[{"content": prompt, "role": "user"}],
        max_completion_tokens=max_tokens_to_sample,
        temperature=temperature,
        stop=stop_sequences if stop_sequences else None,  # API doesn't like empty list
        **kwargs
    )
    
    # Extract the completion text from the response
    completion = response.choices[0].message.content
    
    # Log the prompt and completion if a log file is specified
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    
    return completion


def complete_text(prompt: str, log_file: str, model: str, **kwargs) -> str:
    """ Complete text using the specified model with appropriate API.
    
    Args:
        prompt (str): The input prompt to be completed.
        log_file (str): Path to the log file.
        model (str): The model to be used for completion.
        **kwargs: Additional keyword arguments for the API call.
    
    Returns:
        str: The generated completion text.
    """
    
    if model.startswith("gpt"):
        # Use OpenAI API
        completion = complete_text_openai(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
    # elif model.startswith("claude"):
    #     # Use Anthropic API
    #     completion = complete_text_claude(prompt, stop_sequences=[anthropic.HUMAN_PROMPT, "Observation:"], log_file=log_file, model=model, **kwargs)
    # elif model.startswith("gemini"):
    #     # Use Gemini API
    #     completion = complete_text_gemini(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model}")
    
    return completion

# Specify fast models for summarization etc
FAST_MODEL: str = "gpt-3.5-turbo"

def complete_text_fast(prompt: str, **kwargs) -> str:
    """ Complete text using a fast model for summarization or similar tasks.
    
    Args:
        prompt (str): The input prompt to be completed.
        **kwargs: Additional keyword arguments for the API call.
    
    Returns:
        str: The generated completion text.
    """
    return complete_text(prompt=prompt, model=FAST_MODEL, temperature=0.01, **kwargs)

# def complete_text_openai(prompt, stop_sequences=[], model="gpt-3.5-turbo", max_tokens_to_sample=500, temperature=0.2, log_file=None, **kwargs):
#     """ Call the OpenAI API to complete a prompt."""
#     raw_request = {
#         "model": model,
#         "temperature": temperature,
#         "max_tokens": max_tokens_to_sample,
#         "stop": stop_sequences or None,  # API doesn't like empty list
#         **kwargs
#     }
   
#         messages = [{"role": "user", "content": prompt}]
#         response = openai.ChatCompletion.create(messages=messages, **raw_request)
#         completion = response["choices"][0]["message"]["content"]
   
#     if log_file is not None:
#         log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
#     return completion

# def complete_text_gemini(prompt, stop_sequences=[], model="gemini-pro", max_tokens_to_sample = 2000, temperature=0.5, log_file=None, **kwargs):
#     """ Call the gemini API to complete a prompt."""
#     # Load the model
#     model = GenerativeModel("gemini-pro")
#     # Query the model
#     parameters = {
#             "temperature": temperature,
#             "max_output_tokens": max_tokens_to_sample,
#             "stop_sequences": stop_sequences,
#             **kwargs
#         }
#     safety_settings = {
#             harm_category: SafetySetting.HarmBlockThreshold(SafetySetting.HarmBlockThreshold.BLOCK_NONE)
#             for harm_category in iter(HarmCategory)
#         }
#     safety_settings = {
#         }
#     response = model.generate_content( [prompt], generation_config=parameters, safety_settings=safety_settings)
#     completion = response.text
#     if log_file is not None:
#         log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
#     return completion

# def complete_text_claude(prompt, stop_sequences=[anthropic.HUMAN_PROMPT], model="claude-v1", max_tokens_to_sample = 2000, temperature=0.5, log_file=None, messages=None, **kwargs):
#     """ Call the Claude API to complete a prompt."""

#     ai_prompt = anthropic.AI_PROMPT
#     if "ai_prompt" in kwargs is not None:
#         ai_prompt = kwargs["ai_prompt"]

#     try:
#         if model == "claude-3-opus-20240229":
#             while True:
#                 try:
#                     message = anthropic_client.messages.create(
#                         messages=[
#                             {
#                                 "role": "user",
#                                 "content": prompt,
#                             }
#                         ] if messages is None else messages,
#                         model=model,
#                         stop_sequences=stop_sequences,
#                         temperature=temperature,
#                         max_tokens=max_tokens_to_sample,
#                         **kwargs
#                     )
#                 except anthropic.InternalServerError as e:
#                     pass
#                 try:
#                     completion = message.content[0].text
#                     break
#                 except:
#                     print("end_turn???")
#                     pass
#         else:
#             rsp = anthropic_client.completions.create(
#                 prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {ai_prompt}",
#                 stop_sequences=stop_sequences,
#                 model=model,
#                 temperature=temperature,
#                 max_tokens_to_sample=max_tokens_to_sample,
#                 **kwargs
#             )
#             completion = rsp.completion
        
#     except anthropic.APIStatusError as e:
#         print(e)
#         raise TooLongPromptError()
#     except Exception as e:
#         raise LLMError(e)

    
#     if log_file is not None:
#         log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
#     return completion