from dotenv import load_dotenv
from typing import List, Tuple, Dict, Callable
import openai
import os
import time
import requests
import json

def send_to_local_server_completion(prompt: str, model: str, max_tokens: int = 64, temperature: float = 0.0) -> Tuple[str, float]:
    ''' Sends the given prompt to the model that runs on the local server and returns the response '''
    return _send_to_completion(prompt, model, True, max_tokens, temperature)

def send_to_local_server_chat(messages: list, model: str, temperature: float = 0.0,  max_tokens: int = 64) -> Tuple[str, float]:
    ''' Sends the given prompt to the model that runs on the local server and returns the response as well as the execution time'''
    return _send_to_chat(model, messages, True, max_tokens, temperature)

def send_to_openai_completion(prompt: str, model: str, max_tokens: int = 64, temperature: float = 0.0) -> Tuple[str, float]:
    ''' Sends the given prompt to the OpenAI API completion endpoint and returns the response '''
    return _send_to_completion(prompt, model, False, max_tokens, temperature)

def send_to_openai_chat(messages: list, model: str, max_tokens: int = 64, temperature: float = 0.0) -> Tuple[str, float]:
    ''' Sends the given messages to the OpenAI API chat endpoint and returns the response '''
    return _send_to_chat(model, messages, False, max_tokens, temperature)

def convert_triple_to_text_gradio_server(instruction_generator: Callable[[], str], prompt: str, max_tokens: int = 128, temperature: int = 0) -> Tuple[str, float]:
    instruction = instruction_generator()
    result = send_to_local_gradio_server(instruction, prompt, max_tokens, temperature)

    result = json.loads(result)
    response = _remove_prompt_from_gradio_output(result["data"][0])
    
    return response, result["duration"]

def send_to_local_gradio_server(instruction: str, prompt: str, max_tokens: int = 128, temperature: float = 0.0, top_p: float = 0.75, top_k: int = 40, beams: int = 4) -> dict:
    ''' Sends the given prompt to the local Gradio server and returns the response '''
    url = "http://0.0.0.0:7860/run/predict"
    payload = json.dumps({
      "data": [
        instruction,
        prompt,
        temperature,
        top_p,
        top_k,
        beams,
        max_tokens,
        False
      ]
    })
    headers = {
      'Content-Type': 'application/json'
    }

    response_content = requests.request("POST", url, headers=headers, data=payload)

    return response_content.text

def _send_to_completion(prompt: str, model: str, local_server: bool = True, max_tokens: int = 64, temperature: float = 0.0) -> Tuple[str, float]:
    _set_openai_parameters(local_server)
    start_time = time.time()
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )
    response_content = response.choices[0].text
    end_time = time.time()

    duration = end_time - start_time
    return response_content, duration

def _send_to_chat(model: str, messages: list, local_server: bool = True, max_tokens: int = 64, temperature: float = 0.0) -> Tuple[str, float]:
    _set_openai_parameters(local_server)
    start_time = time.time()
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    response_content = response.choices[0].message.content
    end_time = time.time()

    duration = end_time - start_time
    return response_content, duration
    
def _set_openai_parameters(local_server: bool = True):
    if local_server:
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:8000/v1"
    else:
        # Get OpenAI API key from .env file
        openai.api_key = os.getenv('OPENAI_API_KEY')
        openai.api_base = "https://api.openai.com/v1"

def _remove_prompt_from_gradio_output(output: str) -> str:
    return output.split("###")[0].strip()
    