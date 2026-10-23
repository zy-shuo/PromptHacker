"""
LLM API call wrapper module
Provides unified API interface with retry, rate limiting and other features
Supports both API calls and local inference modes
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI
from core.api_config import resolve_api_endpoint
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMWrapper:
    """Optimized LLM API wrapper class supporting both API calls and local inference"""
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 20
    API_TIMEOUT = 20

    def __init__(self, model_name, max_tokens=2048, temperature=1, top_p=0.9, api_key=None, base_url=None, 
                 use_local=False, local_model_path=None, device_map="auto", torch_dtype=torch.bfloat16,
                 api_role="aux", token_parameter="max_tokens", enable_thinking=None):
        """
        Initialize LLM wrapper
        
        Args:
            model_name: Model name
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature parameter
            top_p: Top-p sampling parameter
            api_key: API key
            base_url: API base URL
            use_local: Whether to use local inference mode
            local_model_path: Local model path (required when use_local=True)
            device_map: Device mapping configuration
            torch_dtype: Model data type
            api_role: "target" for the target model; "aux" for framework models
            token_parameter: API length field: max_tokens or max_completion_tokens
            enable_thinking: Optional provider-specific API request field
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        if token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("token_parameter must be max_tokens or max_completion_tokens.")
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise ValueError("enable_thinking must be a boolean when specified.")
        self.token_parameter = token_parameter
        self.enable_thinking = enable_thinking
        self.call_count = 0  # Record API call count
        endpoint = resolve_api_endpoint(api_role)
        self.api_role = api_role
        if api_key is not None or base_url is not None:
            self.api_key = api_key
            self.base_url = base_url
        else:
            self.api_key = endpoint.api_key
            self.base_url = endpoint.base_url
        self.use_local = use_local
        
        # Local inference related attributes
        self.local_model = None
        self.local_tokenizer = None
        self.local_pipeline = None
        
        # Local weights are loaded on the first generation, from this path only.
        self.local_model_path = local_model_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        if self.use_local:
            if not local_model_path or not Path(local_model_path).is_dir():
                raise ValueError("Local inference requires an existing local_model_path directory.")
    
    def _initialize_local_model(self, model_path: str, device_map: str = "auto", torch_dtype=torch.bfloat16):
        """
        Initialize local model
        
        Args:
            model_path: Local model path
            device_map: Device mapping configuration
            torch_dtype: Model data type
        """
        try:
            print(f"Loading local model: {model_path}")
            
            self.local_tokenizer = AutoTokenizer.from_pretrained(
                model_path, local_files_only=True, trust_remote_code=False,
            )
            self.local_model = AutoModelForCausalLM.from_pretrained(
                model_path, local_files_only=True, trust_remote_code=False,
                torch_dtype=torch_dtype, device_map=device_map,
            )
            self.local_pipeline = transformers.pipeline(
                "text-generation",
                model=self.local_model,
                tokenizer=self.local_tokenizer,
            )
            
            print(f"Local model loading completed: {model_path}")
            
        except Exception as e:
            print(f"Local model loading failed: {e}")
            raise e
    
    def _generate_local(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate response using local model
        
        Args:
            messages: Message list
            
        Returns:
            str: Model generated response
        """
        try:
            # Set terminators
            terminators = [
                self.local_pipeline.tokenizer.eos_token_id,
            ]
            
            # If tokenizer supports special terminators, add them
            if hasattr(self.local_pipeline.tokenizer, 'convert_tokens_to_ids'):
                try:
                    eot_id = self.local_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    if eot_id is not None:
                        terminators.append(eot_id)
                except:
                    pass  # Ignore unsupported terminators
            
            generation = {"do_sample": self.temperature > 0}
            if self.temperature > 0:
                generation.update(temperature=self.temperature, top_p=self.top_p)
            outputs = self.local_pipeline(
                messages,
                max_new_tokens=self.max_tokens,
                eos_token_id=terminators,
                **generation,
                pad_token_id=self.local_pipeline.tokenizer.eos_token_id
            )
            
            # Extract generated text
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]["generated_text"]
                if isinstance(generated_text, list) and len(generated_text) > 0:
                    # Return content of last message
                    return generated_text[-1].get("content", "")
                elif isinstance(generated_text, str):
                    return generated_text
            
            return self.API_ERROR_OUTPUT
            
        except Exception as e:
            print(f"Local inference failed: {e}")
            return self.API_ERROR_OUTPUT
    
    def generate(self, messages):
        """
        Generate response (supports both API call and local inference modes)
        
        Args:
            messages: Message list
            
        Returns:
            str: Model generated response
        """
        # Increment call count
        self.call_count += 1
        
        # Select generation method based on mode
        if self.use_local:
            if self.local_pipeline is None:
                self._initialize_local_model(self.local_model_path, self.device_map, self.torch_dtype)
            return self._generate_local(messages)
        else:
            return self._generate_api(messages)
    
    def _generate_api(self, messages):
        """
        Generate response using API call
        
        Args:
            messages: Message list
            
        Returns:
            str: Model generated response
        """
        if not self.api_key:
            setting_name = "TARGET_API_KEY" if self.api_role == "target" else "AUX_API_KEY"
            raise ValueError(f"Missing API key for {self.api_role} models; configure {setting_name}.")
            
        output = self.API_ERROR_OUTPUT
        for retry in range(self.API_MAX_RETRY):
            try:
                client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                request = {
                    "model": self.model_name, "messages": messages,
                    self.token_parameter: self.max_tokens,
                    "temperature": self.temperature, "top_p": self.top_p,
                }
                if self.enable_thinking is not None:
                    request["extra_body"] = {"enable_thinking": self.enable_thinking}
                response = client.chat.completions.create(**request)
                output = response.choices[0].message.content
                break
            except Exception as e:
                print(f"API call failed (retry {retry + 1}/{self.API_MAX_RETRY}): {type(e).__name__}: {e}")
                if retry < self.API_MAX_RETRY - 1:
                    time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        
        return output
    def clean_up(self):
        """
        Clean up API call count
        """
        self.call_count = 0
    
    def release_local_model(self):
        """
        Release local model resources
        """
        if self.use_local:
            try:
                if self.local_pipeline is not None:
                    # Clean up pipeline
                    del self.local_pipeline
                    self.local_pipeline = None
                
                if self.local_model is not None:
                    del self.local_model
                    self.local_model = None
                
                if self.local_tokenizer is not None:
                    del self.local_tokenizer
                    self.local_tokenizer = None
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                print("Local model resources released")
                
            except Exception as e:
                print(f"Error releasing local model resources: {e}")
    
    def __del__(self):
        """
        Destructor, automatically release resources
        """
        if hasattr(self, 'use_local') and self.use_local:
            self.release_local_model()
