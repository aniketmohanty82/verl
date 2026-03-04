import logging
import uuid
from copy import deepcopy
from typing import Any, Optional

import ray
from ray import serve
import torch

from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutReplica, TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

class RayServeReplica(RolloutReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.router_handle = None

    async def init_standalone(self):
        """
        Bind to the already-running global Ray Serve Inference Scheduler.
        """
        # Read the configurable Ray Serve app name from the extension dictionary (defaults to "default")
        serve_app_name = "default"
        if self.config.custom is not None:
            serve_app_name = self.config.custom.get("serve_app_name", "default")
            
        logger.info(f"Initializing RayServeReplica {self.replica_rank} by connecting to '{serve_app_name}' Ray Serve app.")
        
        # Connect to the external inference scheduler
        self.router_handle = serve.get_app_handle(serve_app_name)
        
        self._server_address = "ray-serve-rpc"
        self._server_handle = self

    # Hybrid and colocated modes are not applicable for external Ray Serve endpoints
    async def init_hybrid(self, worker_group):
        await self.init_standalone()
    
    async def init_hybrid_colocated(self, worker_group, resource_pool):
        await self.init_standalone()

    async def init_colocated(self, resource_pool):
        await self.init_standalone()

    async def launch_servers(self):
        pass

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        **kwargs
    ) -> TokenOutput:
        """Route generation request to py-inference-scheduler via Ray RPC."""
        # 1. Detokenize prompt
        prompt_string = self.model_config.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        
        # 2. Map Sampling Parameters
        sampling_params_copy = deepcopy(sampling_params)
        if "max_new_tokens" in sampling_params_copy:
            max_tokens = sampling_params_copy.pop("max_new_tokens")
        elif "max_tokens" in sampling_params_copy:
            max_tokens = sampling_params_copy.pop("max_tokens")
        else:
            # support the framework-standard fallback math:
            max_tokens = self.config.response_length + self.config.prompt_length - len(prompt_ids)
            
        # Optional clamp for safety (like vLLM does) against max_model_len
        if self.config.max_model_len is not None:
            max_possible_tokens = self.config.max_model_len - len(prompt_ids)
            max_tokens = max(0, min(max_tokens, max_possible_tokens))
            
        openai_params = {
            "max_tokens": max_tokens,
            "temperature": sampling_params_copy.get("temperature", 1.0),
            "top_p": sampling_params_copy.get("top_p", 1.0),
            "top_k": sampling_params_copy.get("top_k", -1),
            "repetition_penalty": sampling_params_copy.get("repetition_penalty", 1.0),
        }
        
        logprobs_enabled = True if sampling_params_copy.get("logprobs", 0) > 0 else False
        if logprobs_enabled:
            openai_params["logprobs"] = True
            openai_params["top_logprobs"] = sampling_params_copy.get("logprobs", 1)

        # 3. Fire High-Speed Ray RPC Request
        target_model = getattr(self.model_config.hf_config, "name_or_path", "qwen-32b")
        
        try:
            # Send RPC to the Ingress app handle (Ray Serve's native create_chat_completion method)
            # The method name depends on the exact Ingress Class in ray.llm. Usually it's `create_chat_completion_openai_v1`
            # or `chat_completions`. We import ChatCompletionRequest structure if needed, or pass raw dicts.
            
            # Since Ray's FastAPI bindings can accept Request objects, passing native python dict is allowed.
            request_dict = {
                "model": target_model,
                "messages": [{"role": "user", "content": prompt_string}],
                "stream": False,
                **openai_params
            }
            
            from ray.llm._internal.serve.observability.logging import get_logger
            from ray.llm._internal.serve.core.configs.openai_api_models import ChatCompletionRequest
            import json

            logger.info(f"Sending request to py-inference-scheduler via RPC: {request_id}")
            request_obj = ChatCompletionRequest(**request_dict)

            # OpenAIIngress exposes 'chat(self, body: ChatCompletionRequest, request: Request)'
            response = await self.router_handle.options(stream=False).chat.remote(
                body=request_obj,
                request=None
            )
            
            # The FastAPI endpoint returns a Starlette JSONResponse. We must read its body.
            response_json = json.loads(response.body.decode("utf-8"))
            
            output_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            if logprobs_enabled:
                log_probs_data = response_json.get("choices", [{}])[0].get("logprobs", {}).get("content", [])
                log_probs = [item.get("logprob") for item in log_probs_data]
            # The result is typically an openai.ChatCompletion-like object or a standard dictionary

            encoded_response = self.model_config.tokenizer(output_text, add_special_tokens=False)
            token_ids = encoded_response["input_ids"]

            return TokenOutput(
                token_ids=token_ids,
                log_probs=log_probs,
                stop_reason="completed",
                num_preempted=0
            )
        except Exception as e:
            logger.error(f"Ray RPC generation failed: {e}")
            raise e
        
    async def clear_kv_cache(self): pass
    async def wake_up(self): pass
    async def sleep(self): pass
    async def start_profile(self, **kwargs): pass
    async def stop_profile(self): pass
    async def abort_all_requests(self): pass
    async def resume_all_requests(self): pass
