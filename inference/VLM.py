import base64
import requests
import anthropic
from vllm import LLM, EngineArgs, SamplingParams
from transformers import AutoProcessor, AutoTokenizer
from dataclasses import asdict
from io import BytesIO

class VLM:
    def __init__(self, model_name, api_key, gpu_count=None, size=None):
        self.model_name = model_name.lower()
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.gpu_count = gpu_count

        if "o1" in self.model_name:
            self.client = None  # OpenAI client will use requests
        elif "gpt-4o" in self.model_name:
            self.client = None # OpenAI client will use requests
        elif "claude" in self.model_name:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        elif "qwen2_5_vl" in self.model_name:
            self.engine_args, self.stop_token_ids = self.load_qwen2_5_vl(size)
        elif "internvl" in self.model_name:
            self.engine_args, self.stop_token_ids = self.load_internvl(size)
        elif "llava-next" in self.model_name:
            self.engine_args, self.stop_token_ids = self.load_llava_next()
        elif "llava-onevision" in self.model_name:
            self.engine_args, self.stop_token_ids = self.load_llava_onevision()
        else:
            raise ValueError("Unsupported model name.")
        
        if "qwen2_5_vl" in self.model_name or "internvl" in self.model_name or "llava-next" in self.model_name or "llava-onevision" in self.model_name:
            self.engine_args = asdict(self.engine_args) | {"seed": 0}
            self.engine_args["tensor_parallel_size"] = self.gpu_count if self.gpu_count else 1
            self.llm = LLM(**self.engine_args)

    def encode_image(self, image):
        buffered = BytesIO()
        image = image.convert("RGB")
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def run(self, prompt, imgs=None, past_messages=None):
        if imgs is None:
            imgs = []
        if past_messages is None:
            past_messages = []

        if "o1" in self.model_name:
            return self.run_o1(prompt, imgs, past_messages)
        elif "gpt-4o" in self.model_name:
            return self.run_gpt4o(prompt, imgs, past_messages)
        elif "claude" in self.model_name:
            return self.run_claude(prompt, imgs, past_messages)
        elif "qwen2_5_vl" in self.model_name or "internvl" in self.model_name or "llava-next" in self.model_name or "llava-onevision" in self.model_name:
            return self.run_open(prompt, imgs, past_messages)
        else:
            raise ValueError("Unsupported model name.")

    def run_o1(self, prompt, imgs, past_messages):
        messages = past_messages.copy()
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        })

        for base64_image in imgs:
            messages[-1]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            })

        payload = {
            "model": "o1",
            "messages": messages
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
        
        usage = response.json()['usage']
        return response.json()['choices'][0].get('message', {}).get('content', ''), usage.get('total_tokens', -1), usage.get('prompt_tokens', -1), usage.get('completion_tokens', -1), usage.get('completion_tokens_details', -1).get('reasoning_tokens', -1)

    def run_gpt4o(self, prompt, imgs, past_messages, temperature=0.2):
        messages = past_messages.copy()
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        })

        for base64_image in imgs:
            messages[-1]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            })

        payload = {
            "model": "gpt-4o",
            "messages": messages,
            "temperature": temperature
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
        usage = response.json()['usage']
        return response.json()['choices'][0].get('message', {}).get('content', ''), usage.get('total_tokens', -1), usage.get('prompt_tokens', -1), usage.get('completion_tokens', -1)


    def run_claude(self, prompt, imgs, past_messages, temperature=0.2):
        messages = past_messages.copy()
        content = [{"type": "text", "text": prompt}]
        for base64_image in imgs:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64_image,
                }
            })
        messages.append({"role": "user", "content": content})
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4096,
            messages=messages,
            temperature=temperature
        )
        usage = response.model_dump()['usage']
        return response.content[0].text, usage.get('input_tokens', -1), usage.get('output_tokens', -1)
    
    def run_open(self, prompt, imgs, past_messages, temperature=0.2):
        if self.stop_token_ids is not None:
            if None in self.stop_token_ids:
                self.stop_token_ids.remove(None)

        sampling_params = SamplingParams(
            temperature=temperature, max_tokens=2048, stop_token_ids=self.stop_token_ids
        )
        outputs = self.llm.chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *(
                            {"type": "image_pil", "image_pil": img} for img in imgs
                        )
                    ]
                }
            ],
            sampling_params=sampling_params
        )
        return outputs[0].outputs[0].text, ''
        
    def load_qwen2_5_vl(self, size='7B'):
        model_name = f"Qwen/Qwen2.5-VL-{size}-Instruct"
        engine_args = EngineArgs(
            model=model_name,
            max_model_len=40000,
            max_num_seqs=5,
            limit_mm_per_prompt={"image": 2}
        )
        return engine_args, None

    def load_internvl(self, size='8B'):
        model_name = f"OpenGVLab/InternVL2_5-{size}"
        engine_args = EngineArgs(
            model=model_name,
            trust_remote_code=True,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 2},
            mm_processor_kwargs={"max_dynamic_patch": 4}
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
        return engine_args, stop_token_ids
    
    def load_llava_next(self):
        model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
        engine_args = EngineArgs(
            model=model_name,
            max_model_len=8192,
            max_num_seqs=16,
            limit_mm_per_prompt={"image": 2}
        )
        return engine_args, None
    
    def load_llava_onevision(self):
        model_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
        engine_args = EngineArgs(
            model=model_name,
            max_model_len=30000,
            max_num_seqs=16,
            limit_mm_per_prompt={"image": 2}
        )
        return engine_args, None