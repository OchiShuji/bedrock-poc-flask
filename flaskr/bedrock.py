import boto3
import json
from typing import Dict, Any
from threading import Lock

class BedrockLLMWrapper:
    def __init__(self, model_id: str, region_name: str = 'ap-northeast-1', max_tokens: int = 1000):
        self.model_id = model_id
        self.region_name = region_name
        self.max_tokens = max_tokens
        self.bedrock_client = boto3.client(service_name='bedrock-runtime', region_name=self.region_name)

        self.model_body_methods = {
            'anthropic.claude-v2:1': self.body_claude_text_completion,
            'anthropic.claude-3-5-sonnet-20240620-v1:0': self.body_claude_messages_api,
            'anthropic.claude-3-haiku-20240307-v1:0': self.body_claude_messages_api,
            'amazon.titan-text-express-v1': self.body_titan_text
        }

        self.response_methods = {
            'anthropic.claude-v2:1': self.get_response_claude_text_completion,
            'anthropic.claude-3-5-sonnet-20240620-v1:0': self.get_response_claude_messages_api,
            'anthropic.claude-3-haiku-20240307-v1:0': self.get_response_claude_messages_api,
            'amazon.titan-text-express-v1': self.get_response_titan_text
        }

    def body_claude_text_completion(self, input_text: str, temperature: float, top_p: float) -> Dict[str, Any]:
        return {
            "prompt": f"\n\nHuman: {input_text}\n\nAssistant:",
            "max_tokens_to_sample": self.max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }

    def body_claude_messages_api(self, input_text: str, temperature: float, top_p: float) -> Dict[str, Any]:
        return {
            "anthropic_version": "bedrock-2023-05-31", 
            "messages": [{"role": "user", "content": input_text}],
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }

    def body_titan_text(self, input_text: str, temperature: float, top_p: float) -> Dict[str, Any]:
        return {
            "inputText": f"User: {input_text} \nBot",
            "textGenerationConfig": {
                "maxTokenCount": self.max_tokens,
                "stopSequences": [],
                "temperature": temperature,
                "topP": top_p
            }
        }

    def get_response_claude_text_completion(self, response_body: Dict[str, Any]) -> str:
        return response_body.get('completion', '')

    def get_response_claude_messages_api(self, response_body: Dict[str, Any]) -> str:
        messages = response_body.get('content', [])
        return messages[0].get('text', '') if messages else ''

    def get_response_titan_text(self, response_body: Dict[str, Any]) -> str:
        results = response_body.get('results', [])
        return results[0].get('outputText', '') if results else ''

    def invoke(self, prompt: str, temperature: float, top_p: float) -> str:
        if self.model_id not in self.model_body_methods:
            raise ValueError(f"Unsupported model ID: {self.model_id}")
        
        body_method = self.model_body_methods[self.model_id]
        response_method = self.response_methods[self.model_id]

        body = json.dumps(body_method(prompt, temperature, top_p))
        accept = 'application/json'
        contentType = 'application/json'

        lock = Lock()

        if not lock.acquire():
            return ""

        try:
            response = self.bedrock_client.invoke_model(
                body=body,
                modelId=self.model_id,
                accept=accept,
                contentType=contentType
            )
            response_body = json.loads(response.get('body').read())
            return response_method(response_body)
        except Exception as e:
            print(f"Error invoking model: {e}")
            raise
        finally:
            lock.release()
