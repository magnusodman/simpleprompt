import json
import boto3
from typing import Dict, Any

bedrock = boto3.client("bedrock-runtime")
model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"


def _create_prompt(message: str, system_prompt: None | str = None) -> Dict[str, Any]:
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": message}]
        if system_prompt is None
        else [
            {"role": "user", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        "temperature": 0.7,
    }


def prompt(message: str, system_prompt=None) -> str:
    # Create the prompt
    body = _create_prompt(message, system_prompt)

    # Invoke the model
    response = bedrock.invoke_model(modelId=model_id, body=json.dumps(body))

    # Parse the response
    response_body = json.loads(response.get("body").read())

    return response_body["content"][0]["text"]


if __name__ == "__main__":
    print(prompt(message="What is the capital of France?"))
