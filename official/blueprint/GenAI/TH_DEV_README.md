## Supported Models
### LLM

Model name | Reasoning Mode | Function Call | Function Call with Reasoning | JSON Output |
| --- | --- | --- | --- | --- |
qwen3-32b-fast | yes | yes | yes | yes |
qwen3-32b-128k | yes | yes | yes | yes |
qwen3-32b | yes | yes | yes | yes |
qwen3-14b-fast | yes | yes | yes | yes |
qwen3-14b-128k | yes | yes | yes | yes |
qwen3-14b | yes | yes | yes | yes |
qwen3-8b-fast | yes | yes | yes | yes |
qwen3-8b-128k | yes | yes | yes | yes |
qwen3-8b | yes | yes | yes | yes |
qwen3-4b-fast | yes | yes | yes | yes |
qwen3-4b | yes | yes | yes | yes |
qwen3-a3b-fast | yes | yes | yes | yes |
qwen3-a3b | yes | yes | yes | yes |
typhoon-gemma3 | yes | partial | partial | yes |
gemma3-27b | no | partial | no | yes |
gemma3-12b | no | partial | no | yes |

The tool_choice need to be `tool_choice="auto"` only. Other values are not supported.

---
### VLM

Model name | 
| --- |
qwen2.5-vl-7b |
qwen2.5-vl-32b |
ui-tars-1.5-7b |

---
### Embedding
Model name | Multilingual | 
| --- | --- |
bge-m3 | yes |
qwen3-embedding-0.6b | yes |
qwen3-embedding-4b | yes |
qwen3-embedding-8b | yes |


## Reasoning Example

### Native Reasoning
```
response = client.chat.completions.create(
    model="qwen3-14b", #qwen3-a3b, typhoon-gemma3, qwen3-14b
    #reasoning_effort="high", #uncommented to enable reasoning, "low", "medium", "high" is not effected, All levels are same results
    messages=[
        {"role": "user", "content": "สวัสดี คุณทำอะไรได้บ้าง"}
    ]
)
```

### Function Call Reasoning
```
response = client.chat.completions.create(
    model="qwen3-14b", #qwen3-a3b, typhoon-gemma3, qwen3-14b
    reasoning_effort="high",
    messages=[
        {"role": "user", "content": "สวัสดี คุณทำอะไรได้บ้าง"}
    ],
    tools=[{
        "type": "function",
        "function": {
          "name": "UserDetail",
          "parameters": {
            "type": "object",
            "title": "UserDetail",
            "properties": {
              "name": {
                "title": "Name",
                "type": "string"
              },
              "age": {
                "title": "Age",
                "type": "integer"
              }
            },
            "required": [ "name", "age" ]
          }
        }
      }],
    tool_choice="auto"
)
```

## JSON Output Example

### Native JSON Output
```
response = client.beta.chat.completions.parse(
    model="qwen3-14b", #qwen3-a3b, typhoon-gemma3, qwen3-14b
    messages=messages,
    response_format={
        "type": "json_object",
        "object": {
            "name": "UserDetail",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }
    },
)
```

### Pydantic JSON Output
Support both `pydantic` and `zod`. Same compatibility like `gemini` [Learn more.](https://ai.google.dev/gemini-api/docs/openai#javascript)


```
class LLMDecisionIntent(BaseModel):
    question_id: int
    solution_id: int

response = client.beta.chat.completions.parse(
    model="qwen3-14b", #qwen3-a3b, typhoon-gemma3, qwen3-14b
    messages=messages,
    response_format=LLMDecisionIntent
)
```

## Vision Example
### Native Vision
```
response = client.chat.completions.create(
    model="qwen2.5-vl-7b",
    messages=[
        {
            "role": "user",
            "content": [ 
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    }
                },
            ],
        }
    ],
    max_tokens=100
)
```

## Embedding Example

### Native OpenAI Client
```
response = client.embeddings.create(
    model="qwen3-embedding-8b",
    input="สวัสดี คุณทำอะไรได้บ้าง"
)
```