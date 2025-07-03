from openai import OpenAI

API_KEY="xxx"
client = OpenAI(
    api_key=API_KEY,
    base_url="xxx"
)
res = client.chat.completions.create(
    model="my-typhoon2",
    messages=[
        {"role": "user", "content": "สวัสดี คุณทำอะไรได้บ้าง"}
    ],
    response_format={
        "type": "object",
        "properties": {
            "greeting": {
                "type": "string",
                "description": "A greeting message"
            },
        },
        "required": ["greeting"]
    }
)

print(res)
print('---')