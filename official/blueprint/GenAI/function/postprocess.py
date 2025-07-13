import json
from typing import List, Iterator
import uuid


def generate_response(output: Iterator) -> Iterator[str]:
    for response in output:

        if "</think>" in response["choices"][0]["text"]:
            response["choices"][0]["delta"] = {
                "content": f"{response['choices'][0]['text']}"
            }
            yield "data: " + json.dumps(response) + "\n"
            response["choices"][0]["delta"] = {"content": f"\n\n"}
            yield "data: " + json.dumps(response) + "\n"

        else:
            response["choices"][0]["delta"] = {
                "content": response["choices"][0]["text"]
            }
            yield "data: " + json.dumps(response) + "\n"

    yield "data: [DONE]"


def fake_generate_response(response):
    try:
        if response["choices"][0]["message"]["tool_calls"] is not None:
            response["choices"][0]["finish_reason"] = None
            del response["choices"][0]["message"]["parsed"]

        new_response = json.loads(json.dumps(response))
        message = json.loads(json.dumps(response["choices"][0]["message"]))
        new_response["object"] = "chat.completion.chunk"
        new_response["choices"][0]["delta"] = response["choices"][0]["message"]
        del new_response["choices"][0]["message"]
        del new_response["usage"]

        new_response["choices"][0]["delta"]["tool_calls"][0]["function"][
            "arguments"
        ] = ""
        new_response["choices"][0]["delta"]["content"] = ""
        new_response["choices"][0]["delta"]["refusal"] = None
        yield "data: " + json.dumps(new_response) + "\n"

        del message["tool_calls"][0]["function"]["name"]
        del message["tool_calls"][0]["type"]
        del message["tool_calls"][0]["id"]

        new_response["choices"][0]["delta"] = message

        if (
            "select"
            not in new_response["choices"][0]["delta"]["tool_calls"][0]["function"][
                "arguments"
            ].lower()
        ):
            new_response["choices"][0]["delta"]["tool_calls"][0]["function"][
                "arguments"
            ] = new_response["choices"][0]["delta"]["tool_calls"][0]["function"][
                "arguments"
            ].replace(
                "'", '"'
            )
        else:
            select_index = new_response["choices"][0]["delta"]["tool_calls"][0][
                "function"
            ]["arguments"].index("SELECT")

            new_select = new_response["choices"][0]["delta"]["tool_calls"][0][
                "function"
            ]["arguments"][:select_index].replace("'", '"')
            original_select = new_response["choices"][0]["delta"]["tool_calls"][0][
                "function"
            ]["arguments"][select_index:]

            new_response["choices"][0]["delta"]["tool_calls"][0]["function"][
                "arguments"
            ] = (new_select + original_select)

            end_index = new_response["choices"][0]["delta"]["tool_calls"][0][
                "function"
            ]["arguments"].index(";")

            new_end = new_response["choices"][0]["delta"]["tool_calls"][0]["function"][
                "arguments"
            ][:end_index]
            original_end = new_response["choices"][0]["delta"]["tool_calls"][0][
                "function"
            ]["arguments"][end_index:].replace("'", '"')

            new_response["choices"][0]["delta"]["tool_calls"][0]["function"][
                "arguments"
            ] = (new_end + original_end)

        del new_response["choices"][0]["delta"]["role"]

        for idx, t in enumerate(
            new_response["choices"][0]["delta"]["tool_calls"][0]["function"][
                "arguments"
            ]
        ):
            arg_response = json.loads(json.dumps(new_response))
            if idx != 0:
                del arg_response["choices"][0]["delta"]["content"]
            arg_response["choices"][0]["delta"]["tool_calls"][0]["function"][
                "arguments"
            ] = t
            yield "data: " + json.dumps(arg_response) + "\n"

        new_response["choices"][0] = {
            "index": 0,
            "logprobs": None,
            "delta": {},
            "finish_reason": "tool_calls",
        }
        yield "data: " + json.dumps(new_response) + "\n"

        yield "data: [DONE]"

    except Exception as e:
        new_response = json.loads(json.dumps(response))

        new_response["object"] = "chat.completion.chunk"
        new_response["choices"][0]["delta"] = response["choices"][0]["message"]
        del new_response["choices"][0]["message"]
        del new_response["usage"]

        yield "data: " + json.dumps(new_response) + "\n"
        yield "data: [DONE]"


def tools_generate_response(output: Iterator, chat_template) -> Iterator[str]:
    is_tools = False
    should_continue = False
    is_newline = False
    accumulate_tools = ""
    accumulate_text = ""
    new_line_content = None
    chat_id = str(uuid.uuid4())
    start_tool_tag, end_tool_tag = get_tool_tags(chat_template)
    for response in output:
        accumulate_text += response["choices"][0]["text"]
        for start_tag in start_tool_tag:
            if start_tag in accumulate_text and not is_tools:
                is_tools = True
                should_continue = True
        if should_continue:
            should_continue = False
            response["choices"][0]["delta"] = {"content": "\n"}
            yield "data: " + json.dumps(response) + "\n"
            continue

        if (
            end_tool_tag in response["choices"][0]["text"]
            and accumulate_tools != ""
            and is_tools
        ):
            accumulate_tools = accumulate_tools.strip()
            start_json = accumulate_tools.find("[")
            end_json = accumulate_tools.rfind("]") + 1

            if start_json != -1:
                tools_json = json.loads(accumulate_tools)[0]
                accumulate_tools = accumulate_tools[start_json:end_json]
            else:
                tools_json = json.loads(accumulate_tools)

            if chat_template == "qwen" or chat_template == "gemma":
                ChatCompletionMessageToolCall = {
                    "index": 0,
                    "id": chat_id,
                    "function": {"arguments": "", "name": tools_json["name"]},
                    "type": "function",
                }

            elif chat_template == "typhoon-gemma3":
                ChatCompletionMessageToolCall = {
                    "id": chat_id,
                    "function": {
                        "arguments": "",
                        "name": tools_json["function"]["name"],
                    },
                    "type": "function",
                }

            response["choices"][0]["delta"] = {
                "tool_calls": [ChatCompletionMessageToolCall]
            }
            new_response = json.loads(json.dumps(response))
            yield "data: " + json.dumps(new_response) + "\n"

            if chat_template == "qwen" or chat_template == "gemma":
                ChatCompletionMessageToolCall = {
                    "index": 0,
                    "id": chat_id,
                    "function": {
                        "arguments": str(tools_json["arguments"]),
                        "name": tools_json["name"],
                    },
                    "type": "function",
                }
            elif chat_template == "typhoon-gemma3":
                ChatCompletionMessageToolCall = {
                    "id": chat_id,
                    "function": {
                        "arguments": str(tools_json["function"]["parameters"]),
                        "name": tools_json["function"]["name"],
                    },
                    "type": "function",
                }

            response["choices"][0]["delta"] = {
                "tool_calls": [ChatCompletionMessageToolCall]
            }

            if (
                "select"
                not in response["choices"][0]["delta"]["tool_calls"][0]["function"][
                    "arguments"
                ].lower()
            ):
                response["choices"][0]["delta"]["tool_calls"][0]["function"][
                    "arguments"
                ] = response["choices"][0]["delta"]["tool_calls"][0]["function"][
                    "arguments"
                ].replace(
                    "'", '"'
                )
            else:
                select_index = response["choices"][0]["delta"]["tool_calls"][0][
                    "function"
                ]["arguments"].index("SELECT")

                new_select = response["choices"][0]["delta"]["tool_calls"][0][
                    "function"
                ]["arguments"][:select_index].replace("'", '"')
                original_select = response["choices"][0]["delta"]["tool_calls"][0][
                    "function"
                ]["arguments"][select_index:]

                response["choices"][0]["delta"]["tool_calls"][0]["function"][
                    "arguments"
                ] = (new_select + original_select)

                end_index = response["choices"][0]["delta"]["tool_calls"][0][
                    "function"
                ]["arguments"].index(";")

                new_end = response["choices"][0]["delta"]["tool_calls"][0]["function"][
                    "arguments"
                ][:end_index]
                original_end = response["choices"][0]["delta"]["tool_calls"][0][
                    "function"
                ]["arguments"][end_index:].replace("'", '"')

                response["choices"][0]["delta"]["tool_calls"][0]["function"][
                    "arguments"
                ] = (new_end + original_end)

            yield "data: " + json.dumps(response) + "\n"

            new_response["choices"][0] = {
                "index": 0,
                "logprobs": None,
                "delta": {},
                "finish_reason": "tool_calls",
            }
            yield "data: " + json.dumps(new_response) + "\n"
            continue
        if is_tools:
            accumulate_tools += response["choices"][0]["text"]
            continue
        else:
            if response["choices"][0]["text"] == "</think>" and is_newline:
                think_content = new_line_content.strip() + "</think>" + "\n\n"
                response["choices"][0]["delta"] = {"content": think_content}
                is_newline = False
                new_line_content = None
                yield "data: " + json.dumps(response) + "\n"
                continue
            elif response["choices"][0]["text"].find("\n") > 0 and not is_newline:
                is_newline = True
                new_line_content = response["choices"][0]["text"]
                continue  # Buffer the newline content
            else:
                is_newline = False
                if new_line_content is not None:
                    response["choices"][0]["text"] = (
                        new_line_content + response["choices"][0]["text"]
                    )
                    response["choices"][0]["delta"] = {
                        "content": response["choices"][0]["text"]
                    }
                else:
                    response["choices"][0]["delta"] = {
                        "content": response["choices"][0]["text"]
                    }
                new_line_content = None
                yield "data: " + json.dumps(response) + "\n"
    yield "data: [DONE]"


def get_tool_tags(chat_template: str):
    if chat_template == "qwen":
        start_tool_tag = ["<tool_call>"]
        end_tool_tag = "</tool_call>"
    elif chat_template == "typhoon-gemma3":
        start_tool_tag = ["```tool_code", "```json"]
        end_tool_tag = "```"
    elif chat_template == "gemma":
        start_tool_tag = ["```json"]
        end_tool_tag = "```"
    else:
        start_tool_tag = "<tool_call>"
        end_tool_tag = "</tool_call>"

    return start_tool_tag, end_tool_tag
