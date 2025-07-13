import json
import time
import uuid


def tools_auto_completion(llm, chat_template, messages, max_tokens=1024, kwargs=None):

    if kwargs is not None:
        output = llm(**kwargs)
    else:
        output = llm(prompt=messages, stream=False, max_tokens=max_tokens)

    text = output["choices"][0]["text"]
    if chat_template == "qwen":
        try:
            tools_tag = (
                output["choices"][0]["text"]
                .split("<tool_call>")[-1]
                .split("</tool_call>")[0]
                .strip()
            )
            tools_json = json.loads(tools_tag)
            ChatCompletionMessageToolCall = {
                "index": 0,
                "id": str(uuid.uuid4()),
                "function": {
                    "arguments": str(tools_json["arguments"]),
                    "name": tools_json["name"],
                },
                "type": "function",
            }
            output["choices"][0]["message"] = {
                "content": text,
                "tool_calls": None,
                "parsed": None,
                "tool_calls": [ChatCompletionMessageToolCall],
                "role": "assistant",
            }
        except Exception as e:
            output["choices"][0]["message"] = {
                "content": text,
                "tool_calls": None,
                "parsed": None,
                "role": "assistant",
            }

    elif chat_template == "typhoon-gemma3":
        try:
            tools_tag = (
                output["choices"][0]["text"]
                .split("```json")[-1]
                .split("```")[0]
                .strip()
            )
            tools_json = json.loads(tools_tag)
            tool_calls = []
            for tool in tools_json:
                ChatCompletionMessageToolCall = {
                    "id": "",
                    "function": {
                        "arguments": str(tool["function"]["parameters"]),
                        "name": tool["function"]["name"],
                    },
                    "type": "function",
                }
                tool_calls.append(ChatCompletionMessageToolCall)
            output["choices"][0]["message"] = {
                "content": text,
                "tool_calls": None,
                "parsed": None,
                "tool_calls": tool_calls,
                "role": "assistant",
            }
        except Exception as e:
            output["choices"][0]["message"] = {
                "content": text,
                "tool_calls": None,
                "parsed": None,
                "role": "assistant",
            }

    elif chat_template == "gemma":
        try:
            tools_tag = (
                output["choices"][0]["text"]
                .split("```json")[-1]
                .split("```")[0]
                .strip()
            )
            tools_json = json.loads(tools_tag)
            tool_calls = []
            ChatCompletionMessageToolCall = {
                "id": "",
                "function": {
                    "arguments": str(tools_json["parameters"]),
                    "name": tools_json["name"],
                },
                "type": "function",
            }
            tool_calls.append(ChatCompletionMessageToolCall)
            output["choices"][0]["message"] = {
                "content": text,
                "tool_calls": None,
                "parsed": None,
                "tool_calls": tool_calls,
                "role": "assistant",
            }
        except Exception as e:
            output["choices"][0]["message"] = {
                "content": text,
                "tool_calls": None,
                "parsed": None,
                "role": "assistant",
            }

    del output["choices"][0]["text"]
    return output


def json_completion(
    inference_engine,
    prompt,
    response_format=None,
    max_tokens=1024,
    model_type="llamacpp",
    image_embeddings=None,
    tokenizer=None,
    model=None,
    kwargs=None,
):

    if model_type == "llamacpp":
        from llama_cpp.llama import LlamaGrammar

        json_string = json.dumps(response_format, indent=4)
        json_string = (
            json_string.replace("'", '"')
            .replace("True", "true")
            .replace("False", "false")
            .replace("json_schema", "object")
        )
        response_grammar = LlamaGrammar.from_json_schema(json_string)
        if kwargs is not None:
            kwargs["grammar"] = response_grammar
            output = inference_engine(**kwargs)
        else:
            output = inference_engine(
                prompt=prompt,
                grammar=response_grammar,
                max_tokens=max_tokens,
                stream=False,
            )

        # Parse the output to JSON
        output_json = output["choices"][0]["text"]
        output["choices"][0]["message"] = {
            "content": output_json,
            "tool_calls": None,
            "parsed": json.loads(output_json),
            "role": "assistant",
        }
        del output["choices"][0]["text"]
        return output

    elif model_type == "exllamav2":
        from lmformatenforcer import JsonSchemaParser
        from exllamav2.generator import ExLlamaV2Sampler
        from exllamav2.generator.filters import ExLlamaV2PrefixFilter
        from function.inference_lmfe_wrapper import ExLlamaV2TokenEnforcerFilter

        if response_format is not None:
            json_string = json.dumps(response_format, indent=4)
            json_string = (
                json_string.replace("'", '"')
                .replace("True", "true")
                .replace("False", "false")
                .replace("json_schema", "object")
            )
            response_format_json = json.loads(json_string)
            if response_format == {"type": "json_object"}:
                response_format_json = None
                try:
                    schema_parser = JsonSchemaParser(None)
                except Exception as e:
                    print(f"Error creating JsonSchemaParser with None: {e}")
                    schema_parser = None
            else:
                schema_parser = JsonSchemaParser(response_format_json)
            try:
                output = inference_engine.generate(
                    prompt=[prompt],
                    filter=[
                        (
                            ExLlamaV2PrefixFilter(model, tokenizer, ["{", " {"]),
                            ExLlamaV2TokenEnforcerFilter(
                                model, tokenizer, schema_parser
                            ),
                        )
                    ],
                    max_new_tokens=max_tokens,
                    filter_prefer_eos=True,
                    add_bos=True,
                    encode_special_tokens=True,
                    decode_special_tokens=True,
                    stop_conditions=[151645, 151643],
                    gen_settings=ExLlamaV2Sampler.Settings.greedy(),
                    embeddings=[image_embeddings],
                    completion_only=True,
                )
            except Exception as e:
                print(f"Error during inference_engine.generate with schema parser: {e}")
        else:
            print("Response format is None, using default settings.")
            output = inference_engine.generate(
                prompt=[prompt],
                max_new_tokens=max_tokens,
                add_bos=True,
                encode_special_tokens=True,
                decode_special_tokens=True,
                stop_conditions=[151645, 151643],
                gen_settings=ExLlamaV2Sampler.Settings.greedy(),
                embeddings=[image_embeddings],
                completion_only=True,
            )
        try:

            json_payload = json.loads(output[0])
            output = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"""{json_payload}""",
                            "tool_calls": None,
                        }
                    }
                ]
            }
        except:
            output = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"""{output[0]}""",
                            "tool_calls": None,
                        }
                    }
                ]
            }

        return output


def completion(
    inference_engine,
    chat_template,
    prompt,
    tokenizer=None,
    image_embeddings=None,
    stream=False,
    json_output=False,
    response_format: dict = None,
    tool_choice=None,
    max_tokens=1024,
    model_type="llamacpp",
    model=None,
    reasoning_effort=None,
    messages=None,
):

    if model_type == "llamacpp":
        kwargs = {"prompt": prompt, "stream": stream, "max_tokens": max_tokens}
        if reasoning_effort is not None and chat_template == "qwen":
            kwargs["temperature"] = 0.6
            kwargs["top_p"] = 0.95
            kwargs["top_k"] = 20
            kwargs["min_p"] = 0
            kwargs["max_tokens"] = 38912  ## Recommended by Qwen team

        elif chat_template == "qwen":
            kwargs["temperature"] = 0.7
            kwargs["top_p"] = 0.8
            kwargs["top_k"] = 20

        if stream:
            output = inference_engine(**kwargs)
            return output
        else:
            if json_output:
                return json_completion(
                    inference_engine,
                    prompt,
                    response_format,
                    max_tokens=max_tokens,
                    kwargs=kwargs,
                )

            if tool_choice == "auto":
                return tools_auto_completion(
                    inference_engine,
                    chat_template,
                    prompt,
                    max_tokens=max_tokens,
                    kwargs=kwargs,
                )

            if tool_choice != None:
                raise ValueError(f"tool_choice {tool_choice} is not supported")

            output = inference_engine(**kwargs)
            return output

    if model_type == "llamacpp-vision":
        kwargs = {"prompt": prompt, "stream": stream, "max_tokens": max_tokens}
        if stream:
            output = inference_engine(**kwargs)
            return output
        else:
            if json_output:
                return json_completion(
                    inference_engine,
                    prompt,
                    response_format,
                    max_tokens=max_tokens,
                    kwargs=kwargs,
                )

            if tool_choice == "auto":
                return tools_auto_completion(
                    inference_engine,
                    chat_template,
                    prompt,
                    max_tokens=max_tokens,
                    kwargs=kwargs,
                )

            if tool_choice != None:
                raise ValueError(f"tool_choice {tool_choice} is not supported")

            kwargs["temperature"] = 0.1
            kwargs["repeat_penalty"] = 1.05
            output = inference_engine(**kwargs)

            output["choices"][0]["message"] = {
                "role": "assistant",
                "content": output["choices"][0]["text"],
                "tool_calls": None,
                "parsed": None,
            }
            return output

    elif model_type == "hf":

        output = inference_engine.generate(
            **prompt,
            temperature=0.1,
            max_new_tokens=4096,
            num_return_sequences=1,
            repetition_penalty=1.2,
            do_sample=True,
        )

        prompt_length = prompt["input_ids"].shape[1]
        new_tokens = output[:, prompt_length:]
        text_output = image_embeddings.tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )

        text_output[0] = text_output[0].strip()

        return {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "created": time.time(),
            "model": model_type,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text_output[0],
                        "refusal": None,
                        "annotations": [],
                    },
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt["input_ids"][0]),
                "completion_tokens": len(new_tokens[0]),
                "total_tokens": len(prompt["input_ids"][0]) + len(new_tokens[0]),
            },
            "service_tier": "free",
            "system_fingerprint": model_type,
        }

    elif model_type == "exllamav2":
        if json_output:
            return json_completion(
                inference_engine,
                prompt,
                response_format,
                max_tokens=max_tokens,
                model_type=model_type,
                image_embeddings=image_embeddings,
                tokenizer=tokenizer,
                model=model,
            )

        from exllamav2.generator import ExLlamaV2Sampler

        if stream:
            raise NotImplementedError(
                "Streaming is not supported for ExLlamaV2 models in this function."
            )

        try:
            output = inference_engine.generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                add_bos=True,
                encode_special_tokens=True,
                decode_special_tokens=True,
                stop_conditions=[151645, 151643],
                gen_settings=ExLlamaV2Sampler.Settings.greedy(),
                embeddings=image_embeddings,
                completion_only=True,
            )
        except Exception as e:
            print(f"Error during inference_engine.generate: {e}")

        output = {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "created": time.time(),
            "model": "exllamav2",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"""{output}""",
                        "refusal": None,
                        "annotations": [],
                    },
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(output.split()),
                "total_tokens": len(prompt.split()) + len(output.split()),
            },
            "service_tier": "free",
            "system_fingerprint": "exllamav2",
        }
        return output
