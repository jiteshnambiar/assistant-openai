import os
import openai
import urllib.request as req
import json
import certifi
import ssl
from typing import List
import sys, getopt
import requests


class OpenAIError(Exception):
    ...


# to invoke from commandline add the following to your ~/.bash_profile
# export OPENAI_ORG_KEY=<>
# export OPENAI_API_KEY=<>
# assistant() {
#         workon assistant
#         python <path-to-this-file>/assistant.py -q $1
#         deactivate
# }
openai.organization = os.getenv("OPENAI_ORG_KEY") # from https://beta.openai.com/account/org-settings
openai.api_key = os.getenv("OPENAI_API_KEY") # from https://beta.openai.com/account/api-keys


def usage():
    print('chat_gpt.py -q <query>')
    print('usage: chat_gpt.py [-q] <input-query> [-m] <input-model> [-l] [-h]')
    print('arguments:')
    print('-h, --help  show this help message and exit')
    print('-l, --list-models  show list of available models and exit')


def openai_choices(resp: dict) -> List[str]:
    try:
        return [r.get('text') for r in resp.get("choices", [])]
    except (AttributeError, TypeError):
        return []


def ask_openai(
        prompt: str,
        api_key: str,
        model: str = "text-curie-001",
        temperature: int = 0.7,
        max_tokens: int = 256,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        timeout: int = 30,
        return_only_choices: bool = True
) -> dict:
    try:
        data = json.dumps({
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }).encode()

        resp = req.Request(
            url="https://api.openai.com/v1/completions",
            method="POST",
            headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
            }
        )

        context = ssl.create_default_context(cafile=certifi.where())
        with req.urlopen(resp, data, timeout=timeout, context=context) as f:
            return json.loads(f.read().decode("utf-8"))

    except Exception as e:
        raise OpenAIError(e)


def chatgpt_generated_code_for_assistant(
        prompt: str,
        api_key: str
):
    # Get the key for the ChatGPT API
    key = api_key

    # Enter the text input
    text = prompt

    # Create the request URL
    url = 'https://api.chatgpt.com/v1/query?key={}&query={}'.format(key, text)

    # Make the request to the ChatGPT API
    response = requests.get(url)

    # Load the response into a json object
    data = json.loads(response.text)

    # Print the response
    print(data['response'])


def model_list():
    try:
        print(openai.Model.list())
    except Exception as e:
        print(f"Error: {e}")


def commandline_params(argv):
    try:
        print(sys.argv[1:])
        opts, args = getopt.getopt(sys.argv[1:], 'hlq:m:', ['help', 'list-models', 'query=', 'model='])
    except getopt.GetoptError as ex:
        usage()
        sys.exit(2)

    query = None
    model = None
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit(2)
        elif opt in ('-l', '--list-models'):
            model_list()
            sys.exit(2)
        elif opt in ('-q', '--query'):
            query = arg
        elif opt in ('--m', '--model'):
            model = arg
        else:
            usage()
            sys.exit(2)

    return [query, model]


if __name__ == '__main__':
    text = """Summarize the following text in bullet points:
    ChatGPT sometimes writes plausible-sounding but incorrect or nonsensical answers. Fixing this issue is challenging, as: (1) during RL training, there’s currently no source of truth; (2) training the model to be more cautious causes it to decline questions that it can answer correctly; and (3) supervised training misleads the model because the ideal answer depends on what the model knows, rather than what the human demonstrator knows.
    ChatGPT is sensitive to tweaks to the input phrasing or attempting the same prompt multiple times. For example, given one phrasing of a question, the model can claim to not know the answer, but given a slight rephrase, can answer correctly.
    The model is often excessively verbose and overuses certain phrases, such as restating that it’s a language model trained by OpenAI. These issues arise from biases in the training data (trainers prefer longer answers that look more comprehensive) and well-known over-optimization issues.12
    Ideally, the model would ask clarifying questions when the user provided an ambiguous query. Instead, our current models usually guess what the user intended.
    While we’ve made efforts to make the model refuse inappropriate requests, it will sometimes respond to harmful instructions or exhibit biased behavior. We’re using the Moderation API to warn or block certain types of unsafe content, but we expect it to have some false negatives and positives for now. We’re eager to collect user feedback to aid our ongoing work to improve this system."""

    query, model = commandline_params(sys.argv[1:])
    query = query or text
    model = model or "text-curie-001"

    try:
        # model_list()
        response = ask_openai(
            prompt=query,
            api_key=openai.api_key,
            model=model
        )

        print("Raw response:")
        print(response)

        print("Choices:")
        suggestions = openai_choices(response)
        if suggestions:
            for suggestion in suggestions:
                print(suggestion)

    except OpenAIError as e:
        print(f"OpenAIError: {e}")
        exit(1)


__all__ = ("ask_openai", "commandline_params", "OpenAIError", "chatgpt_generated_code_for_assistant")