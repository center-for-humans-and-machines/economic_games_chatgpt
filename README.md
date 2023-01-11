# economic_games_chatgpt

## Setting up the repository

We use Python version 3.10.8 to create a virtual environment to be used in this project

```
python3.10 -m venv .venv
source .venv/bin/activate # (for mac/linux users)
# source .venv/Scripts/Activate (for windows users)

(.venv) pip install --upgrade pip
(.venv) pip install wheel
(.venv) pip install -r requirements.txt
```

## Sending prompts to ChatGPT

The OpenAI API makes it possible to send a request to ChatGPT language models and obtain a response back. In order to 
use the API an OpenAI account is necessary. An API key is also needed in order to use the API (more info here: https://beta.openai.com/docs/api-reference/authentication).
Once you have an OpenAI account and have obtained an API key associated to the account create a `.env` file in the root folder
of the repository 

```
OPENAI_API_KEY=<your_api_key_here>
```