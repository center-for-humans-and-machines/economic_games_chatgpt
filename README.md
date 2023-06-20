# economic_games_chatgpt
![example workflow](https://github.com/SaraBonati/economic_games_chatgpt/actions/workflows/black.yml/badge.svg)
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

## Using ChatGPT with the OpenAI API

The OpenAI API makes it possible to send a request to OpenAI language models and obtain a response back. In order to 
use the API an OpenAI account is necessary. An API key is also needed in order to use the API (more info here: https://beta.openai.com/docs/api-reference/authentication).
Once you have an OpenAI account and have obtained an API key associated to the account create a `.env` file in the root folder
of the repository 

```
OPENAI_API_KEY=<your_api_key_here>
```

## Running the analysis script

In order to run the analysis script `ai_participant.py` specify the economic game, the action you want to perform and the type
of language model you want to use.
Economic game can be one of the following options:

* `ultimatum_sender` (sender perspective)
* `ultimatum_receiver` (receiver perspective)
* `dictator_sequential`
* `dictator_sender` (sender perspective)
* `dictator_binary` 

The action to perform can be one of the following options:

* `test_assumptions` (this action sends prompts that test the language model's knowledge of the selected game and checks for the assumptions that the language model has on the game dynamics)
* `prepare_prompts` (this action prepares all prompts in a csv file or dataframe)
* `convert_prompts_to_json` (this action prepares all prompts from csv file to jsonl file, relevant for using EC2)
* `send_prompts_baseline` (this action sends prompts for a specific game, in baseline condition)
* `send_prompts_experimental` (this action sends prompts for a specific game, in experimental condition)
* `convert_jsonl_results_to_csv` (this action prepares all results from jsonl file to csv file, relevant for using EC2)

The language model type can be one of the following options:

* `text-davinci-003` (refers to the text-davinci class of language models, in particular this option stands for `text-davinci-003`)
* `gpt-35-turbo` (refers to the `gpt-3.5-turbo` language model)

E.g. run the script on the trust game from the sender perspective, send all prompts to the language model and save responses
```
(.venv) python ai_participant.py --game trust_sender --mode send_prompts --model gpt-35-turbo
```

### Sending prompts on AWS EC2
Calling the API from the script can be a slow process; for this study we try to make the process faster by running
the code on an AWS EC2 instance located in the US (where the Azure OpenAI model "resides"). We also adapt a parallel processor script
provided by OpenAI (see https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py) that
allows parallel api requests using async.

Note that using this method requires data to be organized in a jsonl file, where each line contains the prompt to send
and metadata such as temperature or frequency penalty.

To run the script `api_helper.py` on EC2 navigate to your project directory and execute the following command on the EC2 terminal (the language model used
by default is `gpt-35-turbo`, and we explicitly say whether we want to collect the baseline or experimental prompts)
```
(.venv) python3 api_helper.py --game dictator_sender --exptype baseline
```

## Factor Levels for prompts

For each game a parameter file in the `params` folder of the repository is created.
The parameter file contains all prompts used in the study, organized by different languages and by prompt type (baseline or experimental).
Additionally, the parameter file includes all factors and language model parameters that are used in the study.
Parameters used include: (work in progress)

