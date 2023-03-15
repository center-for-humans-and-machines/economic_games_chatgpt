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
* `dictator_receiver` (receiver perspective)

The action to perform can be one of the following options:

* `test_assumptions` (this action sends prompts that test the language model's knowledge of the selected game and checks for the assumptions that the language model has on the game dynamics)
* `prepare_prompts` (this action prepares all prompts in a csv file or dataframe)
* `send_prompts_baseline` (this action sends prompts for a specific game, in baseline condition)
* `send_prompts_experimental` (this action sends prompts for a specific game, in experimental condition)

The language model type can be one of the following options:

* `davinci` (refers to the text-davinci clss of language models, in particular this option stands for `text-davinci-003`)
* `chatgpt` (new! refers to the recently released `gpt-3.5-turbo` language model)

E.g. run the script on the trust game from the sender perspective, send all prompts to the language model and save responses
```
(.venv) python ai_participant.py --game trust_sender --mode send_prompts
```

## Factor Levels for prompts

For each game a parameter file in the `params` folder of the repository is created.
The parameter file contains all prompts used in the study, organized by different languages and by prompt type (baseline or experimental).
Additionally, the parameter file includes all factors and language model parameters that are used in the study.
Parameters used include: (work in progress)

