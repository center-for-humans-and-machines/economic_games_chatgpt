# This script defines a class Ai Participant. When initialized and provided with a prompt,
# the Ai Participant sends a request to a ChatGPT model, obtains a response back and saves it
# in tabular data form.
# Sara Bonati - Center for Humans and Machines @ Max Planck Institute for Human Development
# ------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import os
import openai
import tiktoken
import requests
import itertools
from dotenv import load_dotenv
import argparse
import yaml
import re
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


def load_yaml(filename):
    with open(filename) as f:
        data = yaml.safe_load(f)
    return data


class AiParticipant:

    def __init__(self, game: str, model: str):
        """
        Initialzes AIParticipant object
        :param game: the economic game
        :param model: the openai model to use
        """

        # load environment variables
        load_dotenv()
        openai.api_key = os.environ["OPENAI_API_KEY"]

        # load game specific params
        self.game = game
        self.game_params = load_yaml(f"./params/{game}.yml")

        # define OpenAI model to use during experiment
        self.model_code = model

    @staticmethod
    def count_num_tokens(prompt: str, model_code: str):
        """
        This function counts the number of tokens a prompt is mad eof for different models.
        Adapted from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        :param prompt:
        :param model_code:
        :return:
        """

        model_dict = {'davinci': 'text-davinci-003', 'chatgpt': 'gpt-3.5-turbo'}
        model = model_dict[model_code]
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        if model == 'text-davinci-003':
            num_tokens = len(encoding.encode(prompt))
            return num_tokens

        if model == "gpt-3.5-turbo":
            num_tokens = 0
            messages = [{"role": "user", "content": prompt}]
            for message in messages:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens += -1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens
        else:
            raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
    See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

    @staticmethod
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def send_prompt(prompt: str, temp: float, model: str):
        """
        This function sends a prompt to a specific language model with
        specific parameters.
        :param prompt: the prompt text
        :param temp: the temperature model parameter value to use
        :param model: the type of openai model to use in the API request
        :return:
        """

        if model == 'davinci':
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=temp,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            prompt_response = response["choices"][0]["text"].rstrip("\n")

        elif model == 'chatgpt':
            response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
            )
            prompt_response = response["choices"]["message"]["content"]

        return prompt_response

    @staticmethod
    def classify_answer(answer: str, game: str, amount=None):
        """
        This function attempts to classify the language model response
        for a specific game prompt

        :param answer: the answer received from the LM
        :param game: the game type we are examining
        :param amount: the amount of money that the question prompt used
        :return:
        """

        if game == 'dictator_sender' or game == 'ultimatum_sender' or game == 'dictator_sequential':
            # how much money would you give, respond with number only
            give = [int(d) for d in re.findall(r'\d+', answer)]
            return give[0]

        if game == 'dictator_receiver' or game == 'ultimatum_receiver':
            # do you think this is fair, answer with yes or no
            sentences = answer.split("")
            return sentences[0]

        # if game == "trust_sender":
        #     # extract the X amount of euros that the language model would send to other person
        #     sentences = answer.split("")
        #     give_pattern = re.compile("give.*\d+")
        #     amount_give = [int(d) for d in re.findall(r'\d+', give_pattern.findall(sentences[0])[0])]
        #     if amount_give is not None:
        #         return amount_give[0]

        # if game == "ultimatum_sender":
        #     # extract the X amount of euros that the language model would give/keep for itself
        #     # (from the first phrase of the answer text)
        #     sentences = answer.split(".")
        #     keep_pattern = re.compile("keep.*\d+")
        #     give_pattern = re.compile("give.*\d+")
        #     amount_give = [int(d) for d in re.findall(r'\d+', give_pattern.findall(sentences[0])[0])]
        #     if len(keep_pattern.findall(sentences[0])) > 0:
        #         amount_keep = [int(d) for d in re.findall(r'\d+', keep_pattern.findall(sentences[0])[0])]
        #
        #     if amount_give is None and amount_keep is not None:
        #         return amount - amount_keep[0]
        #     elif amount_give is not None:
        #         return amount_give[0]

    def adjust_prompts(self):
        """
        This function adjusts the prompts loaded from a csv file to include factors that
        may influence the AiParticipant response. Each factor is added to the prompt and the
        final prompt is saved in the same file in a new column

        :param prompt_file: (str) the name of the file containing prompts for the specific economic game
        :return:
        """

        # baseline prompts
        row_list_b = []
        factors_b = self.game_params["factors_baseline_names"]
        factors_list_b = [self.game_params[f] for f in factors_b]
        all_comb_b = list(itertools.product(*factors_list_b))
        print(f"Number of combinations (baseline): {len(all_comb_b)}")

        for t in self.game_params['temperature_baseline']:
            for l in self.game_params['language']:
                for comb in all_comb_b:
                    prompt_params_dict = {factors_b[f]: comb[f] for f in range(len(factors_b))}
                    final_prompt = self.game_params[f"prompt_baseline_{l}"].format(**prompt_params_dict)
                    row = [final_prompt] + [t] + [l] + [prompt_params_dict[f] for f in factors_b]
                    row_list_b.append(row)
        baseline_df = pd.DataFrame(row_list_b, columns=list(["prompt", "temperature", "language"] + factors_b))

        # for the baseline prompts these two factors will be filled later
        baseline_df['age'] = None
        baseline_df['gender'] = None
        # add prompt type
        baseline_df['prompt_type'] = 'baseline'
        # add number of tokens that the prompt corresponds to
        baseline_df['n_tokens_davinci'] = baseline_df['prompt'].apply(lambda x: self.count_num_tokens(x,'davinci'))
        baseline_df['n_tokens_chatgpt'] = baseline_df['prompt'].apply(lambda x: self.count_num_tokens(x,'chatgpt'))

        # # experimental prompts
        # row_list_e = []
        # factors_e = self.game_params["factors_experimental_names"]
        # factors_list_e = [self.game_params[f] for f in factors_e]
        # all_comb_e = list(itertools.product(*factors_list_e))
        # print(f"Number of combinations (experimental): {len(all_comb_e)}")
        #
        # for t in self.game_params['temperature']:
        #     for l in self.game_params['language']:
        #         for comb in all_comb_e:
        #             prompt_params_dict = {factors_e[f]: comb[f] for f in range(len(factors_e))}
        #             final_prompt = self.game_params[f"prompt_complete_{l}"].format(**prompt_params_dict)
        #             row = [final_prompt] + [t] + [l] + [prompt_params_dict[f] for f in factors_e]
        #             row_list_e.append(row)
        #
        # experimental_df = pd.DataFrame(row_list_e, columns=list(["prompt", "temperature", "language"] + factors_e))
        # experimental_df['prompt_type'] = 'experimental'
        #
        # # create prompt df with all combinations for baseline and experimental
        # self.prompt_df = pd.concat([baseline_df, experimental_df], axis=0)
        self.prompt_df = baseline_df
        self.prompt_df['game_type'] = self.game

        # save intermediate prompts
        if not os.path.exists(os.path.join(prompts_dir, self.game)):
            os.makedirs(os.path.join(prompts_dir, self.game))
        self.prompt_df.to_csv(os.path.join(prompts_dir, self.game, f'{self.game}_intermediate_prompts.csv'))

    def test_assumptions(self):
        """
        This function sends prompts related to an economic game and returns the rules and strategies
        of the game as the language model assumes it goes
        :return:
        """

        sanity_row_list = []
        for q in self.game_params["sanity_questions"]:
            sanity_row_list.append(self.game_params["prompt_baseline_EN"] + ' ' + q)

        self.assumption_prompt_df = pd.DataFrame(sanity_row_list, columns=["sanity_check_text"])
        self.assumption_prompt_df["answer_text"] = self.assumption_prompt_df["sanity_check_text"].apply(
            self.send_prompt)
        self.assumption_prompt_df.to_csv(os.path.join(prompts_dir, f"{self.game}_sanity_check_answers.csv"))

    @staticmethod
    def build_followup_baseline_prompt(prompt, answer, language, game_params):
        return prompt + '\n' + answer + '\n' + game_params[f'prompt_baseline_followup_{language}']

    def obtain_baseline_info(self):
        """
        This method sends an additional prompt to the language model for baseline condition.
        The prompt is made of the previous model answer + a follow-up question to obtain
        demographic info.
        :return:
        """

        # create new columns for baseline prompts only
        self.prompt_df['baseline_followup'] = None
        self.prompt_df['baseline_followup'] = self.prompt_df[self.prompt_df['prompt_type'] == 'baseline'].apply(
            lambda x:
            self.build_followup_baseline_prompt(self.prompt_df["prompt"],
                                                self.prompt_df["answer_text"],
                                                self.prompt_df["language"],
                                                self.game_params),
            axis=1)

        # obtain answer (language is already set, we specify the temperature parameter)
        self.prompt_df['baseline_followup_answer'] = None
        self.prompt_df['baseline_followup_answer'] = self.prompt_df[self.prompt_df['prompt_type'] == 'baseline'].apply(
            lambda x: self.send_prompt(self.prompt_df["baseline_followup"],
                                       self.prompt_df["temperature"]),
            axis=1)

        # convert column to string
        self.prompt_df['baseline_followup_answer'] = self.prompt_df['baseline_followup_answer'].astype(str)

        # separate gender and age phrases into two temporary columns
        self.prompt_df[['temp_gender', 'temp_age']] = self.prompt_df[self.prompt_df['prompt_type'] == 'baseline'][
            'baseline_followup_answer'].str.split(pat='.',
                                                  expand=True)

        # extract gender
        self.prompt_df.loc[self.prompt_df['prompt_type'] == 'baseline', 'gender'] = \
            self.prompt_df[self.prompt_df['prompt_type'] == 'baseline']['temp_gender'].str.partition('is ')[
                2].str.extract(
                r"([^0-9]+)", expand=False)
        # extract age
        self.prompt_df.loc[self.prompt_df['prompt_type'] == 'baseline', 'age'] = \
            self.prompt_df[self.prompt_df['prompt_type'] == 'baseline']['temp_age'].str.extract(
                r"(\d{1,3})", expand=False).astype(int)

        # remove temp columns
        self.prompt_df.drop(['temp_age', 'temp_gender'], axis=1, inplace=True)

    def collect_answers(self):
        """
        This function sends a request to an openai language model,
        collects the response and saves it in csv file
        :return:
        """
        assert os.path.exists(os.path.join(prompts_dir, self.game, f'{self.game}_intermediate_prompts.csv')), \
            'Prompts file does not exist'

        self.prompt_df = pd.read_csv(os.path.join(prompts_dir, self.game, f'{self.game}_intermediate_prompts.csv'))

        # retrieve answer text
        self.prompt_df["answer_text"] = self.prompt_df.apply(lambda x: self.send_prompt(self.prompt_df["prompt"],
                                                                                        self.prompt_df["temperature"],
                                                                                        self.model_code),
                                                             axis=1)

        # for baseline prompts: obtain demographic info after answer
        self.obtain_baseline_info(self.prompt_df[self.prompt_df['prompt_type'] == 'baseline'])

        # classification of answer
        if self.game == "ultimatum_sender":
            self.prompt_df['money_give'] = self.prompt_df.apply(lambda x: self.classify_answer(x['answer_text'],
                                                                                               self.game,
                                                                                               x['stake']), axis=1)
            self.prompt_df['is_fair'] = np.where(self.prompt_df['money_give'] == (self.prompt_df['stake'] / 2),
                                                 True,
                                                 False)

        self.prompt_df.iloc[:, 1:].to_csv(os.path.join(prompts_dir, f"{self.game}_final2.csv"))


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="""
                                                 Run ChatGPT AI participant for economic games
                                                 (Center for Humans and Machines, 
                                                 Max Planck Institute for Human Development)
                                                 """)
    parser.add_argument('--game',
                        type=str,
                        help="""
                                Which economic game do you want to run \n 
                                (options: dictator_sequential, dictator_sender, dictator_receiver, 
                                ultimatum_sender, ultimatum_receiver)
                                """,
                        required=True)
    parser.add_argument('--mode',
                        type=str,
                        help="""
                                What type of action do you want to run \n 
                                (options: test_assumptions, 
                                          prepare_prompts,
                                          send_prompts_baseline, 
                                          send_prompts_experimental)
                                """,
                        required=True)
    parser.add_argument('--model',
                        type=str,
                        help="""
                                Which model do you want to use \n 
                                (options: chatgpt (stands for gpt-3.5-turbo), davinci (stands for text-davinci-003))
                                """,
                        required=True)

    # collect args and make assert tests
    args = parser.parse_args()
    general_params = load_yaml("params/params.yml")
    assert args.game in general_params["game_options"], f'Game option must be one of {general_params["game_options"]}'
    assert args.mode in general_params["mode_options"], f'Mode option must be one of {general_params["mode_options"]}'
    assert args.model in general_params["model_options"], f'Model option must be one of {general_params["model_options"]}'

    # directory structure
    prompts_dir = '../prompts'

    # initialize AiParticipant object
    P = AiParticipant(args.game, args.model)

    # if args.mode == 'test_assumptions':
    #     P.test_assumptions(general_params["game_assumption_filename"][args.game])

    if args.mode == 'prepare_prompts':
        #print('Work in progress')
        P.adjust_prompts()

    if args.mode == 'send_prompts_baseline':
        print('work in progress')
        # P.collect_answers()
