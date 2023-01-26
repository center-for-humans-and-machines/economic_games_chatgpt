# This script defines a class Ai Participant. When initialized and provided with a prompt,
# the Ai Participant sends a request to a ChatGPT model, obtains a response back and saves it
# in tabular data form.
# Sara Bonati - Center for Humans and Machines @ Max Planck Institute for Human Development
# ------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import os
import openai
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

    def __init__(self, game: str):
        """
        Initializes AiParticipant object
        """
        # assert tests
        assert game in general_params["game_options"], \
            f'Game option not valid, must be one of {general_params["game_options"]}'

        # load environment variables
        load_dotenv()
        openai.api_key = os.environ["OPENAI_API_KEY"]

        # load game specific params
        self.game = game
        self.game_params = load_yaml(f"./params/{game}.yml")

    @staticmethod
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def send_prompt(prompt: str, temp: float):
        """
        This function sends a prompt to a specific language model with
        specific parameters.
        :param prompt: the propt text
        :param temp: the temperature model parameter value to use
        :return:
        """

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
        return prompt_response

    @staticmethod
    def classify_answer(answer: str, game: str, amount=None):
        """
        This function attempts to classify the language model response
        for a specific game prompt
        TODO: so far only ultimatum, add other economic games
        :param answer: the answer received from the LM
        :param game: the game type we are examining
        :param amount: the amount of money that the question prompt used
        :return:
        """

        if game == "trust_receiver":
            # Yes or No type of answer
            sentences = answer.split("")
            return sentences[0]

        if game == "trust_sender":
            # extract the X amount of euros that the language model would send to other person
            sentences = answer.split("")
            give_pattern = re.compile("give.*\d+")
            amount_give = [int(d) for d in re.findall(r'\d+', give_pattern.findall(sentences[0])[0])]
            if amount_give is not None:
                return amount_give[0]

        if game == "ultimatum_sender":
            # extract the X amount of euros that the language model would give/keep for itself
            # (from the first phrase of the answer text)
            sentences = answer.split(".")
            keep_pattern = re.compile("keep.*\d+")
            give_pattern = re.compile("give.*\d+")
            amount_give = [int(d) for d in re.findall(r'\d+', give_pattern.findall(sentences[0])[0])]
            if len(keep_pattern.findall(sentences[0])) > 0:
                amount_keep = [int(d) for d in re.findall(r'\d+', keep_pattern.findall(sentences[0])[0])]

            if amount_give is None and amount_keep is not None:
                return amount - amount_keep[0]
            elif amount_give is not None:
                return amount_give[0]

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
        for t in self.game_params['temperature']:
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
        baseline_df['prompt_type'] = 'baseline'

        # experimental prompts
        row_list_e = []
        factors_e = self.game_params["factors_experimental_names"]
        factors_list_e = [self.game_params[f] for f in factors_e]
        all_comb_e = list(itertools.product(*factors_list_e))
        print(f"Number of combinations (experimental): {len(all_comb_e)}")
        for t in self.game_params['temperature']:
            for l in self.game_params['language']:
                for comb in all_comb_e:
                    prompt_params_dict = {factors_e[f]: comb[f] for f in range(len(factors_e))}
                    final_prompt = self.game_params[f"prompt_complete_{l}"].format(**prompt_params_dict)
                    row = [final_prompt] + [t] + [l] + [prompt_params_dict[f] for f in factors_e]
                    row_list_e.append(row)

        experimental_df = pd.DataFrame(row_list_e, columns=list(["prompt", "temperature", "language"] + factors_e))
        experimental_df['prompt_type'] = 'experimental'

        # create prompt df with all combinations for baseline and experimental
        self.prompt_df = pd.concat([baseline_df, experimental_df], axis=0)
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
            sanity_row_list.append(self.game_params["prompt_baseline"] + ' ' + q)

        self.assumption_prompt_df = pd.DataFrame(sanity_row_list, columns=["sanity_check_text"])
        self.assumption_prompt_df["answer_text"] = self.assumption_prompt_df["sanity_check_text"].apply(
            self.send_prompt)
        self.assumption_prompt_df.to_csv(os.path.join(prompts_dir, f"{self.game}_sanity_check_answers.csv"))

    def collect_answers(self):
        """
        This function sends a request to an openai language model,
        collects the response and saves it in csv file
        :return:
        """
        # retrieve answer text
        self.prompt_df["answer_text"] = self.prompt_df.apply(lambda x: self.send_prompt(self.prompt_df["prompt"],
                                                                                        self.prompt_df["temperature"]),
                                                             axis=1)

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
    parser = argparse.ArgumentParser(description='Run ChatGPT AI participant for economic games')
    parser.add_argument('--game',
                        type=str,
                        help='Which economic game do you want to run',
                        required=True)
    parser.add_argument('--mode',
                        type=str,
                        help='What type of action do you want to run',
                        required=True)

    # collect args and make assert tests
    args = parser.parse_args()
    general_params = load_yaml("params/params.yml")
    assert args.game in general_params["game_options"], f'Game option must be one of {general_params["game_options"]}'
    assert args.mode in general_params["mode_options"], f'Mode option must be one of {general_params["mode_options"]}'

    # directory structure
    prompts_dir = '../prompts'

    # initialize AiParticipant object
    P = AiParticipant(args.game)

    # if args.mode == 'test_assumptions':
    #     P.test_assumptions(general_params["game_assumption_filename"][args.game])

    if args.mode == 'send_prompts':
        P.adjust_prompts()
        # P.collect_answers()
