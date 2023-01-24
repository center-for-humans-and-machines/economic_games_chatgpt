# This script defines a class Ai Participant. When initialized and provided with a prompt,
# the Ai Participant sends a request to a ChatGPT model, obtains a response back and saves it
# in tabular data form.

import numpy as np
import pandas as pd
import os
import openai
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
from sklearn.model_selection import ParameterGrid


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
        self.game_params = load_yaml(f"./params/{game}_params.yml")


        # TODO: integrate parameter grid
        self.parameter_grid = {'temperature': [0.7, 0.8, 0.9],
                               'max_tokens': [256],
                               'top_p': [1],
                               'frequency_penalty': [0],
                               'presence_penalty': [0]
                               }

        #for g in ParameterGrid(self.parameter_grid):
        #    print(g)

    @staticmethod
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def send_prompt(prompt):
        """
        This function sends a prompt to a specific language model with
        specific parameters.
        TODO: make language model and model parameters customizable in the future?
        :return:
        """

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        prompt_response = response["choices"][0]["text"].rstrip("\n")
        return prompt_response

    @staticmethod
    def classify_answer(answer: str, game: str, stake=None):
        """
        This function attempts to classify the language model response
        for a specific game prompt
        TODO: so far only ultimatum, add other economic games
        :param answer:
        :param game:
        :param stake:
        :return:
        """

        if game == "ultimatum":
            # extract the X amount of euros that the language model would keep for itself
            # (from the first phrase of the answer text)
            sentences = answer.split(".")
            keep_pattern = re.compile("keep.*\d+")
            give_pattern = re.compile("give.*\d+")
            amount_give = [int(d) for d in re.findall(r'\d+', give_pattern.findall(sentences[0])[0])]
            if len(keep_pattern.findall(sentences[0])) > 0:
                amount_keep = [int(d) for d in re.findall(r'\d+', keep_pattern.findall(sentences[0])[0])]

            if amount_give is None and amount_keep is not None:
                return stake - amount_keep[0]
            elif amount_give is not None:
                return amount_give[0]

    def adjust_prompts(self, prompt_file: str):
        """
        This function adjusts the prompts loaded from a csv file to include factors that
        may influence the AiParticipant response. Each factor is added to the prompt and the
        final prompt is saved in the same file in a new column

        :param prompt_file: (str) the name of the file containing prompts for the specific economic game
        :return:
        """

        row_list = []
        factors = self.game_params["factors_names"]
        factors_list = [self.game_params[f] for f in factors]
        all_comb = list(itertools.product(*factors_list))
        print(f"Number of combinations: {len(all_comb)}")

        for comb in all_comb:
            prompt_params_dict = {factors[f]: comb[f] for f in range(len(factors))}
            final_prompt = self.game_params["prompt_complete"].format(**prompt_params_dict)
            row = [final_prompt] + [prompt_params_dict[f] for f in factors]
            row_list.append(row)

        self.prompt_df = pd.DataFrame(row_list, columns=list(["prompt"]+factors))

    def test_assumptions(self, prompt_file: str):
        """
        This function sends prompts related to an economic game and returns the rules and strategies
        of the game as the language model assumes it goes

        :param prompt_file: (str) the name of the file containing assumption prompts for the specific economic game
        :return:
        """

        sanity_row_list = []
        for q in self.game_params["sanity_questions"]:
            sanity_row_list.append(self.game_params["prompt_sanity_check"]+' '+q)

        self.assumption_prompt_df = pd.DataFrame(sanity_row_list, columns=["sanity_check_text"])
        self.assumption_prompt_df["answer_text"] = self.assumption_prompt_df["sanity_check_text"].apply(self.send_prompt)
        self.assumption_prompt_df.to_csv(os.path.join(prompts_dir, f"{self.game}sanity_check_answers.csv"))

    def collect_answers(self):
        """
        This function sends a request to an openai language model,
        collects the response and saves it in csv file
        :return:
        """
        # retrieve answer text
        self.prompt_df["answer_text"] = self.prompt_df["prompt"].apply(self.send_prompt)
        # classification of answer
        if self.game == "ultimatum":
            self.prompt_df['money_give'] = self.prompt_df.apply(lambda x: self.classify_answer(x['answer_text'],
                                                                                               self.game,
                                                                                               x['stake']), axis=1)
            self.prompt_df['is_fair'] = np.where(self.prompt_df['money_give'] == (self.prompt_df['stake']/2),
                                                 True,
                                                 False)

        self.prompt_df.to_csv(os.path.join(prompts_dir, f"{self.game}_final2.csv"))


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
        P.adjust_prompts(general_params["game_assumption_filename"][args.game])
        P.collect_answers()
