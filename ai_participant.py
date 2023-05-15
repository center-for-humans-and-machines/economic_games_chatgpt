# This script defines a class Ai Participant. When initialized and provided with a prompt,
# the Ai Participant sends a request to a ChatGPT model, obtains a response back and saves it
# in tabular data form.
# Sara Bonati - Center for Humans and Machines @ Max Planck Institute for Human Development
# ------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pandera as pa
import os
import json
import openai
import tiktoken
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
    def __init__(self, game: str, model: str, n: int):
        """
        Initialzes AIParticipant object
        :param game: the economic game
        :param model: the openai model to use
        :param n: the number of times a prompt is sent for each parameter combination
        """

        # load environment variables
        load_dotenv()
        openai.api_key = os.environ["OPENAI_API_KEY"]
        openai.api_base = os.environ["AZURE_ENDPOINT"]
        openai.api_type = "azure"
        openai.api_version = "2023-03-15-preview"  # this may change in the future

        # This will correspond to the custom name you chose for your deployment when you deployed a model.
        assert model in [
            "gpt-35-turbo",
            "text-davinci-003",
        ], f"model needs to be either gpt-35-turbo or text-davinci-003, got {model} instead"
        self.deployment_name = model

        # load game specific params
        self.game = game
        self.game_params = load_yaml(f"./params/{game}.yml")

        # define OpenAI model to use during experiment
        self.model_code = model

        # define number of times a prompt needs to be sent for each combination
        self.n = n

    @staticmethod
    def count_num_tokens(prompt: str, model_code: str):
        """
        This function counts the number of tokens a prompt is mad eof for different models.
        Adapted from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        :param prompt:
        :param model_code:
        :return:
        """

        model_dict = {"davinci": "text-davinci-003", "chatgpt": "gpt-3.5-turbo"}
        model = model_dict[model_code]
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        if model == "text-davinci-003":
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
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not presently implemented for model {model}.
    See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )

    @staticmethod
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def send_prompt(prompt: str, temp: float, model: str, game: str):
        """
        This function sends a prompt to a specific language model with
        specific parameters.
        :param prompt: the prompt text
        :param temp: the temperature model parameter value to use
        :param model: the type of openai model to use in the API request
        :param game: the economic game
        :return:
        """

        if model == "text-davinci-003":
            response = openai.Completion.create(
                engine=model,
                prompt=prompt,
                temperature=temp,
                max_tokens=50,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )

            prompt_response = response["choices"][0]["text"].rstrip("\n")
            return prompt_response

        elif model == "gpt-35-turbo":
            response = openai.ChatCompletion.create(
                engine=model,
                temperature=temp,
                max_tokens=50,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                messages=[{"role": "user", "content": prompt}],
            )
            prompt_response = response["choices"][0]["message"]["content"]
            # response returned as json
            prompt_response_dict = json.loads(prompt_response)
            if game == "ultimatum_receiver":
                prompt_response_value = prompt_response_dict["decision"]
            else:
                prompt_response_value = prompt_response_dict["amount_sent"]
            finish_reason = response["choices"][0]["message"]["finish_reason"]

        return prompt_response_dict["reply"], prompt_response_value, finish_reason

    # @staticmethod
    # def classify_answer(answer: str, game: str):
    #     """
    #     This function attempts to classify the language model response
    #     for a specific game prompt
    #
    #     :param answer: the answer received from the LM
    #     :param game: the game type we are examining
    #     :param amount: the amount of money that the question prompt used
    #     :return:
    #     """
    #
    #     if game == 'dictator_sender' or game == 'ultimatum_sender' or game == 'dictator_sequential':
    #         # how much money would you give, respond with number only
    #         give = [int(d) for d in re.findall(r'\d+', answer)]
    #         return give[0]
    #
    #     if game == 'ultimatum_receiver':
    #         # do you think this is fair, answer with yes or no
    #         sentences = answer.split("")
    #         return sentences[0]

    def adjust_prompts(self):
        """
        This function adjusts the prompts loaded from a csv file to include factors that
        may influence the AiParticipant response. Each factor is added to the prompt and the
        final prompt is saved in the same file in a new column

        :param prompt_file: (str) the name of the file containing prompts for the specific economic game
        :return:
        """

        # baseline prompts
        if self.game.endswith("_sender"):

            row_list_b = []
            for t in self.game_params["temperature"]:
                row = [self.game_params[f"prompt_baseline"]] + [t]
                row_list_b.append(row)
            baseline_df = pd.DataFrame(
                row_list_b, columns=list(["prompt", "temperature"])
            )

        else:

            row_list_b = []
            factors_b = self.game_params["factors_baseline_names"]
            factors_list_b = [self.game_params[f] for f in factors_b]
            all_comb_b = list(itertools.product(*factors_list_b))
            print(f"Number of combinations (baseline): {len(all_comb_b)}")

            for t in self.game_params["temperature"]:
                for comb in all_comb_b:
                    prompt_params_dict = {
                        factors_b[f]: comb[f] for f in range(len(factors_b))
                    }
                    final_prompt = self.game_params[f"prompt_baseline"].format(
                        **prompt_params_dict
                    )
                    row = (
                        [final_prompt]
                        + [t]
                        + [prompt_params_dict[f] for f in factors_b]
                    )
                    row_list_b.append(row)
            baseline_df = pd.DataFrame(
                row_list_b, columns=list(["prompt", "temperature"] + factors_b)
            )

        # for the baseline prompts these two factors will be filled later
        baseline_df["age"] = None
        baseline_df["gender"] = None
        # add prompt type
        baseline_df["prompt_type"] = "baseline"
        baseline_df["game_type"] = self.game

        # add number of tokens that the prompt corresponds to
        baseline_df["n_tokens_davinci"] = baseline_df["prompt"].apply(
            lambda x: self.count_num_tokens(x, "davinci")
        )
        baseline_df["n_tokens_chatgpt"] = baseline_df["prompt"].apply(
            lambda x: self.count_num_tokens(x, "chatgpt")
        )

        # experimental prompts
        row_list_e = []
        factors_e = self.game_params["factors_experimental_names"]
        factors_list_e = [self.game_params[f] for f in factors_e]
        all_comb_e = list(itertools.product(*factors_list_e))
        print(f"Number of combinations (experimental): {len(all_comb_e)}")

        for t in self.game_params["temperature"]:
            for comb in all_comb_e:
                prompt_params_dict = {
                    factors_e[f]: comb[f] for f in range(len(factors_e))
                }
                final_prompt = self.game_params[f"prompt_complete"].format(
                    **prompt_params_dict
                )
                row = [final_prompt] + [t] + [prompt_params_dict[f] for f in factors_e]
                row_list_e.append(row)

        experimental_df = pd.DataFrame(
            row_list_e, columns=list(["prompt", "temperature"] + factors_e)
        )
        experimental_df["prompt_type"] = "experimental"
        experimental_df["game_type"] = self.game
        # add number of tokens that the prompt corresponds to
        experimental_df["n_tokens_davinci"] = experimental_df["prompt"].apply(
            lambda x: self.count_num_tokens(x, "davinci")
        )
        experimental_df["n_tokens_chatgpt"] = experimental_df["prompt"].apply(
            lambda x: self.count_num_tokens(x, "chatgpt")
        )

        # pandera schema validation
        prompt_schema = pa.DataFrameSchema(
            {
                "prompt": pa.Column(str),
                "temperature": pa.Column(float, pa.Check.isin([0, 0.5, 1, 1.5])),
                "age": pa.Column(str),
                "gender": pa.Column(
                    str, pa.Check.isin(["female", "male", "non-binary"])
                ),
                "prompt_type": pa.Column(str),
                "game_type": pa.Column(str),
                "n_tokens_davinci": pa.Column(int, pa.Check.between(0, 1000)),
                "n_tokens_chatgpt": pa.Column(int, pa.Check.between(0, 1000)),
            }
        )

        try:
            prompt_schema.validate(baseline_df, lazy=True)
        except pa.errors.SchemaErrors as err:
            print("Schema errors and failure cases:")
            print(err.failure_cases)
            print("\nDataFrame object that failed validation:")
            print(err.data)
        try:
            prompt_schema.validate(experimental_df, lazy=True)
        except pa.errors.SchemaErrors as err:
            print("Schema errors and failure cases:")
            print(err.failure_cases)
            print("\nDataFrame object that failed validation:")
            print(err.data)

        # save both prompt types dataframes
        if not os.path.exists(os.path.join(prompts_dir, self.game)):
            os.makedirs(os.path.join(prompts_dir, self.game))
        baseline_df.to_csv(
            os.path.join(prompts_dir, self.game, f"{self.game}_baseline_prompts.csv")
        )
        experimental_df.to_csv(
            os.path.join(
                prompts_dir, self.game, f"{self.game}_experimental_prompts.csv"
            )
        )

    def collect_control_answers(self):
        """
        This method collects answers to control questions
        :return:
        """

        questions = self.game_params["control_questions"] * 10
        control_df = pd.DataFrame(questions, columns=["control_question"])
        control_df["control_answer"], control_df["finish_reason"] = control_df.apply(
            lambda x: self.send_prompt(
                control_df["prompt"], control_df["temperature"], "gpt-35-turbo"
            ),
            axis=1,
        )

        # pandera schema validation
        # define schema
        control_schema = pa.DataFrameSchema(
            {
                "control_question": pa.Column(str),
                "control_answer": pa.Column(
                    str,
                    checks=[
                        pa.Check.str_startswith("value_"),
                        # define custom checks as functions that take a series as input and
                        # outputs a boolean or boolean Series
                        pa.Check(lambda s: s.str.split("_", expand=True).shape[1] == 2),
                    ],
                ),
                "finish_reason": pa.Column(
                    str, pa.Check.isin(["stop", "length", "content_filter", "null"])
                ),
            }
        )

        try:
            control_schema.validate(control_df, lazy=True)
        except pa.errors.SchemaErrors as err:
            print("Schema errors and failure cases:")
            print(err.failure_cases)
            print("\nDataFrame object that failed validation:")
            print(err.data)

        # save control questions and answers
        control_df.to_csv(
            os.path.join(prompts_dir, self.game, f"{self.game}_control_questions.csv"),
            sep=",",
            index=False,
        )

    def collect_answers(self):
        """
        This function sends a request to an openai language model,
        collects the response and saves it in csv file
        :return:
        """
        assert os.path.exists(
            os.path.join(
                prompts_dir, self.game, f"{self.game}_intermediate_prompts.csv"
            )
        ), "Prompts file does not exist"

        prompt_df = pd.read_csv(
            os.path.join(
                prompts_dir, self.game, f"{self.game}_intermediate_prompts.csv"
            )
        )

        # retrieve answer text
        (
            prompt_df["answer_text"],
            prompt_df["answer"],
            prompt_df["finish_reason"],
        ) = prompt_df.apply(
            lambda x: self.send_prompt(
                prompt_df["prompt"],
                prompt_df["temperature"],
                self.model_code,
                self.game,
            ),
            axis=1,
        )

        # pandera schema validation
        final_prompt_schema = pa.DataFrameSchema(
            {
                "prompt": pa.Column(str),
                "temperature": pa.Column(float, pa.Check.isin([0, 0.5, 1, 1.5])),
                "age": pa.Column(str),
                "gender": pa.Column(
                    str, pa.Check.isin(["female", "male", "non-binary"])
                ),
                "prompt_type": pa.Column(str),
                "game_type": pa.Column(str),
                "n_tokens_davinci": pa.Column(int, pa.Check.between(0, 1000)),
                "n_tokens_chatgpt": pa.Column(int, pa.Check.between(0, 1000)),
                "answer_text": pa.Column(str),
                "answer": pa.Column(int, pa.Check.between(0, 10)),
                "finish_reason": pa.Column(
                    str, pa.Check.isin(["stop", "content_filter", "null", "length"])
                ),
            }
        )

        try:
            final_prompt_schema.validate(prompt_df, lazy=True)
        except pa.errors.SchemaErrors as err:
            print("Schema errors and failure cases:")
            print(err.failure_cases)
            print("\nDataFrame object that failed validation:")
            print(err.data)

        prompt_df.to_csv(os.path.join(prompts_dir, f"{self.game}_data.csv"))


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description="""
                                                 Run ChatGPT AI participant for economic games
                                                 (Center for Humans and Machines, 
                                                 Max Planck Institute for Human Development)
                                                 """
    )
    parser.add_argument(
        "--game",
        type=str,
        help="""
                                Which economic game do you want to run \n 
                                (options: dictator_sender, dictator_sequential, ultimatum_sender, ultimatum_receiver)
                                """,
        required=True,
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="""
                                What type of action do you want to run \n 
                                (options: test_assumptions, 
                                          prepare_prompts,
                                          send_prompts_baseline, 
                                          send_prompts_experimental)
                                """,
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="""
                                Which model do you want to use \n 
                                (options: gpt-35-turbo, text-davinci-003)
                                """,
        required=True,
    )

    # collect args and make assert tests
    args = parser.parse_args()
    general_params = load_yaml("params/params.yml")
    assert (
        args.game in general_params["game_options"]
    ), f'Game option must be one of {general_params["game_options"]}'
    assert (
        args.mode in general_params["mode_options"]
    ), f'Mode option must be one of {general_params["mode_options"]}'
    assert (
        args.model in general_params["model_options"]
    ), f'Model option must be one of {general_params["model_options"]}'

    # directory structure
    prompts_dir = "../prompts"

    # initialize AiParticipant object
    P = AiParticipant(args.game, args.model, general_params["n"])

    # if args.mode == 'test_assumptions':
    #     P.test_assumptions(general_params["game_assumption_filename"][args.game])

    if args.mode == "prepare_prompts":
        P.adjust_prompts()

    if args.mode == "send_prompts_baseline":
        print("work in progress")
        # P.collect_answers()
