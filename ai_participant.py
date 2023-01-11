# This script defines a class Ai Participant. When initialized and provided with a prompt,
# the Ai Participant sends a request to a ChatGPT model, obtains a response back and saves it
# in tabular data form.

import pandas as pd
import os
import openai
import itertools
from dotenv import load_dotenv


class AiParticipant:

    def __init__(self):
        """
        Initializes AiParticipant object
        """
        # load environment variables
        load_dotenv()
        openai.api_key = os.environ["OPENAI_API_KEY"]

        # load generic prompts
        self.prompt_df = pd.read_excel("../prompts/prompt_test.xlsx")
        print(list(self.prompt_df.columns))

    @staticmethod
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

        return response["choices"][0]["text"]

    def adjust_prompts(self):
        """
        This function adjusts the prompts loaded from a csv file to include factors that
        may influence the AiParticipant response. Each factor is added to the prompt and the
        final prompt is saved in the same file in a new column
        :return:
        """

        self.person_type_options = ["stranger", "friend", "romantic partner"]
        self.behaviour_type_options = ["selfish", "generous"]

        all_comb = list(itertools.product(self.person_type_options,
                                          self.behaviour_type_options))
        baseline_prompts = list(self.prompt_df['prompt_text'].unique())

        # create elaborate prompts dataframe (created from a list of dicts)
        rows_list = []

        for baseline_prompt in baseline_prompts:
            for comb in all_comb:

                prompt_params_dict = {
                    'person_type': comb[0],
                    'behavior_type': comb[1]
                }
                final_prompt = baseline_prompt.format(**prompt_params_dict)

                rows_list.append([
                    baseline_prompt,
                    prompt_params_dict["person_type"],
                    prompt_params_dict["behavior_type"],
                    final_prompt,
                ])

        add_df = pd.DataFrame(rows_list, columns = list(self.prompt_df.columns))
        self.prompt_df = pd.concat([self.prompt_df, add_df], ignore_index=True)
        self.prompt_df.to_csv("../prompts/prompt_test_intermediate.csv")

    def collect_answers(self):
        """
        This function sends a request to an openai language model,
        collects the response and saves it in csv file
        :return:
        """
        self.prompt_df["answer_text"] = self.prompt_df["final_prompt_text"].apply(self.send_prompt)
        self.prompt_df.to_csv("../prompts/prompt_test_final.csv")


if __name__ == "__main__":

    P = AiParticipant()
    P.adjust_prompts()
    P.collect_answers()
