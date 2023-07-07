import re
import os
import json


def keys_exists(element, *keys):
    """
    Check if *keys (nested) exists in `element` (dict).
    """
    if not isinstance(element, dict):
        raise AttributeError("keys_exists() expects dict as first argument.")
    if len(keys) == 0:
        raise AttributeError("keys_exists() expects at least two arguments, one given.")

    _element = element
    print(_element)
    print(keys)
    for key in keys:
        try:
            _element = _element[key]
        except KeyError:
            return False
    return True


def extract_response(json_line: dict, game: str, exptype: str):
    """

    :param json_line:
    :param game:
    :param exptype:
    :return:
    """

    exception_list = []
    print(json_line[1])

    try:
        # parse as json the model response
        response_dict = json.loads(json_line[1]["choices"][0]["message"]["content"])

        # when experimental then collect age and gender from prompt message
        if exptype == "experimental":
            message = json_line[0]["messages"][0]["content"]
            demographic_phrase = message.split(". ")[1]
            demographic_words = demographic_phrase.split(" ")
            # collect demographic info of each prompt
            age = demographic_words[2]
            gender = demographic_words[-1]
        else:
            age = "unprompted"
            gender = "unprompted"

        # check if we have one dict or nested dicts
        # e.g. some answers are {'reply': 'Text', 'amount_sent':'3'}
        # others are nested in the form of {'answer': {'reply': 'Text', 'amount_sent':'3'}}
        # or {'reply': 'Text', {'answer': {'amount_sent':'3'}}

        if not any(isinstance(i, dict) for i in response_dict.values()):
            if (
                game == "dictator_sender"
                or game == "ultimatum_sender"
                or game == "dictator_sequential"
            ):
                reply = (
                    response_dict[
                        f'{[i for i in list(response_dict.keys()) if "re" in i or "Re" in i][0]}'
                    ],
                )
                value = response_dict[
                    f'{[i for i in list(response_dict.keys()) if "amount" in i or "money" in i or "sent" in i or "Sent" in i][0]}'
                ]

                if game == "dictator_sequential":
                    if exptype == "baseline":
                        fairness = re.findall(
                            r"\d+", json_line[0]["messages"][0]["content"].split(".")[5]
                        )
                    else:
                        fairness = re.findall(
                            r"\d+", json_line[0]["messages"][0]["content"].split(".")[6]
                        )

                    return [
                        json_line[0]["messages"][0]["content"],  # the full prompt
                        json_line[0]["temperature"],  # temperature
                        int(fairness[0]),  # fairness level
                        age,  # age
                        gender,  # gender
                        exptype,  # prompt_type
                        game,  # game type,
                        reply,
                        value,
                        json_line[1]["choices"][0]["finish_reason"],  # finish reason
                    ]

                else:
                    return [
                        json_line[0]["messages"][0]["content"],  # the full prompt
                        json_line[0]["temperature"],  # temperature
                        age,  # age
                        gender,  # gender
                        exptype,  # prompt_type
                        game,  # game type,
                        reply,
                        value,
                        json_line[1]["choices"][0]["finish_reason"],  # finish reason
                    ]

            elif game == "ultimatum_receiver" or game == "dictator_binary":
                if game == "ultimatum_receiver":
                    if exptype == "baseline":
                        fairness = re.findall(
                            r"\d+", json_line[0]["messages"][0]["content"].split(".")[7]
                        )
                    else:
                        fairness = re.findall(
                            r"\d+", json_line[0]["messages"][0]["content"].split(".")[8]
                        )

                    reply = (
                        response_dict[
                            f'{[i for i in list(response_dict.keys()) if "re" in i or "Re" in i][0]}'
                        ],
                    )
                    value = response_dict[
                        f'{[i for i in list(response_dict.keys()) if "decision" in i or "Decision" in i or "suggestion" in i or "dicision" in i][0]}'
                    ]

                    return [
                        json_line[0]["messages"][0]["content"],  # the full prompt
                        json_line[0]["temperature"],  # temperature
                        fairness,
                        age,  # age
                        gender,  # gender
                        exptype,  # prompt_type
                        game,  # game type,
                        reply,
                        value,
                        json_line[1]["choices"][0]["finish_reason"],  # finish reason
                    ]

                else:
                    if exptype == "baseline":
                        fairness = re.findall(
                            r"\d+", json_line[0]["messages"][0]["content"].split(".")[3]
                        )
                    else:
                        fairness = re.findall(
                            r"\d+", json_line[0]["messages"][0]["content"].split(".")[4]
                        )

                    reply = (
                        response_dict[
                            f'{[i for i in list(response_dict.keys()) if "re" in i or "Re" in i][0]}'
                        ],
                    )
                    value = response_dict[
                        f'{[i for i in list(response_dict.keys()) if "option_preferred" in i][0]}'
                    ]

                    return [
                        json_line[0]["messages"][0]["content"],  # the full prompt
                        json_line[0]["temperature"],  # temperature
                        fairness,
                        age,  # age
                        gender,  # gender
                        exptype,  # prompt_type
                        game,  # game type,
                        reply,
                        value,
                        json_line[1]["choices"][0]["finish_reason"],  # finish reason
                    ]

        else:
            exception_list.append("dict_keys")  # the exception type
            exception_list.append(json_line[0]["messages"][0]["content"])  # the prompt
            exception_list.append(json_line[0]["temperature"])  # the model temperature
            exception_list.append(response_dict)  # the model response

            # we need to see if the nested dict is of form {'answer': {'reply': 'Text', 'amount_sent':'3'}}
            # or {'reply': 'Text', {'answer': {'amount_sent':'3'}}
            # print(list(response_dict.keys()))
            # if isinstance(response_dict[list(response_dict.keys())[0]], dict):
            #     keys_2 = list(response_dict[list(response_dict.keys())[0]].keys())
            # elif isinstance(response_dict[list(response_dict.keys())[1]], dict):
            #     keys_2 = list(response_dict[list(response_dict.keys())[1]].keys())

    except:
        if json_line[1]["choices"][0]["finish_reason"] == "content_filter":
            exception_list.append("openai_content_filter")  # the exception type
            exception_list.append(json_line[0]["messages"][0]["content"])  # the prompt
            exception_list.append(json_line[0]["temperature"])  # the model temperature

        else:
            exception_list.append("json_parsing")  # the exception type
            exception_list.append(json_line[0]["messages"][0]["content"])  # the prompt
            exception_list.append(json_line[0]["temperature"])  # the model temperature
            exception_list.append(
                json_line[1]["choices"][0]["message"]["content"]
            )  # the model response
        return exception_list

        # print(keys_exists(response_dict, "answer", "reply"))
        # if keys_exists(response_dict, "suggestion", ['amount_sent', 'amount', 'Sent']):
        #     reply = response_dict[f'{[i for i in list(response_dict.keys()) if "re" in i or "Re" in i][0]}']
        #     value = response_dict['suggestion'][[i for i in list(response_dict.keys()) if "amount" in i or "money" in i or "sent" in i or "Sent" in i][0]]
        #     print(reply, value)
