import re
import os
import json


def keys_exists(element, *keys):
    '''
    Check if *keys (nested) exists in `element` (dict).
    '''
    if not isinstance(element, dict):
        raise AttributeError('keys_exists() expects dict as first argument.')
    if len(keys) == 0:
        raise AttributeError('keys_exists() expects at least two arguments, one given.')

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

    response_dict = json.loads(json_line[1]["choices"][0]["message"]["content"])

    # when experimental then coolect age and gender from prompt message
    if exptype == 'experimental':
        message = json_line[0]['messages'][0]['content']
        demographic_phrase = message.split(". ")[1]
        demographic_words = demographic_phrase.split(' ')
        # collect demographic info of each prompt
        age = demographic_words[2]
        gender = demographic_words[-1]

    # check if we have one dict or nested dicts
    # e.g. some answers are {'reply': 'Text', 'amount_sent':'3'}
    # others are nested in the form of {'answer': {'reply': 'Text', 'amount_sent':'3'}}
    # or {'reply': 'Text', {'answer': {'amount_sent':'3'}}

    if not any(isinstance(i, dict) for i in response_dict.values()):

        if game == "dictator_sender" or game == "ultimatum_sender" or game == "dictator_sequential":
            reply = response_dict[f'{[i for i in list(response_dict.keys()) if "re" in i or "Re" in i][0]}'],
            value = response_dict[
                f'{[i for i in list(response_dict.keys()) if "amount" in i or "money" in i or "sent" in i or "Sent" in i][0]}']
        elif game == "ultimatum_receiver" or game == "dictator_binary":
            if game == 'ultimatum_receiver':
                reply = response_dict[f'{[i for i in list(response_dict.keys()) if "re" in i or "Re" in i][0]}'],
                value = response_dict[
                    f'{[i for i in list(response_dict.keys()) if "decision" in i or "Decision" in i or "suggestion" in i or "dicision" in i][0]}']
            else:
                reply = response_dict[f'{[i for i in list(response_dict.keys()) if "re" in i or "Re" in i][0]}'],
                value = response_dict[f'{[i for i in list(response_dict.keys()) if "option_preferred" in i][0]}']
    else:

        # we need to see if the nested dict is of form {'answer': {'reply': 'Text', 'amount_sent':'3'}}
        # or {'reply': 'Text', {'answer': {'amount_sent':'3'}}
        try:
            x = response_dict['suggestion']['amount_sent']
        except KeyError:
            pass

        # print(keys_exists(response_dict, "answer", "reply"))
        # if keys_exists(response_dict, "suggestion", ['amount_sent', 'amount', 'Sent']):
        #     reply = response_dict[f'{[i for i in list(response_dict.keys()) if "re" in i or "Re" in i][0]}']
        #     value = response_dict['suggestion'][[i for i in list(response_dict.keys()) if "amount" in i or "money" in i or "sent" in i or "Sent" in i][0]]
        #     print(reply, value)

