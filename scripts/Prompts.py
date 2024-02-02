from typing import List

def get_zero_shot_chat_prompt(input_tripleset: List[dict], system_message: bool = False) -> List[dict]:
    ''' Returns the template for a prompt that converts triples to text'''
    message = []

    if system_message:
        message.append({"role": "system", "content": f"{_get_system_prompt()}"})
    
    message.append({"role": "user", "content": f"Input triples: {input_tripleset}"})
    return message

def get_zero_shot_completion_prompt(input_tripleset: List[dict], system_message: bool = True) -> str:
    prompt = ""

    if system_message:
        prompt += f"{_get_system_prompt()}\n\n"

    prompt += f"""Input triples: {input_tripleset}
Output text: """
    return prompt

def get_few_shot_chat_prompt(input_tripleset: List[dict], system_message: bool = False) -> List[dict]:
    ''' Returns the template for a prompt that converts triples to text'''
    message = []

    if system_message:
        message.append({"role": "system", "content": f"{_get_system_prompt()}"})
    
    message.append({"role": "user", "content": f"Input triples: {_get_triples_01()}"})
    message.append({"role": "assistant", "content": f"Output text: {_get_lexicalization_01()}"})
    message.append({"role": "user", "content": f"Input triples: {_get_triples_02()}"})
    message.append({"role": "assistant", "content": f"Output text: {_get_lexicalization_02()}"})
    message.append({"role": "user", "content": f"Input triples: {_get_triples_03()}"})
    message.append({"role": "assistant", "content": f"Output text: {_get_lexicalization_03()}"})
    message.append({"role": "user", "content": f"Input triples: {input_tripleset}"})
    return message

def get_few_shot_completion_prompt(input_tripleset: List[dict], system_message: bool = True) -> str:
    prompt = ""

    if system_message:
        prompt += f"{_get_system_prompt()}\n\n"
    
    prompt += f"""Input triples: {_get_triples_01()}
Output text: {_get_lexicalization_01()}
    
Input triples: {_get_triples_02()}
Output text: {_get_lexicalization_02()}
    
Input triples: {_get_triples_03()}
Output text: {_get_lexicalization_03()}

Input triples: {input_tripleset}
Output text: """
    return prompt

def get_few_shot_reit_chat_prompt(input_tripleset: List[dict], system_message: bool = False) -> List[dict]:
    ''' Returns the template for a prompt that converts triples to text'''
    message = []

    if system_message:
        message.append({"role": "system", "content": f"{_get_system_reit_prompt()}"})
    
    message.append({"role": "user", "content": f"Input triples: {_get_triples_01()}"})
    message.append({"role": "assistant", "content": f"Output text: {_get_lexicalization_01()}"})
    message.append({"role": "user", "content": f"Input triples: {_get_triples_02()}"})
    message.append({"role": "assistant", "content": f"Output text: {_get_lexicalization_02()}"})
    message.append({"role": "user", "content": f"Input triples: {_get_triples_03()}"})
    message.append({"role": "assistant", "content": f"Output text: {_get_lexicalization_03()}"})
    message.append({"role": "user", "content": f"Input triples: {input_tripleset}"})
    return message

def get_few_shot_reit_completion_prompt(input_tripleset: List[dict], system_message: bool = True) -> str:
    prompt = ""

    if system_message:
        prompt += f"{_get_system_reit_prompt()}\n\n"
    
    prompt += f"""Input triples: {_get_triples_01()}
Output text: {_get_lexicalization_01()}

Input triples: {_get_triples_02()}
Output text: {_get_lexicalization_02()}

Input triples: {_get_triples_03()}
Output text: {_get_lexicalization_03()}

Input triples: {input_tripleset}
Output text: """
    return prompt

def get_finetune_instruction() -> str:
    return f"""{_get_system_reit_prompt()}

Input triples: {_get_triples_01()}
Output text: {_get_lexicalization_01()}

Input triples: {_get_triples_02()}
Output text: {_get_lexicalization_02()}

Input triples: {_get_triples_03()}
Output text: {_get_lexicalization_03()}
"""

def get_fintune_input(input_tripleset: List[dict]) -> str:
    return f"""Input triples: {input_tripleset}
Output text: """

def _get_system_prompt() -> str:
    return "Generate a concise text for the given set of triples. Ensure that the generated output only includes the provided information from the triples."

def _get_system_reit_prompt() -> str:
    return f"{_get_system_prompt()} Remember to conform to the output format of the examples. Transfer the entire knowledge contained in the input triples to the text and avoid adding additional information."

def _get_triples_01() -> str:
    return "[{'object': 'Mike_Mularkey','property': 'coach','subject': 'Tennessee_Titans'}]"

def _get_lexicalization_01() -> str:
    return "Mike Mularkey is the coach of the Tennessee Titans."

def _get_triples_02() -> str:
    return "[{'object': 'Albert_E._Austin', 'property': 'successor', 'subject': 'Alfred_N._Phillips'}, {'object': 'Connecticut', 'property': 'birthPlace', 'subject': 'Alfred_N._Phillips'}, {'object': 'United_States_House_of_Representatives', 'property': 'office', 'subject': 'Alfred_N._Phillips'}]"

def _get_lexicalization_02() -> str:
    return "Albert E. Austin succeeded Alfred N. Phillips who was born in Connecticut and worked at the United States House of Representatives."

def _get_triples_03() -> str:
    return "[{'object': 'College_of_William_&_Mary', 'property': 'owner', 'subject': 'Alan_B._Miller_Hall'}, {'object': '2009-06-01', 'property': 'completionDate', 'subject': 'Alan_B._Miller_Hall'}, {'object': '\'101 Ukrop Way\'', 'property': 'address', 'subject': 'Alan_B._Miller_Hall'}, {'object': 'Williamsburg,_Virginia', 'property': 'location', 'subject': 'Alan_B._Miller_Hall'}, {'object': 'Robert_A._M._Stern', 'property': 'architect', 'subject': 'Alan_B._Miller_Hall'}]"

def _get_lexicalization_03() -> str:
    return "The Alan B Miller Hall's location is 101 Ukrop Way, Williamsburg, Virginia. It was designed by Robert A.M. Stern and was completed on 1 June 2009. Its owner is the College of William and Mary."

#######################
#  Text-To-Data Prompts
#######################

def get_text_to_data_zero_shot_chat_prompt(input_text: str, system_message: bool = False) -> List[dict]:
    ''' Returns the template for a prompt that converts text to triples'''
    message = []

    if system_message:
        message.append({"role": "system", "content": f"{_get_text_to_data_system_prompt()}"})
    
    message.append({"role": "user", "content": f"Input text: {input_text}"})
    return message

def get_text_to_data_zero_shot_completion_prompt(input_text: str) -> str:
    return f"""{_get_text_to_data_system_prompt()}

Input text: {input_text}
Output triples: """

def get_text_to_data_few_shot_chat_prompt(input_text: str, system_message: bool = False) -> List[dict]:
    ''' Returns the template for a prompt that converts text to triples'''
    message = []

    if system_message:
        message.append({"role": "system", "content": f"{_get_text_to_data_system_prompt()}"})
    
    message.append({"role": "user", "content": f"Input text: {_get_lexicalization_01()}"})
    message.append({"role": "assistant", "content": f"Output triples: {_get_triples_01()}"})
    message.append({"role": "user", "content": f"Input text: {_get_lexicalization_02()}"})
    message.append({"role": "assistant", "content": f"Output triples: {_get_triples_02()}"})
    message.append({"role": "user", "content": f"Input text: {_get_lexicalization_03()}"})
    message.append({"role": "assistant", "content": f"Output triples: {_get_triples_03()}"})
    message.append({"role": "user", "content": f"Input text: {input_text}"})
    return message

def get_text_to_data_few_shot_completion_prompt(input_text: str) -> str:
    return f"""{_get_text_to_data_system_prompt()}

Input text: {_get_lexicalization_01()}
Output triples: {_get_triples_01()}
    
Input text: {_get_lexicalization_02()}
Output triples: {_get_triples_02()}
    
Input text: {_get_lexicalization_03()}
Output triples: {_get_triples_03()}

Input text: {input_text}
Output triples: """

def get_text_to_data_few_shot_reit_chat_prompt(input_text: str, system_message: bool = False) -> List[dict]:
    ''' Returns the template for a prompt that converts text to triples'''
    message = []

    if system_message:
        message.append({"role": "system", "content": f"{_get_text_to_data_system_reit_prompt()}"})
    
    message.append({"role": "user", "content": f"Input text: {_get_lexicalization_01()}"})
    message.append({"role": "assistant", "content": f"Output triples: {_get_triples_01()}"})
    message.append({"role": "user", "content": f"Input text: {_get_lexicalization_02()}"})
    message.append({"role": "assistant", "content": f"Output triples: {_get_triples_02()}"})
    message.append({"role": "user", "content": f"Input text: {_get_lexicalization_03()}"})
    message.append({"role": "assistant", "content": f"Output triples: {_get_triples_03()}"})
    message.append({"role": "user", "content": f"Input text: {input_text}"})
    return message

def get_text_to_data_few_shot_reit_completion_prompt(input_text: str) -> str:
    return f"""{_get_text_to_data_system_reit_prompt()}

Input text: {_get_lexicalization_01()}
Output triples: {_get_triples_01()}

Input text: {_get_lexicalization_02()}
Output triples: {_get_triples_02()}

Input text: {_get_lexicalization_03()}
Output triples: {_get_triples_03()}

Input text: {input_text}
Output triples: """

def _get_text_to_data_system_prompt() -> str:
    return "Generate a set of RDF triples for the given text. Ensure that the generated triples only include the provided information from the text."

def _get_text_to_data_system_reit_prompt() -> str:
    return f"{_get_text_to_data_system_prompt()} Remember to conform to the output format of the examples. Transfer the entire knowledge contained in the input text to the set of triples and avoid adding additional information."
