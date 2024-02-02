import bert_score
import numpy as np
import pandas as pd
import nltk.translate.meteor_score as meteor_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import Tuple


def get_bleu_score_for_prediction(prediction: str, references: list, sanitize_output: bool) -> float:
    ''' Returns the BLEU score for the given prediction and list of references '''
    if sanitize_output:
        prediction = _sanitize_output(prediction)

    prediction_ = prediction.split()
    references_ = [ref.split() for ref in references]

    chencherry = SmoothingFunction()
    return sentence_bleu(references_, prediction_, smoothing_function=chencherry.method3)


def get_meteor_score_for_prediction(predictions: str, references: list, sanitize_output: bool) -> float:
    ''' Returns the METEOR score for the given prediction and list of references '''
    if sanitize_output:
        predictions = _sanitize_output(predictions)

    predictions_ = predictions.split()
    references_ = [ref.split() for ref in references]

    return meteor_score.meteor_score(references_, predictions_)


def get_bert_score_for_prediction(predictiction: str, references: list, sanitize_output: bool) -> float:
    ''' Returns the BERT score for the given prediction and list of references '''
    return get_all_bert_scores_for_prediction(predictiction, references, sanitize_output)[2]

def get_all_bert_scores_for_prediction(predictiction: str, references: list, sanitize_output: bool) -> Tuple[float, float, float]:
    ''' Returns the BERT precision, recall and f1 scores for the given prediction and list of references '''
    if sanitize_output:
        predictiction = _sanitize_output(predictiction)

    precision, recall, F1 = bert_score.score([predictiction], [references], lang="en", verbose=True)

    return precision.item(), recall.item(), F1.item()


def get_rouge_score_for_prediction(predictiction: str, references: list, sanitize_output: bool) -> float:
    ''' Returns the ROUGE score for the given prediction and list of references '''
    if sanitize_output:
        predictiction = _sanitize_output(predictiction)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = []
    for reference in references:
        score = scorer.score(reference, predictiction)
        scores.append(score)
    return np.mean([x["rougeL"].fmeasure for x in scores])


def get_lexicalisation_of_references(references: list, filter: bool = True) -> list:
    ''' Returns the filtered lexicalisations of the given references. All lexicalisations that are not marked as good are filtered out. '''
    if filter:
        return [x['lex'] for x in references if x["comment"] == "good"]
    else:
        return [x['lex'] for x in references]


def add_results_to_webnlg_evaluation_df(webnlg_df: pd.DataFrame, model: str, webnlg_evaluation_df: pd.DataFrame, sanitize_output: bool = False) -> pd.DataFrame:
    ''' Adds the results of the given webnlg_train_df to the given webnlg_train_evaluation_df '''
    # Add modified tripleset to the evaluation dataframe
    webnlg_evaluation_df['modifiedtripleset'] = webnlg_df['modifiedtripleset']

    # Add lexicalisations to the evaluation dataframe
    webnlg_evaluation_df['lexicalisations'] = [get_lexicalisation_of_references(x) for x in webnlg_df["lexicalisations"]]

    # Add ids to the evaluation dataframe
    webnlg_evaluation_df['id'] = webnlg_df['id']

    # Add predictions to the evaluation dataframe
    webnlg_evaluation_df[f'prediction_{model}'] = webnlg_df[f'prediction_{model}']

    # Add BLEU scores to the evaluation dataframe
    all_bleu_scores = [get_bleu_score_for_prediction(x[0], x[1], sanitize_output) for x in zip(webnlg_evaluation_df[f'prediction_{model}'], webnlg_evaluation_df['lexicalisations'])]
    webnlg_evaluation_df[f'bleu_{model}'] = all_bleu_scores

    # Add METEOR scores to the evaluation dataframe
    all_meteor_scores = [get_meteor_score_for_prediction(x[0], x[1], sanitize_output) for x in zip(webnlg_evaluation_df[f'prediction_{model}'], webnlg_evaluation_df['lexicalisations'])]
    webnlg_evaluation_df[f'meteor_{model}'] = all_meteor_scores

    # Add ROUGE scores to the evaluation dataframe
    all_rouge_scores = [get_rouge_score_for_prediction(x[0], x[1], sanitize_output) for x in zip(webnlg_evaluation_df[f'prediction_{model}'], webnlg_evaluation_df['lexicalisations'])]
    webnlg_evaluation_df[f'rouge_{model}'] = all_rouge_scores
    
    # Add Bert scores to the evaluation dataframe
    all_bert_scores = [get_bert_score_for_prediction(x[0], x[1], sanitize_output) for x in zip(webnlg_evaluation_df[f'prediction_{model}'], webnlg_evaluation_df['lexicalisations'])]
    webnlg_evaluation_df[f'bert_{model}'] = all_bert_scores

    # Add execution time to the evaluation dataframe
    webnlg_evaluation_df[f'execution_time_{model}'] = webnlg_df[f'execution_time_{model}']

    # Add mean scores to the evaluation dataframe
    bleu_mean = webnlg_evaluation_df[f'bleu_{model}'].mean()
    meteor_mean = webnlg_evaluation_df[f'meteor_{model}'].mean()
    rouge_mean = webnlg_evaluation_df[f'rouge_{model}'].mean()
    bert_mean = webnlg_evaluation_df[f'bert_{model}'].mean()
    execution_time_mean = webnlg_evaluation_df[f'execution_time_{model}'].mean()

    new_row = pd.Series([
                        "mean",
                        "-",
                        "-",
                        "-",
                        bleu_mean,
                        meteor_mean,
                        rouge_mean,
                        bert_mean,
                        execution_time_mean
        ], index=webnlg_evaluation_df.columns)
    
    return pd.concat([webnlg_evaluation_df, pd.DataFrame([new_row])], ignore_index=True)


def print_metrics(model: str, webnlg_evaluation_df: pd.DataFrame) -> None:
    print(f"\n{model}:")

    # Calculate mean bleu score of the given model
    print(f"Mean bleu_{model}: {webnlg_evaluation_df[f'bleu_{model}'].mean()}")

    # Calculate mean meteor score of the given model
    print(f"Mean meteor_{model}: {webnlg_evaluation_df[f'meteor_{model}'].mean()}")

    # Calculate mean rouge score of the given model
    print(f"Mean rouge_{model}: {webnlg_evaluation_df[f'rouge_{model}'].mean()}")

    # Calculate mean bert score of the given model
    print(f"Mean bert_{model}: {webnlg_evaluation_df[f'bert_{model}'].mean()}")

    # Calculate mean execution time of the given model
    print(f"Mean execution_time_{model}: {webnlg_evaluation_df[f'execution_time_{model}'].mean()}")

def _sanitize_output(output: str) -> str:
    ''' Sanitizes the given output '''
    return output.replace("Output text: ", "", 1)

def remove_trailing_input_prompt(prediction: str) -> str:
    '''The LLaMA model usally repeates the input prompt at the end of the prediction. This function removes it.'''
    # For the few-shot prompts, LLaMA always starts by repeating the system prompt.
    prediction = prediction.split("\nGenerate a concise text")[0]
    # For the zero-shot prompts, LLaMA always starts by repeating the first user prompt.
    prediction = prediction.split("USER:")[0]
    return prediction

def remove_leading_input_tags(prediction: str) -> str:
    '''For few-shot learning, the prompt nuges the model to start with "Output text:". Zero-shot prompts often start with "Output:". This function removes it.'''
    prediction =  prediction.replace("Output text: ", "")
    prediction =  prediction.replace("Output: ", "")
    return prediction

def clean_response(prediction: str) -> str:
    prediction = remove_trailing_input_prompt(prediction)
    prediction = remove_leading_input_tags(prediction)
    return prediction
