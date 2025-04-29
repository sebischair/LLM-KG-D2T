# A Comparative Analysis of Conversational Large Language Models in Knowledge-Based Text Generation
This GitHub repository contains the code and data resources related to the paper titled "A Comparative Analysis of Conversational Large Language Models in Knowledge-Based Text Generation", which has been accepted at the 18th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2024).

## Citation Information

For citing this study in academic papers, presentations, or theses, please use the following BibTeX entry:

``` 
@inproceedings{schneider-etal-2024-comparative,
    title = "A Comparative Analysis of Conversational Large Language Models in Knowledge-Based Text Generation",
    author = "Schneider, Phillip  and
      Klettner, Manuel  and
      Simperl, Elena  and
      Matthes, Florian",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-short.31",
    pages = "358--367",
    abstract = "Generating natural language text from graph-structured data is essential for conversational information seeking. Semantic triples derived from knowledge graphs can serve as a valuable source for grounding responses from conversational agents by providing a factual basis for the information they communicate. This is especially relevant in the context of large language models, which offer great potential for conversational interaction but are prone to hallucinating, omitting, or producing conflicting information. In this study, we conduct an empirical analysis of conversational large language models in generating natural language text from semantic triples. We compare four large language models of varying sizes with different prompting techniques. Through a series of benchmark experiments on the WebNLG dataset, we analyze the models{'} performance and identify the most common issues in the generated predictions. Our findings show that the capabilities of large language models in triple verbalization can be significantly improved through few-shot prompting, post-processing, and efficient fine-tuning techniques, particularly for smaller models that exhibit lower zero-shot performance.",
}
``` 

## Structure of the Repository
* `results`: Contains the predictions and evaluations for each model-prompt combination.
* `results/human_evaluation`: Contains code to create a file with instances for human annotators to label. Furthermore, it contains code for analyzing the results.
* `lora_adapter`: Contains the WebNLG training dataset and adapter that can be merged with the LLaMA-7B model to create a fine-tuned model. We refer to this model as LLaMA-FT-7B in the paper and as LoRA-7B in the code.
* `scripts/WebNLG\_Preparation.ipynb`: This script converts the XML files of the WebNLG dataset to JSON.
* `scripts/WebNLG\_Finetune\_Dataset.ipynb`: This script was used to create the fine-tuning dataset. It results in webnlg\_finetune\_dataset\_chat.json which we use to create the LoRA-7B model based on LLaMA-7B.
	* This is only needed if you want to change the size of the existing fine-tuning dataset.
* `scripts/WebNLG\_Finetune.ipynb`: Contains the code to fine-tune the LLaMA model using the LoRA (Low-Rank Adaptation) approach and webnlg\_finetune\_dataset\_chat.json as data.
* `scripts/WebNLG\_DataToText_Prediction.ipynb`: This is the code to generate verbalizations based on triples with models running on a local server (e.g., LLaMA, Vicuna, ...) and using the OpenAI API (e.g., GPT-3.5-Turbo).
* `scripts/WebNLG\_DataToText_Evaluation.ipynb`: Script to transform the predictions into the format expected by the official WebNLG evaluation script. Since the official evaluation script aggregates the results, we provide additional code to generate evaluation metrics for specific instances to enable a detailed analysis of the results.

## Setup
1. Clone this repository to your workspace
2. Download the [WebNLG dataset](https://gitlab.com/shimorina/webnlg-dataset/-/tree/master/)
3. Download the WebNLG [Corpus XML Reader](https://gitlab.com/webnlg/corpus-reader)
4. Use the _WebNLG\_Preparation.ipynb_ notebook to translate the XML files of WebNLG to JSON
5. Setup the [LLaMA](https://github.com/facebookresearch/llama) Large Language Model (LLM)
6. Setup the Vicuna LLM using [FastChat](https://github.com/lm-sys/FastChat)
7. If you want to use OpenAI models (e.g., GPT-3.5-turbo), rename the _.env.dist_ file to _.env_ and add your OpenAI API key
8. Use the _WebNLG\_DataToText_Prediction.ipynb_ notebook to transform RDF triples into text, using different LLMs and prompts
9. For evaluation, the _WebNLG\_DataToText\_Evaluation.ipynb_ notebook can be used

## Prompts
The applied prompts are defined in the file `Prompts.py`.

#### Zero-Shot
```
Generate a concise text for the given set of triples. Ensure that the generated output only includes the provided information from the triples.

Input triples: <triples>
```
#### Few-Shot
```
Generate a concise text for the given set of triples. Ensure that the generated output only includes the provided information from the triples.

Input triples: [{’object’: ’Mike\_Mularkey’,’property’: ’coach’,’subject’: ’Tennessee\_Titans’}]

Output text: Mike Mularkey is the coach of the Tennessee Titans.

Input triples: [{’object’: ’Albert\_E.\_Austin’, ’property’: ’successor’, ’subject’: ’Alfred\_N.\_Phillips’}, {’object’: ’Connecticut’, ’property’: ’birthPlace’, ’subject’: ’Alfred\_N.\_Phillips’}, {’object’: ’United\_States\_House\_of\_Representatives’, ’proper ty’: ’office’, ’subject’: ’Alfred\_N.\_Phillips’}]

Output text: Albert E. Austin succeeded Alfred N. Phillips who was born in Connecticut and worked at the United States House of Representatives.

Input triples: [{’object’: ’College\_of\_William\_&\_Mary’, ’property’: ’owner’, ’subject’: ’Alan\_B.\_Miller\_Hall’}, {’object’: ’2009-06-01’, ’property’: ’completionDate’, ’subject’: ’Alan\_B.\_Miller\_Hall’}, {’object’: ’101 Ukrop Way’, ’property’: ’address’, ’subject’: ’Alan\_B.\_Miller\_Hall’}, {’object’: ’Williamsburg,\_Virginia’, ’property’: ’location’, ’subject’: ’Alan\_B.\_Miller\_Hall’}, {’object’: ’Robert\_A.\_M.\_Stern’, ’prop-
erty’: ’architect’, ’subject’: ’Alan\_B.\_Miller\_Hall’}]

Output text: The Alan B Miller Hall’s location is 101 Ukrop Way, Williams- burg, Virginia. It was designed by Robert A.M. Stern and was completed on 1 June 2009. Its owner is the College of William and Mary.

Input triples: <triples>
```
