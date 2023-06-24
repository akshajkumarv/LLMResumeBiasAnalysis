# Are Emily and Greg Still More Employable than Lakisha and Jamal? Investigating Algorithmic Hiring Bias in the Era of ChatGPT

## Table of Contents

- [Installation](#installation)
- [Data](#data)
- [API Keys](#api-keys)
- [Usage](#usage)

## Installation

To install the required dependencies for this project, run the following command in your terminal:

`pip install -r requirements.txt`

This will install all the required dependencies listed in the `requirements.txt` file.

## Data

The dataset used in this project can be downloaded from [Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset). The dataset contains a collection of resumes taken from livecareer.com for categorizing a given resume into any of the labels defined in the dataset.
In our research study, we only focus on the categories Information-Technology, Construction, and Teacher.

To use the dataset in this project, download the dataset from the link above and unzip it in the `data` directory. No preprocessing steps are required as the dataset is already cleaned and ready to use.

## API Keys
In order to be able to use the [GPT API](https://openai.com/blog/openai-api/) you need to have an API key. Once you have access to the API, you can find your API key in the OpenAI dashboard. Store your API key in a txt-file in the `api_keys/gpt/` directory. For GPT, place your organization key in the first line, and your api key in the second line.

You have to do the same for Bard. For Bard, the API is not publicly available yet, but you can request access [here](https://developers.generativeai.google). Once you have access to the API, store your API key in a file in the `api_keys/bard/` directory. As it is only one key, simple place the key in the first line of the txt-file.

Claude is also in private beta, but you can request access [here](https://www.anthropic.com/earlyaccess). Once you have access to the API, store your API key in a file under the name `api_key.txt` in the `api_keys/claude/` directory.
Similarly, for Claude, place your api key in the first line of the txt-file.



## Usage

To create the basic dataset we used in our research study, run the following command in your terminal:

`python create_dataset.py`

This will create a dataset with a total of 1336 resumes for the three categories Information-Technology, Construction, and Teacher. The dataset will be saved in the `data` directory under the name `selected_cats_resumes.csv`.

If you want to create summaries for the resumes in the dataset, run the following command in your terminal:

`python summaries.py --model model_name --api_key api_key_file`

If you want to do the classification task, run the following command in your terminal:

`python classification.py --model model_name --api_key api_key_file`

You can further specify a sensitive attribute with the following flags:

`--political_orientation`
`--pregnancy`
`--employment_gap`

Note, that you can only set one of these flags.

To do the classification task on the summaries, run the following command in your terminal:

`python classification_summaries.py --model model_name --api_key api_key_file`

Again, you have the same choices as above, but you must first run the summaries.py script to create the summaries.
