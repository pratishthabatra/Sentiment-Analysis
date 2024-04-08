## Code

This folder contains Jupyter nootebooks to get sentiment scores on sentence extractions using the following approaches:

* GPT- 3.5 (_gpt35-SentimentScores-DiplomaticPulseData.ipynb_)

* GPT-4 (_gpt4-SentimentScores-DiplomaticPulseData.ipynb_)

* roBERTa (_Roberta-SentimentScores-DiplomaticPulseData-Israel-Palestine.ipynb_)

* Azure (_Azure-SentimentScores-DiplomaticPulseData-Israel-Palestine.ipynb_)

* BERT- Cosine Similarity (_BERT-CosineSimilarity-SentimentScores-DiplomaticPulseData-Israel-Palestine.ipynb_)
  
* VADER (_Vader-SentimentScores-DiplomaticPulseData-Israel-Palestine.ipynb_)



## Data

This folder contains data used by the code in exploratory-codes to get sentiment scores on sentence extractions:

* sentence extractions on the random sample of 50 extractions (_50sample_extractions_moreRestrictive.xlsx_)

* ground truth scores (_AverageSentiment Scoring-RLU-DiplomaticPulse-Israel-oPt_issuingCountry.xlsx_)

* base document containing keywords related to reactions/aids for cosine similarity (_Base-for-Cosine-Similarity-DiplomaticPulseData.xlsx_)


## Prediction eror (MAE) for all approaches
|	|Sentiment scoring approach |	Israel	|oPt	|Hamas	|Overall|
|-------|---------------------------|------------|-------|-------|-------|
|1	|GPT 4|0.18	|0.09	|0.18	|0.15|
2	|Bert-Cosine Similarity|0.25 |0.13	|0.19	|0.19|
3	|roBERTa	|0.32	|0.10|	0.21|	0.21|
4	|Vader	|0.29|	0.17|	0.19|	0.22|
5	|Azure	|0.41	|0.11|	0.18|	0.24|
6	|GPT 3.5 |0.40	|0.12|	0.24|	0.25|
