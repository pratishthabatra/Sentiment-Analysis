## Sentiment Analysis

This folder contains the code to execute a sentiment analysis on international statements on Israel, Palestine and the Hamas, with data sourced from DiplomaticPulse (https://diplomaticpulse.org/en/).
  
## exploratory-code

This folder contains exploratory codes with various approaches tried in the process of creating the final pipeline to perform the sentiment analysis on the DiplomaticPulse data.

* **sentiment_analysis**: contains scripts for different models to perform sentiment analysis.

* **translation**: contains code to translate statements.


## final-code

* **read_DiplomaticPulse_data**: contains code and files needed to download all statements on DiplomaticPulse mentioning Israel, Palestine or Hamas from October 7, 2023 that had not been downloaded yet into a json file. _(private scripts)_

* **extract_sentences**: contains code and files needed to extract sentences voicing a sentiment towards Israel, Palestine or Hamas of downloaded DiplomaticPulse statements where sentences have not yet been extracted, using the GPT-4 model. _(private scripts)_

* **give_sentiment_score**: contains code and files needed to give a sentiment score to the extracted sentences voicing a sentiment towards Israel, Palestine or Hamas of downloaded DiplomaticPulse statements where a sentiment score has not yet been given, using the ensemble model- GPT-4 and roBERTa models.


##

Additionally, there are two files stored locally, which hold the keys to connect to the OpenAI models and the Azure translation service via the API.


