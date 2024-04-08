import pandas as pd
import numpy as np
import re
import openai
import os
import json
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoConfig
import torch
import tiktoken

pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL,model_max_length = 514,truncation=True)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# PATH TO WHERE TO STORE FINAL OUTPUT DATA
diplomatic_pulse_data_sentiment_scores_file_path = os.path.dirname(os.path.abspath(__file__))
# Name of file which stores the data
diplomatic_pulse_data_sentiment_scores_filename = "/DiplomaticPulseIsraelPalestineFrom07Oct2023-SentimentScores.json"

# FILE PATH TO LAST_INDEXED_DATE.TXT
# get this directory
last_index_filepath = os.path.dirname(os.path.abspath(__file__))

# FILE PATH TO INPUT DATA
extractions_df_file_path = os.path.dirname(os.path.abspath(__file__))
# Input file containing extractions, currently in JSON format
extractions_df_filename = "/ExtractedSentences.json"

# Read and store the data in a DataFrame
extractions_df = pd.read_json(extractions_df_file_path+extractions_df_filename,  orient='records')

last_statement_index_file_path = os.path.dirname(os.path.abspath(__file__))
path_to_statementindex = last_statement_index_file_path+"/last_sentiment_index.txt"


# Get index of the first extraction column-positive_israel
idx_extraction = extractions_df.columns.get_loc('positive_israel')

# Column name list for sentiment scores
score1_pos_israel = "gpt4score_positive_israel"
score1_neg_israel = "gpt4score_negative_israel"
score1_pos_opt = "gpt4score_positive_opt"
score1_neg_opt = "gpt4score_negative_opt"
score1_pos_hamas = "gpt4score_positive_hamas"
score1_neg_hamas = "gpt4score_negative_hamas"

fields_list1 = [score1_pos_israel, score1_neg_israel, score1_pos_opt, score1_neg_opt, score1_pos_hamas, score1_neg_hamas]

score2_pos_israel = "Roberta_positive_israel"
score2_neg_israel = "Roberta_negative_israel"
score2_pos_opt = "Roberta_positive_opt"
score2_neg_opt = "Roberta_negative_opt"
score2_pos_hamas = "Roberta_positive_hamas"
score2_neg_hamas = "Roberta_negative_hamas"

fields_list2 = [score2_pos_israel, score2_neg_israel, score2_pos_opt, score2_neg_opt, score2_pos_hamas, score2_neg_hamas]

score_pos_israel = "ensemble_positive_israel"
score_neg_israel = "ensemble_negative_israel"
score_pos_opt = "ensemble_positive_opt"
score_neg_opt = "ensemble_negative_opt"
score_pos_hamas = "ensemble_positive_hamas"
score_neg_hamas = "ensemble_negative_hamas"

fields_list3 = [score_pos_israel, score_neg_israel, score_pos_opt, score_neg_opt, score_pos_hamas, score_neg_hamas]

# FILE PATH TO API KEY
# get this directory
api_key_filepath = os.path.dirname(os.path.abspath(__file__))
api_key_filename = "/openai_apikey.txt" 
api_key_file_path = api_key_filepath + api_key_filename

def get_api_key(file_path=api_key_file_path):
    """
    Function to read the textfile with api key
    """
    with open(file_path, 'r') as file:
        apikey = file.read().strip()  # Read the key and remove any leading/trailing white spaces
    return apikey

openai.api_key = get_api_key()
openai.api_base = "https://unocc-d-openai-eastus2-sentiment-opt-israel.openai.azure.com/" 
openai.api_type = 'azure'
openai.api_version = '2023-05-15' # this may change in the future

deployment_name='unocc-d-aoai-eus2-gpt4-sentiment-opt-israel' #This will correspond to the custom name you chose for your deployment when you deployed a model. 

seed=407

def send_message(messages):
    """
    Function to send the prompt to the ChatGPT model
    """
    response = openai.ChatCompletion.create(
        engine=deployment_name,
        messages=messages,
        seed=seed,
        temperature=0,
        top_p=0.1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response['choices'][0]['message']['content']


def clean_extractions(df,latest_statement_index):
    """
    Function that selects the lates data that needs to be scored,
    and cleans latest extractions in the input DataFrame by converting them to strings & removing brackets
    """
    df = df[df["index"] > latest_statement_index]
    for column in df.columns[idx_extraction:(idx_extraction+6)]:
        df.loc[:,column] = df.loc[:,column].astype(str)
        df.loc[:,column] = df.loc[:,column].str.replace('[]', "no extractions",regex=False) 
        df.loc[:,column] = df.loc[:,column].str.replace("[", "",regex=False)
        df.loc[:,column] = df.loc[:,column].str.replace("]", "",regex=False)
        df.loc[:,column].str.replace("\'", "",regex=True)
        df.loc[:,column].fillna("None", inplace=True) # Empty fields like travel advisories renamed to 'None'
    return df

def analyze_gpt(text, system_message,i):
    """
    Function that makes openai calls
    """

    # Run GPT only when there are extractions, not when there are empty fields and no extractions
    if text in ['None','no extractions']:
        return text
    else:
        messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f""" {text}"""}
        ]
        
        response = send_message(messages)

        # Clean the response that is in form of a list- [-0.5] or ''
        response = response.replace("[", "")
        response = response.replace("]", "")
        response = response.replace("''", "0")
        score = pd.to_numeric(response, errors='coerce')

        # Absolute values for positive extractions and -1 times of absolute value of scores for negative scores
        if i % 2 == 0:
            score = abs(score)
        else:
            score = -1*abs(score)

        return score

# NO- BIAS message
def create_message_no_bias(entity):
    """
    Function to create a system message for GPT
    """
    sentiment_score=[]
    base_system_message =f""" You are helping with a sentiment analysis workflow, and
    returning a list containing a numerical sentiment score (call this list {sentiment_score}).
    Do not include preconceived notions about Israel, Palestine, Hamas, but only refer to the language used when assigning a sentiment score.
    The sentiment scoring should range from -1 (very negative sentiment), 0 neutral, 1 (very positive sentiment). The sentiment score increases or decreases in a continuous .1 interval.
    Please score the sentiments in the text towards {entity} using the previous sentiment scoring rules.
    Do not generate and return any text.
    Return only one overall numerical sentiment score for all the extracted sentences from the statement. 
    Return a list containing overall numerical sentiment score (call this list {sentiment_score})
    Here are a few non-exhaustive examples of sentiment scoring based on the rules provided.
    In this example A, the sentiment score towards Country A is 1.
    Example A Text:
    'your visit is another tangible example of our unequivocal support for Country A.', 'Thank you, for standing with Country A today, tomorrow, and always.', 'we’ve also been inspired by the bravery of Country A’s citizens', 'The people of Country A have long – and rightly – prided themselves on their self-reliance, on their ability to defend themselves, even when the odds are stacked against them.', 'You may be strong enough on your own to defend yourself – but as long as we exists, you will never, ever have to. We will always be there, by your side.', 'We welcome the government’s creation and the unity and resolve that it reflects across Country A’s society.', 'we will work with Congress to make sure that they’re met.', 'We has Country A’s back.', 'We’re pursuing intensive diplomacy throughout the region to prevent the conflict from spreading', 'Country A has the right – indeed, the obligation – to defend itself and to ensure that this never happens again.', 'That is what we will do, standing with Country A, working together with its people – and all those in this region who remain committed to the vision of a more peaceful, a more integrated, a more secure, a more prosperous Middle East.'
    In this example B, the sentiment score towards Country A is 0.833.
    Example B Text:
    'We stand with the people of Country A in the face of this terrorist assault.', 'Country A has the right to defend itself and its people.', 'We reiterate our ironclad support for Country A's exercise of its inherent right to defend itself in the face of this unjustified attack.'
    In this example C, the sentiment score towards Country A is 0.416.
    Example C Text:
    'Yesterday, I wrote to Country A's Foreign Minister Eli Cohen to express our deep condolences.', 'I visited Country A, and the occupied Country A's territory just one month ago.'
    In this example D, the sentiment score towards Country A is 0.
    Example D Text:
    'The Government unreservedly condemns the attacks on Country A carried out by the terrorist organisation Hamas.', 'In response to the attacks, the Government will instruct to conduct a review of aid to Country A to ensure that no funds go to actors that do not unconditionally condemn Hamas, that commit violence, threaten or encourage violence against the State of Country A or its population, or pursue an antisemitic agenda, nor to people associated with such actors.'
    In this example E, the sentiment score towards Country A is -0.417.
    Example E Text:
    'The ongoing harsh conditions under which the Country As live in veritable colonialism and Country A's sense of insecurity will contribute to a cycle of violence until those realities are definitively addressed.'
    In this example F, the sentiment score towards Country A is -0.917.
    Example F Text:
    'We hold Country A, the occupying power, fully responsible for the repercussions of the continuation of this sinful aggression,'
    In this example G, the sentiment score towards Country A is 0.75.
    Example G Text:
    'As Chair of the international donor group, we encourage the international community to continue its financial assistance to the Country A's people.', 'Humanitarian assistance to the people of Gaza should be a key priority.', 'Functioning Country A's institutions and adequate service delivery are critical to avoid further destabilization and maintaining the objective of the two-state solution.', 'I support any efforts at preventing a further deterioration of the situation.', 'To achieve peace, there is no alternative other than to restart a political process between Country A and Country A's.'
    In this example H, the sentiment score towards Country A is 0.5.
    Example H Text:
    'I was clear that continued development and humanitarian support to the Country A's people is essential.', 'I welcomed  the confirmation that humanitarian aid to Country A's will continue uninterrupted, for as long as needed.'
    In this example I, the sentiment score towards Country A is 0.
    Example I Text:
    'At the same time, as we're pursuing normalization, it's imperative that it not be a substitute for Country A and Country A's resolving the differences between them.  On the contrary, it needs to be something that actually advances that prospect and supports it.'
    In this example J, the sentiment score towards Country A is -0.25.
    Example J Text:
    ‘… underlined the need for the Country A's Authority to clearly distance itself from the Hamas terrorist organisation and condemn its attacks.'
    In this example K, the sentiment score towards Hamas is -0.083.
    Example K Text:   
    'We oppose and condemn acts harming civilians.'
    In this example L, the sentiment score towards Hamas is -0.583.
    Example L Text:   
    'The Government strongly condemns the terrorist attack in Country A today that caused loss of precious human lives and injured many more.'
    In this example M, the sentiment score towards Hamas is -0.75.
    Example M Text:   
    "Nothing justifies what we have seen Hamas engage in, and you've heard me, I think you and I have spoken before about our position in terms of seeking a just and enduring two-state solution which recognises the legitimate aspirations of both the Jewish and Country A's peoples.", 'Nothing justifies the violence, the hostage taking, the killing of civilians, the awful scenes we have seen Hamas engaging in.'
    In this example N, the sentiment score towards Hamas is -1.
    Example N Text:   
    'Hamas will understand that by attacking us, they have made a mistake of historic proportions.', 'The savage attacks that Hamas perpetrated against innocent Country A are mindboggling: Slaughtering families in their homes, massacring hundreds of young people at an outdoor festival, kidnapping scores of women, children and elderly, even Holocaust survivors.', 'Hamas terrorists bound, burned and executed children. They are savages.', 'Hamas is ISIS.', 'And just as the forces of civilization united to defeat ISIS, the forces of civilization must support Country A in defeating Hamas.'
    
    Please round the sentiment score upto 1 decimal point. Make sure that there are no score with more than 1 decimal point.
    """
    return base_system_message

def roberta_score(text, idx,i):
    """
    Function that runs roBERTa model
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    if inputs['input_ids'].numel() == 0:
        return 0
    output = model(**inputs)
    if idx < 0 or idx >= output.logits.size(1):
        return 0
    preds = torch.softmax(output.logits, dim=1).tolist()[0]

    # Absolute values for positive extractions and -1 times of absolute value of scores for negative scores
    if i % 2 == 0:
        score = abs(preds[idx])
    else:
        score = -1*abs(preds[idx])

    return score

def split_text_into_chunks(text, max_token_length=512):
    """
    Split longer text into chunks based on the maximum token length (roBERTa model) starting from the first sentence
    Return chunks
    """
    # Split the text into sentences at punctuations .!?;
    sentences = re.split(r'([.!?;])', text)
    
    chunks = []
    current_chunk_tokens = []
    chunks_tokens = []
    current_chunk = ""

    for i, sentence in enumerate(sentences):
        # Tokenize the sentence
        tokens = tokenizer.encode(sentence)
        #print(i, len(tokens), sentence)
        #print("Current token length = ",len(current_chunk_tokens))
        
        # If adding the current sentence exceeds the max token length, start a new chunk
        if len(current_chunk_tokens) + len(tokens) > max_token_length:
            chunks_tokens.append(current_chunk_tokens)
            chunks.append(current_chunk)
            current_chunk_tokens = tokens
            current_chunk = sentence
     
        else:
            current_chunk_tokens += tokens
            current_chunk += sentence

    # If we have iterated over the entire text, append the remaining chunk
    if current_chunk:
        chunks_tokens.append(current_chunk_tokens)
        chunks.append(current_chunk)

    return chunks

def split_text_into_chunks_reverse(text, max_token_length=512):
    """
    Split longer text into chunks based on the maximum token length
    Chunks are created from the last sentence to the first.
    Return chunks
    """
    # Split the text into sentences at punctuations .!?;
    sentences = re.split(r'([.!?;])', text)
    
    chunks = []
    current_chunk_tokens = []
    chunks_tokens = []
    current_chunk = ""

    # Iterate over sentences in reverse order
    for i in range(len(sentences) - 1, -1, -1):
        sentence = sentences[i]
        # Tokenize the sentence
        tokens = tokenizer.encode(sentence)
        #print(i, len(tokens), sentence)
        #print("Current token length = ",len(current_chunk_tokens))
        
        # If adding the current sentence exceeds the max token length, start a new chunk
        if len(current_chunk_tokens) + len(tokens) > max_token_length:
            chunks_tokens.append(current_chunk_tokens)
            chunks.append(current_chunk)
            current_chunk_tokens = tokens
            current_chunk = sentence
            
        else:
            current_chunk_tokens += tokens
            current_chunk = sentence + current_chunk  # Append to the beginning

    # If we have iterated over the entire text, append the remaining chunk
    if current_chunk:
        chunks_tokens.append(current_chunk_tokens)
        chunks.append(current_chunk)

    return chunks

def roberta_sentiment_score(text, idx, i):
    """
    main function to run roberta model given the text, index to sentiment type-positive/negative, and index of extractions
    chunk statement if necessary, then evaluate chunks or entire statement, and 
    return a score
    """
    if text in ['None','no extractions']:
        return text
    else:
        token_len = len(tokenizer.encode(text))
        max_allowable_token = 512

        if token_len > max_allowable_token:
            # split message into chunks
            chunk_scores=[]
            message_chunks = []
            message_chunks = split_text_into_chunks(text, max_token_length=max_allowable_token)
            for j,chunk in enumerate(message_chunks):
                print("Forward Chunk ......", j, "...........")
                #print(chunk)
                score= roberta_score(chunk, idx,i)
                chunk_len = len(tokenizer.encode(chunk))
                print("Chunk length :   ",chunk_len)
                weight = chunk_len / token_len
                print("Chunk weight :   ", weight)
                chunk_scores.append(score * weight)
            final_score_1 = np.sum(chunk_scores)

            chunk_scores=[]
            message_chunks = []
            message_chunks = split_text_into_chunks_reverse(text, max_token_length=max_allowable_token)
            for j,chunk in enumerate(message_chunks):
                print("Reverse Chunk ......", j, "...........")
                #print(chunk)
                score= roberta_score(chunk, idx,i)
                chunk_len = len(tokenizer.encode(chunk))
                print("Chunk length :   ",chunk_len)
                weight = chunk_len / token_len
                print("Chunk weight :   ", weight)
                chunk_scores.append(score * weight)
            final_score_2 = np.sum(chunk_scores)
            final_score = np.mean([final_score_1,final_score_2])
            
        else:
            final_score = roberta_score(text,idx,i)

        return final_score
          

def get_scores(df=extractions_df,
               latest_statement_index = 0,
               write_data=True,
               write_data_path=diplomatic_pulse_data_sentiment_scores_file_path, 
               write_data_filename=diplomatic_pulse_data_sentiment_scores_filename):
    """
    Function that takes extractions DataFrame and an index for new statements as input.
    Cleans extractions- rename empty fields as 'None' and fields with no extractions '[]' as 'no extractions'
    Gets sentiment scores through GPT and roBERTa models
    Calculates ensemble scores- equally weighted average sentiment scores from both models
    Returns a DataFrame with all sentiment scores and writes/appends it in a JSON file
    """
    df = clean_extractions(df,latest_statement_index)

    # Select entity that is passes into the message for GPT 
    for i in range(0, len(fields_list1)):
        column_index = idx_extraction + i
        if i<=1 :
            entity = "Israel"
        elif 1<i<=3 :
            entity = "Palestine"
        elif 3<i<=5:
            entity = "Hamas"
        
        print(entity)
        base_system_message =create_message_no_bias(entity)
        system_message = f"{base_system_message}".strip().replace('\n', '')
        
        df.loc[:, fields_list1[i]] = df.iloc[:, column_index].apply(lambda x: analyze_gpt(x, system_message,i))

    # Empty scores if any are converted to 0.
    df[fields_list1] = df[fields_list1].fillna(0)

    for i in range(0, len(fields_list2)):
        column_index = idx_extraction + i
        print(i)
        if column_index % 2 == 0:
            rob_score = df.iloc[:,column_index].apply(lambda x: roberta_sentiment_score(x,0,i))
        else:
            rob_score = df.iloc[:,column_index].apply(lambda x: roberta_sentiment_score(x,2,i))
        df.loc[:,fields_list2[i]] = rob_score
        
    df[fields_list2] = df[fields_list2].fillna(0)

    # Ensemble scores- equally weighted average of GPT and roBERTa scores
    idx_score = df.columns.get_loc(fields_list1[0])
    weight_model1 = 0.5
    weight_model2 = 0.5
    for i in range(len(fields_list3)):
        column_index_1 = idx_score + i  # GPT scores
        column_index_2 = idx_score + i + 6  # roBERTa scores

        # Average only where there are numerical scores
        for index, value in enumerate(df.iloc[:, column_index_1]):
            if value in ['None', 'no extractions']:
                df.at[index, fields_list3[i]] = value
            else:
                df.at[index, fields_list3[i]] = weight_model1 * df.iloc[index, column_index_1] + weight_model2 * df.iloc[index, column_index_2]
    
    df['date'] = df['date'].astype(str)
    # If we want to write the data, append to JSON file
    if write_data:

        # get full filepath
        file = write_data_path + write_data_filename

        # new data
        new_data = pd.DataFrame(df.iloc[:,])
        
        # Check if the JSON file exists
        if os.path.exists(file):
            # If the file exists, read the existing data
            existing_data = pd.read_json(file, orient='records')
            
            # Append the new data to the existing data
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            # If the file doesn't exist, create a new DataFrame
            updated_data = new_data

        # Write the DataFrame to a JSON file
        updated_data.to_json(file,  orient='records')

        latest_index = df['index'].values.max()
        with open(path_to_statementindex, 'w') as file:
            file.write(str(latest_index))

    return df

with open(path_to_statementindex, 'r') as file:
    latest_statement_index = int(file.read().strip())  # Read the key and remove any leading/trailing white spaces

scores = get_scores(latest_statement_index=latest_statement_index)