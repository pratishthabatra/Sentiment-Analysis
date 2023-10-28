import pandas as pd
import openai
import tiktoken
import time
import os


# PATH TO WHERE TO STORE FINAL OUTPUT DATA
diplomatic_pulse_data_sentiment_scores_file_path = os.path.dirname(os.path.abspath(__file__))
# Name of file which stores the data
diplomatic_pulse_data_sentiment_scores_filename = "/DiplomaticPulseIsraelPalestineFrom07Oct2023-SentimentScores.csv"

# FILE PATH TO LAST_INDEXED_DATE.TXT
# get this directory
last_index_filepath = os.path.dirname(os.path.abspath(__file__))

# FILE PATH TO INPUT DATA
extractions_df_file_path = os.path.dirname(os.path.abspath(__file__))
#extractions_df_filename = "/50sample_extractions_moreRestrictive.xlsx" # Input CSV file containing extractions, currently in Excel format
extractions_df_filename = "/50sample_extractions_moreRestrictive.csv"

# Read and store the data in a DataFrame
#extractions_df = pd.read_excel(extractions_df_file_path+extractions_df_filename) # Change this to pd.read_csv
extractions_df = pd.read_csv(extractions_df_file_path+extractions_df_filename)


# Get index of the first extraction column-positive_israel
idx_extraction = extractions_df.columns.get_loc('positive_israel')

# Column name list for sentiment scores
score_pos_israel = "gpt35score_positive_israel"
score_neg_israel = "gpt35score_negative_israel"
score_pos_opt = "gpt35score_positive_opt"
score_neg_opt = "gpt35score_negative_opt"
score_pos_hamas = "gpt35score_positive_hamas"
score_neg_hamas = "gpt35score_negative_hamas"

fields_list = [score_pos_israel, score_neg_israel, score_pos_opt, score_neg_opt, score_pos_hamas, score_neg_hamas]

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

deployment_name='unocc-d-aoai-eus2-gpt-35-turbo-sentiment-opt-israel' #This will correspond to the custom name you chose for your deployment when you deployed a model. 

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """
    Function to count number of tokens in a message
    """
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def send_message(messages):
    """
    Function to send the prompt to the ChatGPT model
    """
    response = openai.ChatCompletion.create(
        engine=deployment_name,
        messages=messages,
        temperature=0.2,
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
    df = df[df.iloc[:,0] > latest_statement_index]
    for column in df.columns[idx_extraction:(idx_extraction+6)]:
        df.loc[:,column] = df.loc[:,column].astype(str)
        df.loc[:,column] = df.loc[:,column].str.replace("[", "",regex=False)
        df.loc[:,column] = df.loc[:,column].str.replace("]", "",regex=False)
        df.loc[:,column] = df.loc[:,column].str.replace('[]', "",regex=False)
    return df

def analyze_gpt35(text, system_message):
    """
    Function that makes openai calls
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f""" {text}"""}
        ]

    #print("num_tokens_from_messages: ", num_tokens_from_messages(messages))
   
    response = send_message(messages)
    
    #print("response: ", response)

    return response

def get_scores(df=extractions_df,
               latest_statement_index = 0,
               write_data=False,
               write_data_path=diplomatic_pulse_data_sentiment_scores_file_path, 
               write_data_filename=diplomatic_pulse_data_sentiment_scores_filename):
    """
    Function that takes extractions DataFrame and an index for new statements as input and returns a DataFrame with sentiment scores and writes/appends it in a CSV file
    """
    df = clean_extractions(df,latest_statement_index)

    sentiment_score=[]
    base_system_message =f""" You are given extractions from a statement that you need to score. Return only one overall numerical sentiment score for all the extracted sentences.
    The sentiment scoring should range from -1 (very strong condemnation) to 1 (very strong support).
    For neutral statements, the sentiment score should be close to 0.
    There should be a scope in labelling the level of condemnation/support, based on the language used and whether or not any actions are announced, e.g. supply of aid (monetary, military, …) or sanctions.
    If there are no extractions, the score should be 0.
    Return a list containing overall numerical sentiment score (call this list {sentiment_score})"""

    for i in range(0, len(fields_list)):
        column_index = idx_extraction + i
        
        sentiment_score=[]
        system_message = f"{base_system_message}".strip().replace('\n', '')

        df.loc[:, fields_list[i]] = df.iloc[:, column_index].apply(lambda x: analyze_gpt35(x, system_message))

        # May have to pause openai calls if quota/token rate exceeds limit
        #runs_before_pause = 2   # This depends on size of the data
        #if i % runs_before_pause == 0:
        #   print("Pausing for 60 seconds...")
        #   time.sleep(60)  # Pause for 60 seconds
 
    for column in df.columns[(idx_extraction+6):]:
        df.loc[:, column] = df.loc[:, column].astype(str)
        df.loc[:, column] = df.loc[:, column].str.replace("[", "",regex=False)
        df.loc[:, column] = df.loc[:, column].str.replace("]", "",regex=False)
        df.loc[:, column] = df.loc[:, column].str.replace("''", "0",regex=False)
        df.loc[:, column] = df.loc[:, column].apply(pd.to_numeric, errors='coerce')
    
    # Keep the absolute values for positive sentences, and -1 * absolute value for the negative sentences
    # Get index of the first score column, i.e. for positive_israel
    col_idx_score = df.columns.get_loc(fields_list[0])
    positive_extractions = [col_idx_score, (col_idx_score+2), (col_idx_score+4)]
    negative_extractions = [(col_idx_score+1), (col_idx_score+3), (col_idx_score+5)]
    df.iloc[:, positive_extractions] = df.iloc[:, positive_extractions].abs()
    df.iloc[:, negative_extractions] = -df.iloc[:, negative_extractions].abs()

    # If we want to write the data, append to CSV file
    if write_data:

        # get full filepath
        file = write_data_path + write_data_filename

        # new data
        new_data = pd.DataFrame(df.iloc[:,])
        
        # Check if the CSV file exists
        if os.path.exists(file):
            # If the file exists, read the existing data
            existing_data = pd.read_csv(file)
            
            # Append the new data to the existing data
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            # If the file doesn't exist, create a new DataFrame
            updated_data = new_data

        # Write the DataFrame to an Excel file
        updated_data.to_csv(file, index=False)

    return df

scores = get_scores(write_data=True, latest_statement_index = 0)