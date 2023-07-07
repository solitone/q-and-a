"""
The following code is adapted, with minor modifications, from the OpenAI tutorial 
titled "How to build an AI that can answer questions about your website." 
For further reference, see:
- https://platform.openai.com/docs/tutorials/web-qa-embeddings
- https://github.com/openai/openai-cookbook/tree/main/apps/web-crawl-q-and-a

------------------------------------------------------------------------------
MIT License

Copyright (c) 2023 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
------------------------------------------------------------------------------
"""
import os
import pandas as pd
import time
import openai
import tiktoken
import config # set openai.api_key
from textutils import clean_text, split_into_sentences

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")

def get_n_tokens(text):
    """
    Return the number of tokens of a given text.
    """
    return len(tokenizer.encode(text))

def split_into_many(text, max_tokens = config.MAX_TOKENS):
    """
    Split the text into chunks of a maximum number of tokens.
    """

    # Split the text into sentences
    sentences = split_into_sentences(text)

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the
        # current sentence is greater than the max number of tokens, then 
        # add the chunk to the list of chunks and reset the chunk 
        # and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than 
        # the max number of tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of 
        # tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks

def delayed_embedding(x, engine='text-embedding-ada-002', delay_in_seconds: float = 1):
    """
    Pace requests in order to avoid reaching the rate limit:
    https://platform.openai.com/docs/guides/rate-limits/overview
    """
    # Sleep for the delay
    time.sleep(delay_in_seconds)
    # Return embedding 
    return openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding']

def create_embeddings():
    """
    Create embeddings for the content in the text files.
    """

    # Create a list to store the text files
    texts=[]
    
    # Get all the text files in the text directory
    for file in os.listdir(config.UPLOAD_FOLDER):
    
        # Open the file and read the text
        with open(config.UPLOAD_FOLDER + file, "r", encoding="UTF-8") as f:
            text = f.read()
            cleansed_text = clean_text(text)

            # Add a tuple (file name, file text conent) to 'texts' list.
            #  - and _ in text file name are replaced with spaces.
            texts.append((file.replace('-',' ').replace('_', ' '), cleansed_text))
    
    # Create a dataframe from the list of texts
    df = pd.DataFrame(texts, columns = ['filename', 'text'])
    
    # Create a directory to store the csv files
    if not os.path.exists("processed"):
            os.mkdir("processed")
    df.to_csv('processed/texts.csv')
    #print(df.head())
    
    """
    Record current length of each row 
    to identify which rows need to be split
    """

    
    df = pd.read_csv('processed/texts.csv', index_col=0)
    df.columns = ['title', 'text']
    
    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    
    #print(df.head()) 
    
    """
    Split longer lines into smaller chunks
    """
    shortened = []
    
    # Loop through the dataframe
    for row in df.iterrows():
    
        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue
        
        # If the number of tokens is greater than the max number of tokens
        # split the text into chunks
        if row[1]['n_tokens'] > config.MAX_TOKENS:
            shortened += split_into_many(row[1]['text'])
        
        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append( row[1]['text'] )
    
    df = pd.DataFrame(shortened, columns = ['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    #print(df.head()) 
    
    # Create embeddings, using the function delayed_embedding()
    df['embeddings'] = df.text.apply(delayed_embedding)
    df.to_csv('processed/embeddings.csv')
    #print(df.head()) 

if __name__ == "__main__":
    create_embeddings()