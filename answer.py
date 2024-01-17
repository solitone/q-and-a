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
import pandas as pd
import numpy as np
import os
from datetime import datetime
import openai
from openai.embeddings_utils import distances_from_embeddings
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff
import embed
import config # set openai.api_key
from textutils import clean_text

# Turn the embeddings into a NumPy array, which will provide more flexibility
df=pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
#print(df.head())

def get_context_texts(question, df, max_len=1800):
    """
    Given a question, return the most similar context texts from the dataframe along with
    their distances from the question.
    
    :param question: The input question as a string.
    :param df: The DataFrame containing the texts and their embeddings.
    :param max_len: The maximum length of the combined context texts.
    :return: A list of tuples where each tuple contains a context text and its distance from the question.
    """

    # Step 1: Compute embeddings for the input question
    q_embeddings = openai.Embedding.create(
            input=question, engine='text-embedding-ada-002'
        )['data'][0]['embedding']

    # Step 2: Compute the distances between question embeddings and context text embeddings
    df['distances'] = distances_from_embeddings(
        q_embeddings, df['embeddings'].values, distance_metric='cosine')

    # Step 3: Initialize an empty list to store the selected context texts and their distances
    context_texts_with_distances = []
    current_length = 0

    # Step 4: Iterate over the texts sorted by their distances in ascending order
    for index, row in df.sort_values('distances', ascending=True).iterrows():

        # Update the current length including a buffer for extra characters
        current_length += row['n_tokens'] + 4

        # If adding the next text exceeds the maximum length, stop adding texts
        if current_length > max_len:
            break

        # Add the text and its distance to the list
        text = row["text"]
        distance = row["distances"]
        context_texts_with_distances.append((text, distance))

    # Return the list of context texts along with their distances
    return context_texts_with_distances

def answer_question(df, model="gpt-3.5-turbo", question="Di cosa parla il testo?", max_len=1800, max_tokens=200, debug=False):
    """
    Answer a question based on the most similar context from the dataframe texts.
    
    :param df: The DataFrame containing the embeddings and text.
    :param model: The language model to be used.
    :param question: The input question as a string.
    :param max_len: The maximum length of the combined context texts.
    :param max_tokens: The maximum number of tokens for the language model response.
    :param debug: Boolean to control debug prints.
    :return: The answer string and context texts with their distances.
    """

    # Step 1: Define system message instructing the model how to behave
    system_msg = "Rispondi alla domanda basandoti UNICAMENTE sul contesto sotto. Usa un linguaggio semplice e chiaro. Se non puoi dare una risposta basandoti sul contesto, rispondi \"Non so.\", NON usare la tua conoscenza personale."
    
    # Step 2: Get the context texts related to the input question
    context_texts_with_distances = get_context_texts(question, df, max_len=max_len)
    # Extract only the texts from the (text, distance) tuples for context
    context_texts = "\n\n###\n\n".join([text_and_dist[0] for text_and_dist in context_texts_with_distances])

    # Step 3 (Optional): Debug prints for token counts
    if debug:
        print("System message tokens: " + str(embed.get_n_tokens(system_msg)))
        print("Context tokens: " + str(embed.get_n_tokens(context_texts)))
        print("Question tokens: " + str(embed.get_n_tokens(question)))

    # Step 4: Generate the answer using the OpenAI API
    try:
        # Make a request to OpenAI API with the context and question
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Contesto: {context_texts}"},
                {"role": "user", "content": f"Domanda: {question}"}
            ],
            max_tokens=max_tokens,
        )

        # Extract the answer from the API response
        answer = response["choices"][0]["message"]["content"].strip()

        # Step 5 (Optional): Debug prints for response details
        if debug:
            print("Prompt tokens: " + str(response["usage"]["prompt_tokens"]))
            print("Completion tokens: " + str(response["usage"]["completion_tokens"]))
            print("Total tokens: " + str(response["usage"]["total_tokens"]))
            print("Question: " + question)
            print("Context: " + context_texts)

        # Return the answer along with context_texts_with_distances
        return answer, context_texts_with_distances

    except Exception as e:
        # Handle exceptions and print error message
        print(e)
        return ""

def summarize_text(text, model="gpt-3.5-turbo", max_tokens=1000, debug=False):
    system_msg = (
                'Fai un breve riassunto del testo riportato qui sotto. NON parlare in prima persona. '
                'Limitati a spiegare quello che si dice nel testo, '
                'evitando formule del tipo "nel testo si dice" oppure "questo testo parla". '
                'Usa un linguaggio chiaro, semplice e accattivante, come fosse un blog, '
                'in modo da suscitare l\'interesse del lettore.'
                )
    
    cleansed_text = clean_text(text)
    chunks = embed.split_into_many(text=cleansed_text, max_tokens=1000)
    
    summaries = []
    num_chunks = len(chunks)

    for index, chunk in enumerate(chunks):
        if debug:
            # Ottieni l'ora corrente
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"**Text #{str(index+1)} of {num_chunks} ({current_time})")
            print("-------------")
            print(chunk[:160] + "...\n\n")
        try:
            # Make a request to OpenAI API with the context and question
            response = chat_completion_with_backoff(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": chunk},
                ],
                max_tokens=max_tokens,
            )

            # Extract the answer from the API response
            summary = response["choices"][0]["message"]["content"].strip()

            # (Optional): Debug prints for response details
            if debug:
                print("Prompt tokens: " + str(response["usage"]["prompt_tokens"]))
                print("Completion tokens: " + str(response["usage"]["completion_tokens"]))
                print("Total tokens: " + str(response["usage"]["total_tokens"]))
                print(f"**Summary #{str(index+1)} of {num_chunks}")
                print("-------------")
                print(summary + "\n\n")

            summaries.append(summary)       

        except Exception as e:       
            # Get the full exception class name
            exception_fullname = type(e).__module__ + "." + type(e).__name__
            # Get the error message
            error_message = str(e)
            # Combine them
            full_error = f"{exception_fullname}: {error_message}"
            # Handle exceptions and return error message
            return {"success": False, "error": full_error}

    summarized_text = "\n".join(summaries)

    if debug:
        # Ensure the logs directory exists
        if not os.path.exists('logs'):
            os.makedirs('logs')
        # Summaries will be written to a log file
        with open('logs/summary.log', 'a') as file:
            current_datetime = datetime.now()
            file.write(f"{current_datetime} ------------------------------------------\n")  
            file.write(f"{summarized_text}\n\n")

    if len(summarized_text.split()) > 1000:
        # Se il riassunto ha ancora un numero di parole superiore alla soglia, chiamare nuovamente la funzione
        return summarize_text(summarized_text, model, max_tokens, debug)
    else:
        # Altrimenti, restituisci le sintesi
        return {"success": True, "summaries": summaries}





def summarize(df, model="gpt-3.5-turbo", max_tokens=1000, debug=False):
    # Define system message instructing the model how to behave
    system_msg = (
                'Fai un breve riassunto del testo riportato qui sotto. NON parlare in prima persona. '
                'Limitati a spiegare quello che si dice nel testo, '
                'evitando formule del tipo "nel testo si dice" oppure "questo testo parla". '
                'Usa un linguaggio chiaro, semplice e accattivante, come fosse un blog, '
                'in modo da suscitare l\'interesse del lettore.'
                )
    
    summaries = []

    # Numero di righe del dataframe
    num_rows = len(df)
    # Loop in every text chunk found in the dataframe
    for index, row in df.iterrows():
        longtext = row["text"]
        if debug:
                # Ottieni l'ora corrente
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"**Text #{str(index+1)} of {num_rows} ({current_time})")
                print("-------------")
                print(longtext[:160] + "...\n\n")
        try:
            # Make a request to OpenAI API with the context and question
            response = chat_completion_with_backoff(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": longtext},
                ],
                max_tokens=max_tokens,
            )

            # Extract the answer from the API response
            summary = response["choices"][0]["message"]["content"].strip()

            # (Optional): Debug prints for response details
            if debug:
                print("Prompt tokens: " + str(response["usage"]["prompt_tokens"]))
                print("Completion tokens: " + str(response["usage"]["completion_tokens"]))
                print("Total tokens: " + str(response["usage"]["total_tokens"]))
                print(f"**Summary #{str(index+1)} of {num_rows}")
                print("-------------")
                print(summary + "\n\n")

            summaries.append(summary)

        except Exception as e:       
            # Get the full exception class name
            exception_fullname = type(e).__module__ + "." + type(e).__name__
            # Get the error message
            error_message = str(e)
            # Combine them
            full_error = f"{exception_fullname}: {error_message}"
            # Handle exceptions and return error message
            return {"success": False, "error": full_error}
        
    # Return the summaries list with success status
    return {"success": True, "summaries": summaries}    


@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout)), 
    wait=wait_random_exponential(multiplier=1, max=60), 
    stop=stop_after_attempt(10)
)
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)



if __name__ == "__main__":
    try:
        with open('processed/summary.txt', 'r') as file:
            text = file.read()

        result = summarize_text(text, debug=True)
        if result["success"]:
            # Write summaries to a file
            with open('processed/summary_recursive.txt', 'w') as file:
                for summary in result["summaries"]:
                    file.write(summary + "\n")
        else:
            print(f"Errore: {result['error']}")
        
    except Exception as e:
        # Return an error response
        print(f"Exception: {type(e).__module__}.{type(e).__name__}: {str(e)}")