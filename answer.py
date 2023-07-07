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
import openai
from openai.embeddings_utils import distances_from_embeddings
import embed
import config # set openai.api_key

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
        input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

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

def answer_question(df, model="gpt-3.5-turbo", question="Esiste una caverna chiamata caverna gigante?", max_len=1800, max_tokens=200, debug=False):
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
    system_msg = "Rispondi alla domanda basandoti sul contesto sotto. Usa un linguaggio adatto a un ragazzino di 14 anni. Se non puoi dare una risposta basandoti sul contesto rispondi \"Non so.\""
    
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

if __name__ == "__main__":
    question="Che caratteristiche hanno le caverne che si aprono nella parete del Seguret?"
    print(question)
    answer, context_texts_with_dists = answer_question(df, question=question)
    print(answer)