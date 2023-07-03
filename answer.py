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
import config # set openai.api_key

# Turn the embeddings into a NumPy array, which will provide more flexibility
df=pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
#print(df.head())

def create_context(
    question, df, max_len=1800
):
    """
    Create a context for a question by finding the most similar context from the dataframe.
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model="gpt-3.5-turbo",
    question="Esiste una caverna chiamata caverna gigante?",
    max_len=1800,
    max_tokens=200,
    debug=False
):
    """
    Answer a question based on the most similar context from the dataframe texts.
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
    )

    if debug:
        print(question)
        print(context)
        
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Rispondi alla domanda basandoti sul contesto sotto. Se non puoi dare una risposta basandoti sul contesto rispondi \"Non so.\""},
                {"role": "user", "content": f"Contesto: {context}"},
                {"role": "user", "content": f"Domanda: {question}"}
            ],
            max_tokens=max_tokens,
        )
        return response["choices"][0]["message"]["content"].strip()
            
    except Exception as e:
        print(e)
        return "" 

if __name__ == "__main__":
    question="Che caratteristiche hanno le caverne che si aprono nella parete del Seguret?"
    print(question)
    answer = answer_question(df, question=question)
    print(answer)