from flask import Flask, request, render_template, redirect, url_for
import os
import config
import embed  
import answer  

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """
    Handle file upload and preview if necessary.
    """
    pass

@app.route('/delete/<filename>')
def delete_file(filename):
    """
    Handle file deletion.
    """
    pass

@app.route('/embed')
def create_embeddings():
    """
    Call functions in embed.py to create embeddings.
    """
    pass

@app.route('/query', methods=['GET', 'POST'])
def submit_query():
    """
    Allow users to input a query and display the response.
    """
    if request.method == 'POST':
        # When form is submitted, process the query
        question = request.form['question']
        # Use answer.py script to get the response
        response = answer.answer_question(answer.df, question=question)
        return render_template('response.html', question=question, answer=response)
    else:
        # Render a form to input a question
        return render_template('submit_query.html')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5555)
