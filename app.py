from flask import Flask, request, render_template, redirect, url_for, flash
import os
from flask import jsonify
import config
import embed  
import answer  

app = Flask(__name__)

# Initialize the list to store the history of questions and answers
conversation_history = []

@app.route('/')
def index():
    summaries = []

    # Open summary file
    with open('processed/summary_recursive.txt', 'r') as file:
        lines = file.readlines()  # leggi tutte le linee del file
        #text = file.read()

    # result = answer.summarize_text(text, debug=False)
    # if result["success"]:
    #     lines = result["summaries"]

    for line in lines:
        line = line.strip()  # rimuovi gli spazi bianchi (inclusi i ritorni a capo) all'inizio e alla fine della stringa
        if line:  # se la linea non è vuota
            summaries.append(line)  # aggiungila alla lista   
    
    return render_template('index.html', paragraphs=summaries)

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        # Use answer.py script to get a summary of the text
        result = answer.summarize(answer.df, debug=True)
        if result["success"]:
            # Ensure the text directory exists
            if not os.path.exists('processed'):
                os.makedirs('processed')
        
            # Write summaries to a file
            with open('processed/summary.txt', 'w') as file:
                for summary in result["summaries"]:
                    file.write(summary + "\n")

            # Join the summaries into a single string
            summary = "\n".join(result["summaries"])
            result = answer.summarize_text(summary, debug=True)
            if result["success"]:
                # Write summaries to a second file
                with open('processed/summary_recursive.txt', 'w') as file:
                    for summary in result["summaries"]:
                        file.write(summary + "\n")
        
            # Return a success response
            return jsonify({"success": True, "message": "Riassunti scritti con successo."}), 200
        else:
            return jsonify({"success": False, "error": result["error"]}), 500
    except Exception as e:
        # Return an error response
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/query', methods=['GET', 'POST'])
def query():
    """
    Allow users to input a query and display the response.
    """
    context_texts = []
    if request.method == 'POST':
        # When form is submitted, process the query
        question = request.form['question']

        # Use answer.py script to get the response
        response, context_texts = answer.answer_question(answer.df, question=question, max_len=3300, max_tokens=600, debug=True)

        # Prepend the question and response to the conversation history to display it at the top
        conversation_history.insert(0, {'question': question, 'answer': response})

    return render_template('query.html', conversation_history=conversation_history, context_texts=context_texts)

@app.route('/clear_conversation', methods=['POST'])
def clear_conversation():
    # Clear the conversation history
    conversation_history.clear()

    # Redirect back to the conversation page
    return redirect(url_for('query'))

@app.route('/files')
def list_files():
    # Ottieni la lista dei file nella cartella 'text'
    files = os.listdir('text')
    # Renderizza il template, passando la lista dei file
    return render_template('file_list.html', file_list=files)

@app.route('/view_file/<filename>')
def view_file(filename):
    # Leggi il contenuto del file
    with open(os.path.join('text', filename), 'r') as file:
        content = file.read()
    # Renderizza il template, passando il nome del file e il suo contenuto
    return render_template('view_file.html', filename=filename, content=content)

@app.route('/delete_file/<filename>', methods=['GET'])
def delete_file(filename):
    # Crea il percorso completo del file
    file_path = os.path.join(config.UPLOAD_FOLDER, filename)
    
    # Verifica se il file esiste
    if os.path.exists(file_path):
        # Prova a eliminare il file
        try:
            os.remove(file_path)
            # Messaggio di successo (opzionale)
            flash('File eliminato con successo.', 'success')
        except:
            # Messaggio di errore (opzionale)
            flash('Si è verificato un errore durante l\'eliminazione del file.', 'error')
    else:
        # Messaggio di errore (opzionale)
        flash('File non trovato.', 'error')
    
    # Reindirizza l'utente alla pagina della lista dei file
    return redirect(url_for('list_files'))

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Controlla se c'è un file come parte della richiesta
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        # Se l'utente non seleziona un file, il browser potrebbe
        # inviare una parte di un file senza nome.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # Salva il file
        if file:
            filename = file.filename
            file.save(os.path.join(config.UPLOAD_FOLDER, filename))
            flash('File caricato con successo!', 'success')
            return redirect(url_for('list_files'))

    # Se il metodo è GET, mostra il form di upload
    return render_template('upload.html')

@app.route('/create_embeddings', methods=['POST'])
def create_embeddings():
    """
    Call a function in embed.py to create embeddings.
    """    
    try:
        # Lancia la funzione per creare gli embedding
        embed.create_embeddings()
        # Ritorna un successo
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        # In caso di errore ritorna un messaggio di errore
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Configura la chiave segreta per i messaggi flash
    app.secret_key = config.app_key
    app.run(debug=True, host="0.0.0.0", port=5555)
