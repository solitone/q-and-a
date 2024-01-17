import re
import spacy

def clean_text(text):
    """
    Remove blank lines from the input series to declutter the text files
    and make them easier to process.
    """
    
    # Replace multiple newlines (and whitespaces around or within them) with a single newline
    text = re.sub(r'\s*\n\s*', '\n', text)
    
    # Replace dash followed by a newline with an empty string
    text = re.sub(r'-\n', '', text)
    
    # Replace remaining newlines with a space
    text = text.replace('\n', ' ')
    
    # Ensure that every full stop is followed by a blank
    #text = text.replace('.', '. ')

    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    
    return text

def split_into_sentences(text):
    # Carica il modello di lingua. 'it_core_news_sm' è per l'italiano.
    # Carica il modello di lingua. 'en_core_web_sm' è per l'inglese.
    nlp = spacy.load('en_core_web_sm')

    # Processa il testo con spaCy
    doc = nlp(text)
    
    # Estrai le frasi dal documento processato
    sentences = [sent.text for sent in doc.sents]
    
    return sentences

if __name__ == "__main__":
    # # Leggi il file di testo
    # with open('text/fenomeno_carsico.txt', 'r') as file:
    #     text = file.read()

    # # Pulisci il testo
    # cleaned_text = clean_text(text)

    # # Scrivi il testo pulito in un nuovo file
    # with open('text/fenomino_carsico-cleansed.txt', 'w') as file:
    #     file.write(cleaned_text)


    # Testo di esempio
    text = "In Europa ci sono molte belle montagne, per esempio il M. Genevris si trova in Italia. È una bella località."

    # Suddividi il testo in frasi
    sentences = split_into_sentences(text)

    # Stampa le frasi
    for sentence in sentences:
        print(sentence)

