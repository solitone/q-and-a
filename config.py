import os
from dotenv import load_dotenv
import openai

# Folder where text files are saved
UPLOAD_FOLDER = 'text/'

# Maximum number of tokens of each text chunk that gets embedded
MAX_TOKENS = 500 

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
#print(openai.api_key)
app_key = os.getenv("APP_KEY")