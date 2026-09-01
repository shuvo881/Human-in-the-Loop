import os
from langchain_mistralai import ChatMistralAI
from dotenv import  load_dotenv
load_dotenv()

llm = ChatMistralAI(model="mistral-medium-3-5")