import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_groq import ChatGroq

load_dotenv()

client = MongoClient(os.getenv("MONGODB_URI"))
client.admin.command("ping")
print("MongoDB connected")

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
print("Groq says:", llm.invoke("Say hi in 5 words").content)