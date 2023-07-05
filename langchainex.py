import langchain
from langchain.llms import OpenAI


OPENAI_API_KEY="sk-"
 

llm = OpenAI(openai_api_key=OPENAI_API_KEY)

llm = OpenAI(temperature=0.9)

llm.predict("What would be a good company name for a company that makes colorful socks?")
# >> Feetful of Fun