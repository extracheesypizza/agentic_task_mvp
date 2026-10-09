import logging
logging.basicConfig(level=logging.INFO)
from services.llm_client import chat
result = chat('You are a helpful assistant.', 'Что такое гемоглобин?')
print(result)