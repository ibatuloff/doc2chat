import llama_index
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
from llama_index.llms.ollama import Ollama
import telebot
import os
import logging

BOT_TOKEN = os.environ.get("BOT_TOKEN")
DATA_DIR = "/app/data"
PERSIST_DIR = "/app/storage"
query_engine = None

Settings.embed_model = OptimumEmbedding(folder_name="/app/models/all-mpnet-base-v2_onnx")
Settings.llm = Ollama(model="qwen25_7b", request_timeout=180.0)

def create_index() -> object:
    reader = SimpleDirectoryReader(DATA_DIR)
    docs = reader.load_data()
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index

def create_query_engine() -> object:
    global query_engine
    if os.path.exists(PERSIST_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    else:
        index = create_index()
    query_engine = index.as_query_engine()

bot = telebot.TeleBot(BOT_TOKEN) 
create_query_engine()
logging.info("query engine ready!")

@bot.message_handler(commands = ['start'])
def handle_start(message):
    bot.send_message(message.chat.id, 'Greetings! I am you personal assistant in your preferred domain. Any questions?')

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        user_query = message.text
        logging.info(f'user query: {user_query}')
        response = query_engine.query(user_query)
        bot.send_message(message.chat.id, response)
    except Exception as e:
        logging.exception(f"query processing failed {e}")
        bot.reply_to(message, "Sorry, an error occured while processing your message...")

bot.polling()