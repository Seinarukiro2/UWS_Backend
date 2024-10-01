import os
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader, UnstructuredImageLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents.base import Document
import openai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

class NodeInstallationBot:
    def __init__(self):
        self.db_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db")
        self.vector_db = None
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.client = openai.OpenAI()
        os.environ['USER_AGENT'] = 'NodeInstallationBot/1.0'

        if not os.path.exists(self.db_directory):
            os.makedirs(self.db_directory)

        # Initialize Chroma with the OpenAI embedding model
        self.vector_db = Chroma(persist_directory=self.db_directory, embedding_function=OpenAIEmbeddings(model='text-embedding-3-small'))
        
    def load_and_store_data(self, url):
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()

            # Проверяем, является ли результат списком объектов Document
            if isinstance(documents, list) and all(isinstance(doc, Document) for doc in documents):
                # Извлекаем текст из каждого документа
                text = "\n".join([doc.page_content for doc in documents])
            else:
                raise ValueError(f"Ожидался список объектов Document, но получены: {[type(doc) for doc in documents]}")

            # Используем RecursiveCharacterTextSplitter для разбивки текста на части
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
            document_chunks = text_splitter.split_text(text)  # Разбиваем текст на части

            # Добавляем текстовые части в векторное хранилище
            self.vector_db.add_texts(document_chunks)
            print(f"Данные успешно загружены и сохранены из {url}")
            return True
        except Exception as e:
            print(f"Ошибка при загрузке и сохранении данных: {e}")
            raise

    def extract_text_from_image(self, image_path):
        try:
            loader = UnstructuredImageLoader(image_path)
            data = loader.load()
            # Combine all extracted text from the image into one string
            text = "\n".join([doc.page_content for doc in data])
            return text
        except Exception as e:
            logging.error(f"Error extracting text from image: {e}")
            return ""

    def ask_question(self, question, image_path=None):
        # Extract text from the image if provided
        image_text = self.extract_text_from_image(image_path) if image_path else ""

        # Combine question and image text
        combined_question = question + "\n" + image_text

        # Perform similarity search in the vector database with MMR for diversity
        results = self.vector_db.max_marginal_relevance_search(combined_question, k=3, fetch_k=5)
        if results:
            # Use Contextual Compression to focus on the most relevant parts
            compressor = LLMChainExtractor.from_llm(self.client)
            compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=self.vector_db.as_retriever())

            compressed_docs = compression_retriever.get_relevant_documents(combined_question)
            context = "\n".join([doc.page_content for doc in compressed_docs])

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Ты бот асистент компании ClickTime. Отвечай на том языке на котором тебе был задан вопрос. Игнорируй вопросы, которые не относятся к теме."}, 
                    {"role": "user", "content": f"{combined_question}"},
                    {"role": "assistant", "content": context}
                ]
            )
            logging.info(f"Response generated for question: {question}")
            return response.choices[0].message.content
        else:
            logging.warning(f"No relevant information found for question: {question}")
            return "No relevant information found."


bot = NodeInstallationBot()

response = bot.ask_question("Как установить ноду gaiaNet")
print(response)