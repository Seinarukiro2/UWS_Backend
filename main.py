import os
import asyncio
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.document_loaders import WebBaseLoader, UnstructuredImageLoader
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
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
        os.environ['USER_AGENT'] = 'NodeInstallationBot/1.0'
        
        # Initialize Chroma with OpenAI embedding model
        self.vector_db = Chroma(persist_directory=self.db_directory, embedding_function=OpenAIEmbeddings(model='text-embedding-3-small'))

        # Initialize ChatOpenAI as LLM
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    async def scrape_website(self, url):
        try:
            logging.info(f"Starting web scraping for URL: {url}")
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url)
                content = await page.content()
                await browser.close()
            soup = BeautifulSoup(content, "html.parser")
            text = soup.get_text()
            logging.info(f"Successfully scraped text from {url}")
            return text
        except Exception as e:
            logging.error(f"Error scraping website {url}: {e}")
            raise

    async def load_and_store_data(self, url):
        try:
            # Use asynchronous website scraping with Playwright
            scraped_text = await self.scrape_website(url)
            logging.info(f"Scraped data from {url}")

            # Split texts into chunks using RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
            document_chunks = text_splitter.create_documents([scraped_text])
            logging.info(f"Split text into {len(document_chunks)} chunks.")

            # Add text chunks to the vector database
            self.vector_db.add_documents(document_chunks)
            logging.info(f"Data successfully loaded and stored from {url}")
            return True
        except Exception as e:
            logging.error(f"Error loading and storing data: {e}")
            return False

    async def load_and_store_data_alternative(self, url):
        try:
            logging.info(f"Loading data from URL: {url}")
            loader = WebBaseLoader(url)
            documents = loader.load()
            logging.info(f"Loaded {len(documents)} documents from URL.")

            text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
            documents = text_splitter.split_documents(documents)
            logging.info(f"Split documents into {len(documents)} chunks.")

            # Use page_content attribute to extract text from documents
            document_texts = [doc.page_content for doc in documents]
            self.vector_db.add_documents(document_texts)
            logging.info(f"Data successfully stored in vector database from {url}")
            return True
        except Exception as e:
            logging.error(f"Error loading and storing data: {e}")
            return False

    async def ask_question(self, question, image_path=None):
        try:
            # Extract text from image if path is provided
            image_text = await self.extract_text_from_image(image_path) if image_path else ""
            combined_question = question + "\n" + image_text
            
            # Search for similarity in the vector database
            results = self.vector_db.max_marginal_relevance_search(combined_question, k=3, fetch_k=5)
            
            # Use contextual compression to focus on the most important parts
            compressor = LLMChainExtractor.from_llm(self.llm)
            compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=self.vector_db.as_retriever())
            
            # Make sure that `compression_retriever.invoke` is an async function
            compressed_docs = await compression_retriever.invoke(combined_question) if asyncio.iscoroutinefunction(compression_retriever.invoke) else compression_retriever.invoke(combined_question)
            context = "\n".join([doc.page_content for doc in compressed_docs])
            
            # Generate response (ensure `self.llm` is async if using await)
            response = await self.llm(combined_question) if asyncio.iscoroutinefunction(self.llm) else self.llm(combined_question)
            
            return {
                "response": response,
                "logs": {
                    "question": combined_question,
                    "context": context,
                    "relevant_documents": [doc.page_content[:100] for doc in results]
                }
            }
        except Exception as e:
            logging.error(f"Error while asking question: {e}")
            return {
                "response": "An error occurred while processing your request.",
                "logs": {
                    "question": combined_question if 'combined_question' in locals() else "",
                    "context": "",
                    "relevant_documents": []
                }
            }


    async def extract_text_from_image(self, image_path):
        try:
            logging.info(f"Extracting text from image: {image_path}")
            loader = UnstructuredImageLoader(image_path)
            data = loader.load()
            text = "\n".join([doc.page_content for doc in data])
            logging.info(f"Extracted text from image.")
            return text
        except Exception as e:
            logging.error(f"Error extracting text from image: {e}")
            raise
