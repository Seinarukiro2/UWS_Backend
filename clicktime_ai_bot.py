import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader, UnstructuredImageLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
import openai

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
            text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
            documents = text_splitter.split_documents(documents)

            # Use page_content attribute to extract text from documents
            document_texts = [doc for doc in documents]

            self.vector_db.add_documents(document_texts)
            return True
        except Exception as e:
            return False

    def extract_text_from_image(self, image_path):
        try:
            loader = UnstructuredImageLoader(image_path)
            data = loader.load()
            # Combine all extracted text from the image into one string
            text = "\n".join([doc.page_content for doc in data])
            return text
        except Exception as e:
            print(f"Error extracting text from image: {e}")
            return ""

    def ask_question(self, question, image_path=None):
        # Extract text from the image if provided
        image_text = self.extract_text_from_image(image_path) if image_path else ""

        # Combine question and image text
        combined_question = question + "\n" + image_text

        # Perform similarity search in the vector database
        results = self.vector_db.similarity_search(combined_question, k=1)
        if results:
            context = results[0].page_content

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a technical assistant for working with the ClickTime platform."},
                    {"role": "user", "content": f"Can you help me with this request? Answer in the same language as question - {combined_question}"},
                    {"role": "assistant", "content": context}
                ]
            )
            print(response)
            return response.choices[0].message.content
        else:
            return "No relevant information found."

# Example usage:
# bot = NodeInstallationBot()
# bot.load_and_store_data("https://teletype.in/@vibeloglazov/GaiaNet")
# response = bot.ask_question("Какую информацию я получаю командой gaianet info?", "path_to_image.png")
# print(response)
