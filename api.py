from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel
from typing import Optional
from main import NodeInstallationBot  # Импортируйте ваш класс бота
import os
import sqlite3
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify allowed origins like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализируем бот
bot = NodeInstallationBot()

class QuestionRequest(BaseModel):
    question: str
    image_path: Optional[str] = None

class LoadDataRequest(BaseModel):
    url: str

DATABASE_PATH = "db/chroma.sqlite3"

@app.get("/tables/")
async def get_tables():
    # Connect to the SQLite database
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Query to get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Close the connection
    conn.close()

    # Return the list of table names
    return {"tables": [table[0] for table in tables]}


@app.post("/load_data/")
async def load_data(request: LoadDataRequest):
    url = request.url
    if not url:
        raise HTTPException(status_code=400, detail="URL parameter is required.")
    
    try:
        success = await bot.load_and_store_data(url)
        if success:
            return {"message": "Data successfully loaded and stored."}
        else:
            raise HTTPException(status_code=500, detail="Failed to load data.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    try:
        response = await bot.ask_question(request.question, request.image_path)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary location
        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file.file.read())
        
        # Extract text from the uploaded image
        text = bot.extract_text_from_image(temp_file_path)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)  # Clean up the temporary file
