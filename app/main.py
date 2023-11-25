from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chat_models import ChatOllama  # Make sure this import matches your setup
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import requests  # For sending the response back to the external server

app = FastAPI()

# Initialize the ChatOllama model directly
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
chat_model = ChatOllama(model="llama2:7b", callback_manager=callback_manager)

# Pydantic model for receiving questions from the external server
class QuestionData(BaseModel):
    question: str

# Pydantic model for sending responses back to the external server
class ResponseData(BaseModel):
    answer: str

# Replace with the actual URL of the external server
EXTERNAL_SERVER_URL = "http://localhost:8080/"

# Endpoint to receive questions from the external server
@app.post("/receive_question/")
async def receive_question(data: QuestionData):
    try:
        # Use the chat_model to generate a response
        response = chat_model.call_as_llm(data.question)

        # Send the response back to the external server
        send_response_to_external_server(ResponseData(answer=response))

        return {"status": "Question processed and response sent"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Function to send the response back to the external server
def send_response_to_external_server(response_data: ResponseData):
    # Assuming the external server expects a POST request with JSON body
    headers = {"Content-Type": "application/json"}
    # Use `model_dump` instead of `dict`
    response = requests.post(EXTERNAL_SERVER_URL, json=response_data.model_dump(), headers=headers)
    
    # Handle potential errors in the response
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to send response to external server")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

