from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

app = FastAPI()

# Initialize the ChatOllama model directly
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
chat_model = ChatOllama(model="llama2:7b", callback_manager=callback_manager)

# Pydantic model for receiving questions
class QuestionData(BaseModel):
    question: str

# Endpoint to receive questions and return the response
@app.post("/receive_question/")
async def receive_question(data: QuestionData):
    try:
        # Use the chat_model to generate a response
        response = chat_model.call_as_llm(data.question)
        return {"answer": response}  # Return response directly to the caller
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

