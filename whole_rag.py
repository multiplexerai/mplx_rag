from datetime import datetime
import csv
import logging
from fastapi import FastAPI, status, HTTPException, Header, Depends, Request, APIRouter
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import openai
from openai import AsyncOpenAI
import pinecone
import redis
import json
import asyncio
from uuid import uuid4
from typing import List, Tuple, Dict

# Constants
from keys import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENV
MODEL = "YOUR_EMBEDDER"
INDEX_NAME = 'YOUR_INDEX_NAME'

# Configure logging at the beginning of your script or main function
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Instantiate Async OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Configure Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
index = pinecone.Index(INDEX_NAME)

# Initialize the Redis client with a connection pool
pool = redis.ConnectionPool(host='localhost', port=6379, db=0)
r = redis.Redis(connection_pool=pool)

api_router = APIRouter()

app = FastAPI()
security = HTTPBearer()

class QueryRequest(BaseModel):
    query_text: str

class FeedbackRequest(BaseModel):
    userInput: str
    response: str

@api_router.get("/start_chat")
def start_chat():
    session_token = str(uuid4())
    r.set(session_token, json.dumps([]))
    return {"session_token": session_token}

@api_router.get("/send_message/{message}")
async def send_message(message: str, session_token: str):
    logging.info(f"Received message from frontend: {message}")  # Log the incoming message

    if not r.exists(session_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session token"
        )

    chat_history = json.loads(r.get(session_token) or '[]')
    if len(chat_history) > 0:
        chat_string_history = [item['content'] for item in chat_history]
        prompt = " ".join(chat_string_history) + " " + message
        logging.info(f"Modified query with chat history: {prompt}")  # Log the modified query
        
        system_message = {"role": "system", "content": "Please rephrase the user's question based on the given conversation history."}
        user_message = {"role": "user", "content": prompt}
        messages = [system_message, user_message]
        response = await client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        temperature=0.4,
                        max_tokens=200
                    )
        if response is None:
            return {"response": "GPT API request timed out"}
        rephrased_message = response.choices[0].message.content
        chat_history.append({"role": "assistant", "content": rephrased_message})
        r.set(session_token, json.dumps(chat_history))
    else:
        chat_history.append({"role": "user", "content": message})
        r.set(session_token, json.dumps(chat_history))
    return {"response": "message received"}

@app.get("/get_history")
def get_history(token: HTTPAuthorizationCredentials = Depends(security)):
    session_token = token.credentials
    if not r.exists(session_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session token"
        )
    return {"history": json.loads(r.get(session_token) or '[]')}


@api_router.get("/{vectordb}_query/{query_text}")
async def query_vectordb(vectordb: str, query_text: str, token: HTTPAuthorizationCredentials = Depends(security)):
    logging.info(f"Received query for {vectordb}: {query_text}")  # Log the incoming query
    
    session_token = token.credentials
    if not r.exists(session_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session token")

    # Decomposing complex question into simpler questions
    simple_questions = await decompose_complex_question(query_text, vectordb)
    
    # Processing each simple question to get answers
    simple_answers_tasks = [vectordb_type_query(vectordb, sq, session_token) for sq in simple_questions]
    simple_answers_results = await asyncio.gather(*simple_answers_tasks)
    simple_answers = [{"question": sq, "answer": result} for sq, result in zip(simple_questions, simple_answers_results)]
    
    # Aggregating answers to form a final response
    final_answer = await aggregate_answers(query_text, simple_answers, session_token)

    # Log the entire lifecycle of a query in a single JSON object
    log_full_lifecycle(vectordb, query_text, simple_questions, simple_answers, final_answer)

    logging.info(f"Final answer: {final_answer}")

    return {"response": final_answer}

# Helper function for logging the full lifecycle of a query
def log_full_lifecycle(vectordb: str, complex_question: str, simple_questions: list, simple_answers: list, complex_answer: str):
    lifecycle_entry = {
        "vectordb": vectordb,
        "complex_question": complex_question,
        "simple_questions": simple_questions,
        "simple_answers": simple_answers,
        "complex_answer": complex_answer
    }
    with open(f"{vectordb}_query_log.jsonl", 'a', encoding='utf-8') as log_file:
        json.dump(lifecycle_entry, log_file)
        log_file.write('\n')  # Write a new line to separate JSON objects


#for validating the given answer 
@api_router.get("/{vectordb}_validate/{query_text}")
async def query_validate(vectordb: str, query_text: str, token: HTTPAuthorizationCredentials = Depends(security)):
    # Log the function call
    logging.info(f"query_validate called with vectordb: {vectordb} and query_text: {query_text}") #somehow adding logging fixed my issue or 404 error?
    
    session_token = token.credentials
    if not r.exists(session_token):
        logging.warning("Invalid session token") # see above for somehow fixing error
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session token")

    simple_questions = await decompose_complex_answer(query_text, vectordb)
    simple_answers_tasks = [vectordb_type_query(vectordb, sq, session_token) for sq in simple_questions]
    simple_answers_results = await asyncio.gather(*simple_answers_tasks)
    simple_answers = [{"question": sq, "answer": result} for sq, result in zip(simple_questions, simple_answers_results)]

    final_answer = await validate_answer(query_text, simple_answers, session_token)

    return {"response": final_answer}

app.include_router(api_router, prefix="/api")

async def decompose_text(text: str, namespace: str, system_message_content: str) -> List[str]:
    system_message = {"role": "system", "content": system_message_content}
    user_message = {"role": "user", "content": text}
    messages = [system_message, user_message]

    response = await client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=messages,
        temperature=0.4,
        max_tokens=250  # Adjust as needed
    )
    if response is None:
        return ["GPT API request timed out"]

    decomposed_text = response.choices[0].message.content.strip().split('\n')

    return decomposed_text

# Updated functions using the new utility function
async def decompose_complex_question(question: str, namespace: str) -> List[str]:
    system_message_content = f"To optimize for RAG, select the best 2 to 4 questions that best deconstruct the complex question about the {namespace}."
    return await decompose_text(question, namespace, system_message_content)

async def decompose_complex_answer(answer: str, namespace: str) -> List[str]:
    system_message_content = "We want to validate the answer given here, break it down to several texts to best semantic search."
    return await decompose_text(answer, namespace, system_message_content)



async def vectordb_type_query(vectordb: str, query_text: str, session_token: str):
    namespaces = [f'{vectordb}n', f'{vectordb}q', f'{vectordb}s']
    all_matches_for_logging = []
    for namespace in namespaces:
        if r.exists(session_token) and len(json.loads(r.get(session_token) or '[]')) > 0:
            matches = await query(query_text, namespace, top_k=3, session_token=session_token)
        else:
            matches = await query(query_text, namespace, top_k=3)
        all_matches_for_logging.extend(matches)
    combined_matches_text = " ".join(all_matches_for_logging)
    prompt = f"In the following data, find the relevant info to answer: {query_text}\n{combined_matches_text}"
    response = await get_gpt_response(prompt, session_token)
    log_file = f"{vectordb}_query_response_log.csv"
    with open(log_file, 'a', newline='', encoding='utf-8') as log_file:
        writer = csv.writer(log_file)
        writer.writerow([prompt, response])
    return {"response": response}

async def query(query_text: str, namespace, top_k=2, session_token=None):
    if session_token is not None and r.exists(session_token):
        query_text = json.loads(r.get(session_token) or '[]')[-1]['content']
    response = await client.embeddings.create(input=[query_text], model=MODEL)
    xq = response.data[0].embedding
    res = index.query([xq], top_k=top_k, include_metadata=True, namespace=namespace)
    matches_list = [match['metadata']['text'] for match in res['matches']]
    return matches_list

async def get_gpt_response(prompt, session_token: str = Header(None)):
    if session_token is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session token must be provided"
        )
   
    
    system_message = {"role": "system", "content": "Please find the revelvent info and provide a concise and accurate answer based on the information provided."}
    messages = [system_message, {"role": "user", "content": prompt}]

    response = await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.4,
                    max_tokens=300
                )
    if response is None:
        return "GPT API request timed out"
    
    assistant_message = response.choices[0].message.content
        
    return assistant_message

async def process_answer(complex_text: str, simple_qas: List[Dict[str, str]], session_token: str, purpose: str = "aggregate") -> str:
    # Construct the prompt with the complex text and the Q&A pairs
    qa_text = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in simple_qas])
    prompt = f"Given the following questions and answers, provide a concise explanation for the complex question.\n\n{qa_text}\n\nComplex Question: {complex_text}\nAnswer:"

    # Generate the response using your preferred GPT model
    response = await client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Model name can be parameterized as needed
                    messages=[{"role": "system", "content": purpose}, {"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=500
                )

    if response is None:
        return "GPT API request timed out"
    
    aggregated_answer = response.choices[0].message.content.strip()

    # Append this complex Q&A to the existing chat history in Redis
    try:
        chat_history = json.loads(r.get(session_token) or '[]')
        chat_history.append({"role": "user", "content": complex_text})
        chat_history.append({"role": "assistant", "content": aggregated_answer})
        r.set(session_token, json.dumps(chat_history), ex=3600)  # Adjust expiry as needed
    except redis.exceptions.ConnectionError:
        logging.error("Failed to update chat history in Redis due to connection error.")

    return aggregated_answer

# Updated functions using the new utility function
async def aggregate_answers(complex_question: str, simple_qas: List[Dict[str, str]], session_token: str):
    return await process_answer(complex_question, simple_qas, session_token, "aggregate")

async def validate_answer(complex_question: str, simple_qas: List[Dict[str, str]], session_token: str):
    return await process_answer(complex_question, simple_qas, session_token, "validate")

#
#@app.post("/feedback_log/{vectordb}")
#async def vectordb_feedback_log(vectordb: str, request: FeedbackRequest):
#    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#    log_file = f"help_{vectordb}_query_response_log.csv"
#    with open(log_file, 'a', newline='') as log_file:
#        writer = csv.writer(log_file)
#        writer.writerow([now, request.userInput, request.response])
#    return {"message": "Feedback logged"}