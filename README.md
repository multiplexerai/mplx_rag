# mplx_rag
Complex RAG backend

GPT generated summery fir now

Initialization and Configuration: It sets up logging, initializes an asynchronous OpenAI client with API keys, configures Pinecone for vector database interactions, and establishes a Redis connection for session management.

API Router and Models: Defines an API router and Pydantic models for handling query and feedback requests.

Chat Start and Message Handling: Implements endpoints to start a chat session (generating a unique session token) and handle incoming messages. It leverages the OpenAI API for generating responses based on the chat history stored in Redis.

History Retrieval: Provides an endpoint to fetch chat history for a given session token, ensuring session validation through HTTP Bearer tokens.

Complex Query Processing: Details a complex querying process involving decomposing queries into simpler questions, querying a vector database (Pinecone), and aggregating responses to form a comprehensive answer. It includes functions for decomposing text, querying by type, and aggregating answers.

Logging and Validation: Includes functionality for logging the lifecycle of queries and validating given answers through decomposed components, utilizing asynchronous tasks for efficient processing.