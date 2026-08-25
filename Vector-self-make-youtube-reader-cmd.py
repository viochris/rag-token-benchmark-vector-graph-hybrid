# ==============================================================================
# 0. REQUIRED LIBRARIES & DEPENDENCIES
# ==============================================================================
# Standard library imports for system operations, logging, and asynchronous execution
import os
import sys
import logging
import traceback
import asyncio
from urllib.parse import urlparse, parse_qs

# Third-party utility libraries for environment management, tokenization, and API interaction
import chromadb
import nest_asyncio
import litellm
import tiktoken
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

# LlamaIndex core components for document modeling, vector storage, and application context
from llama_index.core import (
    Settings,
    Document,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler, LlamaDebugHandler
from llama_index.core.callbacks.schema import EventPayload
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.memory import Memory

# LlamaIndex integrations for specific external services (ChromaDB, Google GenAI, LiteLLM)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.litellm import LiteLLM

# Phoenix OpenTelemetry instrumentation for distributed tracing and performance observability
from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

# ==============================================================================
# 1. ENVIRONMENT CONFIGURATION
# ==============================================================================
# Apply nested asyncio to permit the execution of asynchronous event loops within 
# environments that already manage their own loops (e.g., Jupyter, interactive terminals).
nest_asyncio.apply()

# Suppress verbose underlying library logs (such as HTTP requests and raw API debugs) 
# to maintain a clean, readable Command Line Interface (CLI) for the end user.
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

# Securely load environment variables from the local .env file.
load_dotenv()

# Validate the presence of critical API keys before proceeding to prevent late runtime failures.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("🚨 [SYSTEM ERROR] GOOGLE_API_KEY is missing. Please check your .env file.")

PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY")
if not PHOENIX_API_KEY:
    raise ValueError("🚨 [SYSTEM ERROR] PHOENIX_API_KEY is missing. Please check your .env file.")

# Configure Phoenix OpenTelemetry routing parameters to ingest telemetry traces into the cloud.
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com/s/viochristian12/v1/traces")

# Define the target video URLs for data ingestion.
youtube_urls = ["https://www.youtube.com/watch?v=i3OYlaoj-BM"]

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def load_youtube_transcripts(youtube_urls: list[str]) -> list[Document]:
    """
    Parses YouTube URLs, extracts their subtitles utilizing the YouTubeTranscriptApi, 
    and formats the resulting text into LlamaIndex Document objects with relevant metadata.
    """
    print("📥 [SYSTEM] Fetching YouTube transcripts...")
    ytt_api = YouTubeTranscriptApi()
    documents = []

    for url in youtube_urls:
        # Parse the video ID dynamically to support both standard and shortened YouTube URL formats.
        if "youtube" in url:
            video_id = url.split("/")[-1].split("v=")[-1].split("&")[0]
        elif "youtu.be" in url:
            video_id = url.split("/")[-1].split("?")[0]
        else:
            parsed_url = urlparse(url=url)

            if "v" in parse_qs(parsed_url.query):
                print(f"🔍 [SYSTEM] Extracting video ID from URL query parameters: {url}")
                video_id = parse_qs(parsed_url.query)["v"][0]
            elif parsed_url.netloc in ["youtu.be"]:
                print(f"🔍 [SYSTEM] Extracting video ID from shortened URL: {url}")
                video_id = parsed_url.path.lstrip("/")
            else:
                raise ValueError(f"🚨 [SYSTEM ERROR] Invalid or unsupported YouTube URL format: {url}")

        # Retrieve the transcript chunks and compile them into a single cohesive text block.
        fetched_transcript = ytt_api.fetch(video_id=video_id)
        full_text = " ".join([snippet.text for snippet in fetched_transcript])

        # Encapsulate the raw text and contextual metadata into a LlamaIndex Document instance.
        documents.append(Document(text=full_text.strip(), metadata={"video_id": video_id, "url": url}))

    return documents

try:
    # ==============================================================================
    # 3. SYSTEM SETTINGS & OBSERVABILITY
    # ==============================================================================
    print("⚙️ [SYSTEM] Configuring LlamaIndex Settings (LLM & Embeddings)...")

    # Initialize Token Counter utilizing the OpenAI cl100k_base encoding standard.
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.get_encoding("cl100k_base").encode,
        verbose=False
    )
    
    # Initialize the debug handler to trace LLM inputs and outputs for cost calculations.
    debug_handler = LlamaDebugHandler(print_trace_on_end=True)
    
    # Establish the global Language Model utilizing LiteLLM as a unified proxy router, 
    # equipped with fallback redundancy to ensure high availability.
    Settings.llm = LiteLLM(
        model="gemini/gemini-2.5-flash",
        temperature=0.3,
        # additional_kwargs={
        #     "fallbacks": ["gemini/gemini-3.6-flash", "groq/openai/gpt-oss-20b"],
        #     "drop_params": True,
        #     "num_retries": 2
        # }
    )
    
    # Establish the global Semantic Embedding Model.
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="models/gemini-embedding-2", 
        api_key=GOOGLE_API_KEY,
        embed_batch_size=50
    )
    
    # Configure the document parsing strategy to segment large texts into manageable embedding nodes.
    Settings.node_parser = SentenceSplitter(
        chunk_size=2048,
        chunk_overlap=200
    )

    # Attach the observability callbacks to the global Settings manager.
    Settings.callback_manager = CallbackManager([token_counter, debug_handler])

    print("📊 [SYSTEM] Initializing Phoenix Observability & Tracing...")
    
    # Register the OpenTelemetry provider to route telemetry data to Arize Phoenix Cloud.
    tracer_provider = register(
        project_name="youtube_vector_project",
        auto_instrument=True
    )

    # Instrument LlamaIndex to dispatch tracing data automatically during execution.
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

    print("⚙️ [SYSTEM] Initializing Cross-Encoder Reranker...")
    
    # Configure the cross-encoder model to refine and re-score the top retrieved nodes 
    # to maximize context relevancy before generating the final answer.
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2", 
        top_n=3
    )

    # ==============================================================================
    # 4. DATABASE INITIALIZATION (CHROMA CLOUD)
    # ==============================================================================
    print("⏳ [SYSTEM] Initializing ChromaDB Cloud Client...")
    
    # Initialize the ChromaDB vector storage client utilizing cloud credentials.
    chroma_client = chromadb.CloudClient(api_key=os.getenv("CHROMA_API_KEY"))
    
    # Retrieve or create the designated collection intended for transcript vectors.
    chroma_collection = chroma_client.get_or_create_collection("youtube_docs")

    # Instantiate the ChromaVectorStore adapter to bridge ChromaDB with LlamaIndex's storage context.
    print("🔌 [SYSTEM] Attaching ChromaVectorStore adapter...")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # ==============================================================================
    # 5. DATA INGESTION & INDEXING
    # ==============================================================================
    # Assess the current state of the database to determine if embedding generation is necessary,
    # preventing redundant API calls.
    if chroma_collection.count() == 0:
        print("⏳ [SYSTEM] ChromaDB is empty. Extracting transcripts from provided YouTube URLs...")
        
        documents = load_youtube_transcripts(youtube_urls=youtube_urls)
        print(f"✅ [SYSTEM] Successfully loaded transcripts from {len(youtube_urls)} video(s).")
        
        print("⚙️ [SYSTEM] Creating Vector Store Index and embedding documents to ChromaDB Cloud...")
        index = VectorStoreIndex.from_documents(documents=documents, storage_context=storage_context)
        print("✅ [SYSTEM] Index successfully generated and stored in ChromaDB Cloud.")
    else:
        print("📂 [SYSTEM] Existing data found in ChromaDB. Bypassing extraction phase...")
        
        # Reconstruct the index mapping directly from the persisted cloud vector store.
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        print("✅ [SYSTEM] Index successfully loaded from cloud storage.")

    # ==============================================================================
    # 6. CHAT ENGINE & PERSISTENT MEMORY SETUP
    # ==============================================================================
    print("🧠 [SYSTEM] Initializing Chat Engine and persistent PostgreSQL memory...")
    
    # Establish a persistent conversational memory backend using AsyncPG to maintain user context across queries.
    memory = Memory.from_defaults(
        session_id="youtube_session", 
        token_limit=40000,
        async_database_uri=os.getenv("ASYNCPG_DATABASE_URL"),
        table_name="youtube_vector_memory"
    )

    # Define the operational boundaries and strict behavioral constraints for the AI Agent.
    system_prompt = (
        "You are an intelligent and analytical AI assistant. "
        "Your primary objective is to answer user queries strictly based on the provided YouTube video transcript context. "
        "If a question requires details from the video, provide clear and comprehensive explanations. "
        "If the answer cannot be found in the context, explicitly state that you do not know. Do not hallucinate."
    )

    # Instantiate the Context Chat Engine unifying retrieval, memory, reranking, and generation.
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        llm=Settings.llm,
        memory=memory,
        node_postprocessors=[reranker],
        system_prompt=system_prompt,
        similarity_top_k=5
    )
    print("✅ [SYSTEM] Chat Engine successfully integrated and ready for interaction.")

    # ==============================================================================
    # 7. INTERACTIVE ASYNCHRONOUS CHAT LOOP
    # ==============================================================================
    async def main():
        print("\n" + "="*70)
        print("💬 [SYSTEM] Interactive YouTube Reader Session Started. Type 'exit' to stop.")
        print("="*70)

        # Clear existing buffers prior to starting the session
        token_counter.reset_counts()
        debug_handler.flush_event_logs()

        while True:
            # Capture continuous user input from the terminal CLI.
            user_input = input("\n🗣️ [USER] You: ").strip()

            # Process session termination commands gracefully.
            if user_input.lower() in ["exit", "close", "out", "break", "keluar", "x"]:
                print("🛑 [SYSTEM] Terminating session. Goodbye!")
                break
                
            # Bypass engine execution if the input is entirely empty.
            if not user_input:
                print("⚠️ [SYSTEM] Input cannot be empty. Please ask a question.")
                continue
                
            # Execute the asynchronous query utilizing the vector context retrieval.
            print("🧠 [AI] Processing your query against the YouTube transcript context...")
            ai_response = await chat_engine.achat(user_input)

            # Initialize variables to compute inference costs based on LiteLLM's event payload.
            total_cost = 0.0
            model_used = "Unknown"
            
            # Retrieve tracked inputs and outputs from the LlamaDebugHandler.
            llm_events = debug_handler.get_llm_inputs_outputs()

            # Iterate through the captured LLM events safely to accumulate completion costs.
            for start_event, end_event in llm_events:
                response_obj = end_event.payload.get(EventPayload.RESPONSE)
                if response_obj is not None and getattr(response_obj, "raw", None):
                    response_raw = response_obj.raw
                    total_cost += litellm.completion_cost(completion_response=response_raw)
                    model_used = response_raw.get('model', 'Unknown')
                    
            # Validate the existence of an AI response to prevent downstream rendering errors.
            if not ai_response:
                print("⚠️ [SYSTEM] AI failed to generate a response or no relevant context found.")
                continue
                
            # Render the final synthesized AI response to the terminal output.
            print("-" * 70)
            print("🤖 [AI RESPONSE]:")
            print(ai_response)
            print("-" * 70)

            # Extract and display the underlying source nodes evaluated during the generation phase.
            source_nodes = ai_response.source_nodes
            print(f"\n🔍 [SOURCE NODES] Retrieved {len(source_nodes)} relevant nodes from the vector store:")

            for i, node in enumerate(source_nodes, 1):
                print(f"--- Node {i} (score: {node.score}) ---")
                print(f"📄 Text Context: {node.text}")
                print()

            # ==============================================================================
            # 8. TOKEN USAGE & COST INSPECTION
            # ==============================================================================
            print("\n📊 [LLAMAINDEX CALLBACK TOKEN OVERVIEW]:")
            print(f"Model Used        : {model_used}")
            print("------------------+")
            print(f"Prompt Tokens     : {token_counter.prompt_llm_token_count}")
            print(f"Completion Tokens : {token_counter.completion_llm_token_count}")
            print("------------------+")
            print(f"Total LLM Tokens  : {token_counter.total_llm_token_count}")
            print("=" * 70)

            print(f"\n💰 [COST ANALYSIS]:")
            print(f"Total Estimated Cost: ${total_cost:.6f} USD")
            print("=" * 70)

            # Reset token counters and flush debug logs to prevent event overlap in the next loop iteration.
            token_counter.reset_counts()
            debug_handler.flush_event_logs()
            
    # Execute the asynchronous event loop.
    if __name__ == "__main__":
        asyncio.run(main())

# ==============================================================================
# 9. EXCEPTION HANDLING & ERROR ROUTING
# ==============================================================================
except Exception as e:
    # Extract exception attributes to facilitate precise conditional error matching.
    error_type = type(e).__name__
    error_msg = str(e).lower()
    error_raw = str(e)

    print("\n" + "="*70)
    print("💥 [CRITICAL FAILURE] LlamaIndex execution aborted!")
    print("="*70)
    
    # Print the comprehensive traceback stack for deep developer debugging.
    print("🔍 [TRACEBACK LOG]:")
    traceback.print_exc()
    print("-" * 70)

    # Output a structured, human-readable root cause summary for CLI diagnostics.
    print("📌 [ERROR SUMMARY]:")
    
    # Catch Google API quota limitations or rate limits (HTTP 429).
    if error_type == "ResourceExhausted" or "quota" in error_msg or "429" in error_msg:
        print(f"🚨 [API ERROR] {error_type}: Google API quota exceeded or rate limited. Details: {error_raw}")
        
    # Catch data deficiency errors such as missing context or empty initialization data.
    elif error_type == "ValueError" and "empty" in error_msg:
        print(f"🚨 [DATA ERROR] {error_type}: A data source is empty or missing valid inputs. Details: {error_raw}")
        
    # Catch authentication rejections from LiteLLM or upstream providers.
    elif error_type == "InvalidArgument" or "api_key" in error_msg:
        print(f"🚨 [AUTH ERROR] {error_type}: Invalid API Key configuration. Details: {error_raw}")
        
    # Catch ChromaDB cloud connection or initialization failures.
    elif error_type in ["OperationalError", "DatabaseError", "InvalidCollectionException"] or "chroma" in error_msg:
        print(f"🚨 [DATABASE ERROR] {error_type}: ChromaDB Cloud connection failed. Details: {error_raw}")
        
    # Catch AsyncPG (PostgreSQL) persistent memory connection failures.
    elif "asyncpg" in error_msg or "postgres" in error_msg:
        print(f"🚨 [DATABASE ERROR] {error_type}: Failed to connect or write to the PostgreSQL memory database. Details: {error_raw}")
        
    # Catch OS-level file system or network permission blocks.
    elif error_type == "PermissionError":
        print(f"🚨 [ACCESS ERROR] {error_type}: System denied access. Details: {error_raw}")
        
    # Default fallback routing for unhandled or unexpected exceptions.
    else:
        print(f"🚨 [UNKNOWN ERROR] {error_type}: Unexpected failure. Details: {error_raw}")
        
    print("="*70 + "\n")
    
    # Terminate the application with a non-zero exit code to signal execution failure to the OS.
    sys.exit(1)