"""
BM25 just searches for words in the whole document.
This system combines BM25 (keyword) and Dense (semantic) search with
cross-encoder re-ranking for optimal retrieval.

Usage:
1. First run: Automatically builds vocabulary if not found
2. Subsequent runs: Uses cached vocabulary for fast searches
"""
from huggingface_hub import InferenceClient
import numpy as np
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector
from typing import TypedDict, Optional
from collections import Counter
import pickle
import re
import os
load_dotenv()

router=FastAPI()

# This setting is added for CORS Issue
router.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # IMPORTANT: allows OPTIONS
    allow_headers=["*"]
)
# Reads .env file

class AnemicDiet(BaseModel):
    name: str = Field(description="Name of the food item or diet component recommended for anemia")
    iron_content: float = Field(description="Approximate iron content in milligrams (mg) per standard serving")

class StructuredOutput(BaseModel):
    guidelines: str = Field(description="Important instructions and recommendations to follow with the diet")
    diet: List[AnemicDiet] = Field(description="List of iron-rich foods or diet items suitable for people with anemia")

class AnemiaPayload(BaseModel):
    query: str

api = FastAPI()
DENSE_COLLECTION = "Diet_Stokesy"
SPARSE_COLLECTION = "Diet_Stokesy_BM25"
VOCABULARY_FILE = "bm25_vocabulary.pkl"

class GraphState(TypedDict):
    """State object that flows through the graph"""
    query: str
    result: Optional[str]
    meal_plan_bm25: Optional[dict]
    meal_plan_dense: Optional[dict]
    cross_encoder_results: Optional[dict]

qdrant_url=os.getenv("QDRANT_URL")
qdrant_api_key=os.getenv("QDRANT_API_KEY")
llm_api_key=os.getenv("GROQ_API_KEY")

# INITIALIZE MODELS AND CLIENTS (Global - loaded once)
model_id = "sentence-transformers/all-MiniLM-L6-v2"
client = InferenceClient(token=os.getenv("HF_TOKEN"))
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=llm_api_key,
)

system_prompt = SystemMessage(
    """You are a Diet Planner AI that creates only 100% vegetarian diet plans for people with anemia.    
Your role is to suggest iron-rich vegetarian meals for all times of the day, focusing on balanced nutrition and iron absorption.
Do not provide medical advice, diagnoses, supplements, or answer non-diet questions.
Keep all guidance simple, safe, and food-based."""
)

# VOCABULARY MANAGEMENT
def tokenize_text(text):
    """
    Break text into individual words (tokens).
    Example: "Hello World!" -> ["hello", "world"]
    """
    return re.findall(r'\w+', text.lower())

def build_vocabulary_from_collection(collection_name=SPARSE_COLLECTION):
    """
    Build vocabulary by extracting all unique words from the collection.
    Returns:
    - word_to_index dictionary
    """
    print("=" * 70)
    print("BUILDING VOCABULARY FROM COLLECTION")
    print("=" * 70)
    try:
        print(f"Fetching documents from {collection_name}...")
        all_texts = []
        offset = None
        # Scroll through all documents in the collection
        while True:
            result = qdrant_client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            points, next_offset = result
            
            for point in points:
                text = point.payload.get("text", "")
                if text:
                    all_texts.append(text)
            
            if next_offset is None:
                break
            
            offset = next_offset
        
        print(f"✓ Retrieved {len(all_texts)} documents\n")
        # Build vocabulary
        print("Building vocabulary...")
        all_words = []
        
        for text in all_texts:
            words = tokenize_text(text)
            all_words.extend(words)
        
        unique_words = sorted(set(all_words))
        word_to_index = {word: idx for idx, word in enumerate(unique_words)}
        
        print(f"✓ Vocabulary size: {len(word_to_index):,} unique words\n")
        # Save vocabulary
        print(f"Saving vocabulary to {VOCABULARY_FILE}...")
        with open(VOCABULARY_FILE, 'wb') as f:
            pickle.dump(word_to_index, f)
        print(f"✅ Vocabulary saved successfully!")
        print("=" * 70 + "\n")
        return word_to_index
    
    except Exception as error:
        print(f"❌ Error building vocabulary: {error}")
        import traceback
        traceback.print_exc()
        return None

def load_or_build_vocabulary():
    """
    Load vocabulary from file, or build it if it doesn't exist.
    Returns:
    - word_to_index dictionary
    """
    # Check if vocabulary file exists
    if os.path.exists(VOCABULARY_FILE):
        print("=" * 70)
        print("LOADING VOCABULARY FROM FILE")
        print("=" * 70)
        try:
            with open(VOCABULARY_FILE, 'rb') as f:
                word_to_index = pickle.load(f)
            
            print(f"✓ Loaded vocabulary with {len(word_to_index):,} words")
            print("=" * 70 + "\n")
            
            return word_to_index
        
        except Exception as error:
            print(f"⚠️  Error loading vocabulary: {error}")
            print("Building new vocabulary...\n")
            return build_vocabulary_from_collection()
    else:
        print(f"⚠️  Vocabulary file '{VOCABULARY_FILE}' not found")
        print("Building vocabulary for the first time...\n")
        return build_vocabulary_from_collection()

# Load vocabulary at startup
VOCABULARY = load_or_build_vocabulary()

if VOCABULARY is None:
    raise RuntimeError("Failed to load or build vocabulary. Cannot proceed.")

def create_query_sparse_vector(query_text):
    """
    Convert search query into BM25 sparse vector.
    
    Parameters:
    - query_text: User's search query
    
    Returns:
    - SparseVector for BM25 search
    """
    query_words = tokenize_text(query_text)
    word_frequencies = Counter(query_words)
    indices = []
    scores = []
    for word, frequency in word_frequencies.items():
        if word in VOCABULARY:
            index = VOCABULARY[word]
            indices.append(index)
            scores.append(float(frequency))

    if not indices:
        # Return minimal vector if no words match
        return SparseVector(indices=[0], values=[0.0])

    return SparseVector(indices=indices, values=scores)

# GRAPH NODES
def start_node(state: dict):
    """
    Initial node that processes the query.
    """
    query = state["query"]
    print(f"Query: {query}\n")
    return {
        "query": query,
        "result": None,
        "meal_plan_bm25": None,
        "meal_plan_dense": None,
        "cross_encoder_results": None
    }

def meal_plan_bm25_node(state: dict):
    query = state["query"]
    print("=" * 70)
    print("🔎 BM25 KEYWORD SEARCH")
    try:
        # Convert query to sparse vector
        query_sparse_vector = create_query_sparse_vector(query)
        print(f"✓ Query vector: {len(query_sparse_vector.indices)} matching words")
        # Search the BM25 collection
        search_results = qdrant_client.query_points(
            collection_name=SPARSE_COLLECTION,
            query=query_sparse_vector,
            using="bm25",
            limit=10,
            with_payload=True
        ).points
        # Extract results
        result_payloads = []
        result_texts = []
        result_score = []
        for result in search_results:
            result_score.append(result.score)
            result_payloads.append(result.payload)
            result_texts.append(result.payload.get("text", ""))
        print("Result_Payload_BM25-:",result_payloads)
        print(f"✓ Found {len(result_payloads)} BM25 results")
        
        if result_texts:
            preview = result_texts[0][:150] + "..." if len(result_texts[0]) > 150 else result_texts[0]
            print(f"  Top result: {preview}")
        
        print("=" * 70 + "\n")
        return {
            "meal_plan_bm25": {
                "list": result_payloads,
                "sparse_embedding_list": result_texts,
                "confidence_score_list":result_score
            }
        }
    
    except Exception as error:
        print(f"❌ BM25 search failed: {error}")
        import traceback
        traceback.print_exc()
                
        return {
            "meal_plan_bm25": {
                "list": [],
                "sparse_embedding_list": []
            }
        }

def meal_plan_dense_node(state: dict):
    query = state["query"]
    print("🔎 DENSE EMBEDDING SEARCH")    
    try:
        # Encode query to dense vector
        response = client.feature_extraction(query, model=model_id)
        embedding = np.array(response)

        # If token embeddings → mean pool
        if embedding.ndim == 2:
            embedding = embedding.mean(axis=0)

        # If already sentence embedding → use directly
        query_vector = embedding.tolist()
        print("✓ Query encoded to dense vector")
    
        # Search the dense collection
        search_results = qdrant_client.query_points(
            collection_name=DENSE_COLLECTION,
            query=embedding,
            limit=10,
            with_payload=True
        ).points
        
        # Extract results
        result_payloads = []
        result_texts = []
        result_score=[]
        for result in search_results:
            result_score.append(result.score)
            result_payloads.append(result.payload)
            result_texts.append(result.payload.get("text", ""))
        print("Result_Payload-:",result_payloads)
        print(f"✓ Found {len(result_payloads)} dense embedding results")
        
        if result_texts:
            preview = result_texts[0][:150] + "..." if len(result_texts[0]) > 150 else result_texts[0]
            print(f"  Top result: {preview}")
        
        print("=" * 70 + "\n")
        
        return {
            "meal_plan_dense": {
                "list": result_payloads,
                "dense_embedding_list": result_texts,
                "confidence_score_list":result_score
            }
        }
    
    except Exception as error:
        print(f"❌ Dense search failed: {error}")
        import traceback
        traceback.print_exc()
        print("=" * 70 + "\n")
        
        return {
            "meal_plan_dense": {
                "list": [],
                "dense_embedding_list": []
            }
        }

def cross_encoder_node(state: dict):
    query = state["query"]
    print("🔄 CROSS-ENCODER RE-RANKING")
    # Get results from both searches
    sparse_texts = state.get("meal_plan_bm25", {}).get("sparse_embedding_list", [])
    sparse_score = state.get("meal_plan_bm25",{}).get("confidence_score_list", [])
    dense_texts = state.get("meal_plan_dense", {}).get("dense_embedding_list", [])
    dense_score= state.get("meal_plan_dense", {}).get("confidence_score_list", [])

    w_sparse = 0.6
    w_dense  = 0.4

    final_scores = [
        w_sparse * s + w_dense * d
        for s, d in zip(sparse_score, dense_score)
    ]

    ranked_docs = sorted(
        enumerate(final_scores),
        key=lambda x: x[1],
        reverse=True
    )
    top_indices = [idx for idx, score in ranked_docs[:2]]
    top_texts = []
    for i in top_indices:
        top_texts.append(sparse_texts[i])
        top_texts.append(dense_texts[i])

    return {
        "cross_encoder_results": {
            "top_texts": top_texts,
        }
    }

def llm_qa_node(state: dict):
    query = state["query"]
    top_texts = state.get("cross_encoder_results", {}).get("top_texts", [])
    print("🤖 GENERATING ANSWER WITH LLM")
    print("=" * 70 + "\n")
    
    if not top_texts:
        print("⚠️  No context available, generating general response...")
        context = "No specific information found in the database."
    else:
        # Build context from retrieved documents
        context = "\n\n".join([
            f"Context {i+1}:\n{text}" 
            for i, text in enumerate(top_texts)
        ])
        print(f"Using {len(top_texts)} context documents\n")
    
    # Create prompt with context
    messages = [
        system_prompt,
        HumanMessage(
            content=f"""Based on the following relevant information from our nutrition database, please answer the user's query.
                    Retrieved Context:{context}
                    User Query: {query}
                    Please provide a personalized, practical response based on the context and your expertise."""
            )
    ]
    structured_llm=llm.with_structured_output(StructuredOutput)
    response=structured_llm.invoke(messages)
    # print("Response:", response)
    result=response
    # print("✅ FINAL ANSWER")
    # print(result)
    # print("=" * 70 + "\n")
    return {
        "result": result
    }

workflow_graph = StateGraph(GraphState)
workflow_graph.add_node("start", start_node)
workflow_graph.add_node("bm25_search", meal_plan_bm25_node)
workflow_graph.add_node("dense_search", meal_plan_dense_node)
workflow_graph.add_node("rerank", cross_encoder_node)
workflow_graph.add_node("generate_answer", llm_qa_node)
workflow_graph.add_edge("start", "bm25_search")
workflow_graph.add_edge("start", "dense_search")
workflow_graph.add_edge("bm25_search", "rerank")
workflow_graph.add_edge("dense_search", "rerank")
workflow_graph.add_edge("rerank", "generate_answer")
workflow_graph.set_entry_point("start")
app = workflow_graph.compile()

def search_and_answer(user_query: str):
    print("🚀 STARTING RAG PIPELINE")
    result_state = app.invoke({
        "query": user_query
    })
    print("Meal_Plan",result_state["meal_plan_dense"])
    print("=" * 70 + "\n")
    print("Meal_Plan_BM25",result_state["meal_plan_bm25"])
    return result_state

@router.get("/")
def response():
    return {
        "content":"The name's William Butcher, pro in disposing of Shitbag supes."
    }

@router.post("/query/")
def answer(payload:AnemiaPayload):
    question=payload.query
    answer=search_and_answer(question)
    return answer["result"]

# if __name__ == "__main__":
#     user_query = input("\nEnter your question: ").strip()
#     if user_query:
#         final_state = search_and_answer(user_query)
#         print("📊 PIPELINE SUMMARY")
#         print(f"Query: {final_state['query']}")
#         print(f"BM25 results: {len(final_state.get('meal_plan_bm25', {}).get('sparse_embedding_list', []))}")
#         print(f"Dense results: {len(final_state.get('meal_plan_dense', {}).get('dense_embedding_list', []))}")
#         print(f"Re-ranked results: {len(final_state.get('cross_encoder_results', {}).get('top_texts', []))}")
#         print(f"Answer generated: {'Yes' if final_state.get('result') else 'No'}")