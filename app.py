"""
FastAPI backend for the Thirukkural Emotional Support Chatbot.
Exposes the RAG chatbot as an API with streaming support.
"""

import os
import json
import hashlib
from datetime import date
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from deep_translator import GoogleTranslator

from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─── App Setup ────────────────────────────────────────────────
app = FastAPI(title="Thirukkural Emotional Support Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Embedding + Vector DB ──────────────────────────────
embedding_model = OllamaEmbeddings(model="bge-m3")

vector_db = Chroma(
    persist_directory=r"D:\MinorProj\thirukkural_bge_db",
    embedding_function=embedding_model,
    collection_name="thirukkural-bge-m3",
)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# ─── LLM ─────────────────────────────────────────────────────
llm = ChatOllama(model="mistral")

# ─── Unified RAG Prompt ───────────────────────────────────────
rag_prompt = ChatPromptTemplate.from_template("""
You are an emotional therapist. Answer with love and care. Base your answer only on the Thirukkural context.
You MUST respond strictly in the following language/style: {language_instruction}

IMPORTANT: When quoting a Kural, you MUST include BOTH lines of the couplet completely. Never truncate or give only one line.

Format:
"As stated in Kural (Kural Number) from Adhigaaram (Chapter Name),\n[Full Kural - BOTH lines exactly as given in context].\n[Explanation]."

If no answer is found, say: "I don't have an answer for that."

Context:
{context}

Question: {question}
Answer:
""")

# ─── RAG Chain ────────────────────────────────────────────────
rag_chain = (
    {
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"],
        "language_instruction": lambda x: x["language_instruction"]
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# ─── Language Instructions ────────────────────────────────────
# Tamil uses English internally, then Google Translate for high quality output
LANGUAGE_INSTRUCTIONS = {
    "tamil": "English language",  # will be Google-translated to Tamil after generation
    "hindi": "Hindi language (हिन्दी)",
    "english": "English language",
    "thanglish": "CRITICAL: Respond ENTIRELY in Thanglish (Tamil language transliterated in the English alphabet). DO NOT use English sentences. Mix Tamil words written in English letters with GenZ slangs (e.g., 'macha', 'bro', 'da', 'vibe', 'romba'). Example: 'Kavala padatha macha, ellam seri aagidum.' DO NOT USE TAMIL SCRIPT."
}


def translate_to_tamil(text: str) -> str:
    """Translate English text to Tamil using Google Translate.
    Splits into chunks to avoid API limits (max ~5000 chars per request)."""
    try:
        chunk_size = 4500
        if len(text) <= chunk_size:
            return GoogleTranslator(source='en', target='ta').translate(text)
        # Split on sentence boundaries for large responses
        chunks = []
        current = ""
        for sentence in text.replace("\n", " \n ").split(". "):
            if len(current) + len(sentence) < chunk_size:
                current += sentence + ". "
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence + ". "
        if current:
            chunks.append(current.strip())
        translated_chunks = [
            GoogleTranslator(source='en', target='ta').translate(chunk)
            for chunk in chunks
        ]
        return " ".join(translated_chunks)
    except Exception as e:
        return f"[Translation error: {e}]\n\n" + text

# ─── Daily Thoughts ──────────────────────────────────────────
_thoughts = []

def _load_thoughts():
    global _thoughts
    if not _thoughts:
        with open(r"D:\MinorProj\final dataset.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        _thoughts = [
            {
                "kural": item["kural"],
                "meaning": item["meaning"],
                "adhigaaram": item["adhigaaram"],
                "kural_number": item["kural_number"],
            }
            for item in data
            if "meaning" in item and "kural" in item
        ]
    return _thoughts


def get_daily_thought():
    thoughts = _load_thoughts()
    today = date.today().isoformat()
    idx = int(hashlib.md5(today.encode()).hexdigest(), 16) % len(thoughts)
    return thoughts[idx]


# ─── API Models ───────────────────────────────────────────────
class ChatRequest(BaseModel):
    language: str
    message: str


# ─── Routes ───────────────────────────────────────────────────
@app.get("/api/thought")
async def daily_thought():
    return get_daily_thought()


@app.post("/api/tts")
async def tts(req: ChatRequest):
    lang_code = "ta" if req.language.lower() == "tamil" else "en"
    if req.language.lower() == "hindi":
        lang_code = "hi"
    
    from gtts import gTTS
    import io
    tts_obj = gTTS(text=req.message, lang=lang_code)
    mp3_fp = io.BytesIO()
    tts_obj.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return StreamingResponse(mp3_fp, media_type="audio/mpeg")

@app.post("/api/chat")
async def chat(req: ChatRequest):
    lang_key = req.language.lower().strip()
    instruction = LANGUAGE_INSTRUCTIONS.get(lang_key, LANGUAGE_INSTRUCTIONS["english"])
    is_tamil = lang_key == "tamil"
    is_thanglish = lang_key == "thanglish"

    if is_tamil:
        # For Tamil: collect full English response, then translate via Google Translate
        async def generate_tamil():
            try:
                full_response = ""
                async for chunk in rag_chain.astream({
                    "question": req.message,
                    "language_instruction": instruction
                }):
                    full_response += chunk
                # Translate the complete English response to Tamil
                tamil_response = translate_to_tamil(full_response)
                yield tamil_response
            except Exception as e:
                yield f"\n\n⚠️ Error: {str(e)}"
        return StreamingResponse(generate_tamil(), media_type="text/plain")
    elif is_thanglish:
        async def generate_thanglish():
            try:
                # First get the english RAG output
                rag_output = rag_chain.invoke({
                    "question": req.message,
                    "language_instruction": "English language"
                })
                # Then rewrite it
                genz_rewriter_prompt = ChatPromptTemplate.from_template("""
You are a highly empathetic Gen-Z friend. Your job is to comfort the user using Thanglish (Tamil words written in English letters). 
You will be given a Thirukkural and its explanation. 
You must output a highly conversational, comforting response in Thanglish, using Gen-Z slang (macha, bro, da, vibe, chill, feel pannatha, etc).

Rules:
1. Speak ENTIRELY in Thanglish. NO full English sentences. NO Tamil script (except for the Kural itself).
2. Start by comforting the user like a friend (e.g. "Macha, don't worry da...", "Bro, feel pannatha...").
3. Include the Thirukkural.
4. Explain the meaning in casual Thanglish like you're talking to a friend over coffee.

Original Text:
{text}

Your Gen-Z Thanglish Response:
""")
                rewriter = genz_rewriter_prompt | llm | StrOutputParser()
                async for chunk in rewriter.astream({"text": rag_output}):
                    yield chunk
            except Exception as e:
                yield f"\n\n⚠️ Error: {str(e)}"
        return StreamingResponse(generate_thanglish(), media_type="text/plain")
    else:
        # For other languages: stream tokens directly
        async def generate():
            try:
                async for chunk in rag_chain.astream({
                    "question": req.message,
                    "language_instruction": instruction
                }):
                    yield chunk
            except Exception as e:
                yield f"\n\n⚠️ Error: {str(e)}"
        return StreamingResponse(generate(), media_type="text/plain")


# ─── Serve Frontend ──────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ─── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
