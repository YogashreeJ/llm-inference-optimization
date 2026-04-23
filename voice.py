import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
import torch
import tempfile
from gtts import gTTS
from playsound import playsound
from deep_translator import GoogleTranslator
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from langchain_ollama import OllamaEmbeddings



# === Load Embedding Model ===
embedding_model = OllamaEmbeddings(model="bge-m3")

# === Load Chroma DB ===
vector_db = Chroma(
    persist_directory=r"D:\MinorProj\thirukkural_bge_db",
    embedding_function=embedding_model,
    collection_name="thirukkural-bge-m3"
)
retriever = vector_db.as_retriever(search_kwargs={"k": 4})

# === Base LLM for RAG ===
llm = ChatOllama(model="mistral")

# === Unified RAG Prompt ===
prompt = ChatPromptTemplate.from_template("""
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

# === RAG Chain ===

rag_inputs = {
    "context": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"],
    "language_instruction": lambda x: x["language_instruction"]
}

rag_chain = rag_inputs | prompt | llm | StrOutputParser()

# === Text-to-Speech ===
def speak_text(text, lang_code):
    try:
        import pygame
        import time
        # Use an absolute path as playsound on Windows often fails with relative paths
        filename = os.path.abspath("tts_output.mp3")
        tts = gTTS(text=text, lang=lang_code)
        tts.save(filename)
        
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.music.unload()
            pygame.mixer.quit()
        except Exception as e:
            print(f"pygame playback failed: {e}. Trying fallback...")
            # Fallback to default media player if pygame fails
            import platform
            if platform.system() == "Windows":
                os.system(f'start "" "{filename}"')
            else:
                os.system(f'afplay "{filename}" &')
            
        try:
            time.sleep(1) # Give it a moment to unlock
            if os.path.exists(filename):
                os.remove(filename)  # Optional cleanup
        except OSError:
            pass  # Ignore if the file is locked by the media player
    except Exception as e:
        print(f"⚠️ Speech failed: {e}")

LANG_TO_GTT_LANG = {
    "1": "ta",  # Tamil
    "2": "en",  # Thanglish
    "3": "hi",  # Hindi
    "4": "en"   # English
}

# Tamil uses English internally -> Google Translate for high quality output
LANGUAGE_INSTRUCTIONS = {
    "1": "English language",  # will be Google-translated to Tamil after generation
    "2": "CRITICAL: Respond ENTIRELY in Thanglish (Tamil language transliterated in the English alphabet). DO NOT use English sentences. Mix Tamil words written in English letters with GenZ slangs (e.g., 'macha', 'bro', 'da', 'vibe', 'romba'). Example: 'Kavala padatha macha, ellam seri aagidum.' DO NOT USE TAMIL SCRIPT.",
    "3": "Hindi language (हिन्दी)",
    "4": "English language"
}


def translate_to_tamil(text: str) -> str:
    """Translate English text to Tamil using Google Translate."""
    try:
        chunk_size = 4500
        if len(text) <= chunk_size:
            return GoogleTranslator(source='en', target='ta').translate(text)
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
        return " ".join(
            GoogleTranslator(source='en', target='ta').translate(chunk)
            for chunk in chunks
        )
    except Exception as e:
        print(f"⚠️ Translation error: {e}")
        return text

# === User Interaction ===
if __name__ == "__main__":
    print("Choose Output Language:")
    print("1 - Tamil\n2 - Thanglish\n3 - Hindi\n4 - English")
    lang = input("Enter your choice (1-4): ").strip()
    query = input("\n🧠 Ask your question (English or Thanglish):\n> ")

    instruction = LANGUAGE_INSTRUCTIONS.get(lang, "English language")

    print("\n💬 I understand what you are feeling right now:\n")

    full_response = ""
    if lang == "2":  # Thanglish
        print("🔄 Generating GenZ Thanglish response...")
        rag_output = rag_chain.invoke({"question": query, "language_instruction": "English language"})
        genz_rewriter_prompt = ChatPromptTemplate.from_template("""
You are a helpful Gen-Z friend. Rewrite the following explanation of a Thirukkural so it sounds like a casual conversation with a friend in 'Thanglish' (Tamil words written in English letters). 
Use GenZ slang like 'macha', 'bro', 'da', 'vibe', 'verithanam', 'romba'. 
Make it relatable, empathetic, and conversational.
DO NOT use Tamil script. Keep the original Thirukkural in Tamil script if it's there, but the explanation MUST be in Thanglish.

Original Text:
{text}

Rewritten Gen-Z Thanglish Text:
""")
        rewriter = genz_rewriter_prompt | llm | StrOutputParser()
        for chunk in rewriter.stream({"text": rag_output}):
            print(chunk, end="", flush=True)
            full_response += chunk
    else:
        for chunk in rag_chain.stream({"question": query, "language_instruction": instruction}):
            if lang != "1":  # Stream to console for non-Tamil
                print(chunk, end="", flush=True)
            full_response += chunk

        if lang == "1":  # Tamil: translate then print
            print("🔄 Translating to Tamil...")
            full_response = translate_to_tamil(full_response)
            print(full_response)

    print("\n")

    # === Speak the response ===
    speak_text(full_response, LANG_TO_GTT_LANG.get(lang, "en"))
