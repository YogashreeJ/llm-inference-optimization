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
        # Use an absolute path as playsound on Windows often fails with relative paths
        filename = os.path.abspath("tts_output.mp3")
        tts = gTTS(text=text, lang=lang_code)
        tts.save(filename)
        
        try:
            from playsound import playsound
            playsound(filename)
        except Exception as e:
            print(f"playsound failed: {e}. Trying fallback...")
            # Fallback to default media player if playsound fails
            import platform
            if platform.system() == "Windows":
                os.system(f'start "" "{filename}"')
            else:
                os.system(f'afplay "{filename}" &')
            
        try:
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
    "2": "Casual Thanglish (Tamil language written in the English alphabet. Do NOT use Tamil script. Use GenZ conversational words like 'macha', 'bro', 'romba').",
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
