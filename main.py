from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel
import os

app = FastAPI(title="AI Chat API")

# السماح للتطبيق بالتواصل مع الخادم
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# مفتاح Groq (سيتم إضافته من Render لاحقاً)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "ضع_مفتاحك_هنا")

client = Groq(api_key=GROQ_API_KEY)

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        messages = [
            {
                "role": "system",
                "content": "أنت مساعد ذكي ومفيد. أجب دائماً باللغة العربية بوضوح."
            }
        ]
        
        for msg in request.history:
            messages.append(msg)
        
        messages.append({
            "role": "user",
            "content": request.message
        })
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        
        return {"reply": response.choices[0].message.content}
    
    except Exception as e:
        return {"reply": f"عذراً، حدث خطأ: {str(e)}"}

@app.get("/")
async def root():
    return {"message": "الخادم يعمل! 🎉"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
