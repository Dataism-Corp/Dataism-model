from fastapi import APIRouter, Request
import re, json, asyncio

router = APIRouter()

async def call_chat_model(prompt):
    # Replace with real call
    return f"ChatModel: {prompt}"

async def call_code_model(prompt):
    return f"CodeModel: {prompt}"

@router.post("/route")
async def route_request(request: Request):
    body = await request.json()
    prompt = body.get("prompt","")
    if re.search(r"def |class |for |while|import", prompt):
        resp = await call_code_model(prompt)
    else:
        resp = await call_chat_model(prompt)
    return {"response": resp}
