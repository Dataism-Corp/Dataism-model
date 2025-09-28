
# core.py
# Dateria Core Functions (Phase 2 foundation)

import datetime

# Memory store (for now, just in memory — later we’ll expand to JSON/DB)
conversation_history = []

def add_to_memory(user_input, bot_response):
    """Save conversation turns to memory."""
    timestamp = datetime.datetime.now().isoformat()
    conversation_history.append({
        "time": timestamp,
        "user": user_input,
        "dateria": bot_response
    })

def get_memory():
    """Return all stored conversation history."""
    return conversation_history

def generate_response(user_input):
    """
    Very basic response generator (placeholder).
    Later this will connect to model inference.
    """
    if "hello" in user_input.lower():
        return "Hello, this is Dateria Core online."
    elif "time" in user_input.lower():
        return f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')}."
    else:
        return "I hear you. Phase 2 core is active."

def chat(user_input):
    """Process input, generate reply, and store in memory."""
    response = generate_response(user_input)
    add_to_memory(user_input, response)
    return response
