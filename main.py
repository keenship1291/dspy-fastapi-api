from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
import os
from anthropic import Anthropic
import dspy

# Load API key from environment variable
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")

# Custom Anthropic LM for DSPy
class CustomAnthropic(dspy.LM):
    def __init__(self, api_key):
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"
        self.kwargs = {"max_tokens": 1000}
        self.history = []
        
    def basic_request(self, prompt, **kwargs):
        """Core method that handles the actual API call"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', 1000),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            return "Error generating response"
    
    def __call__(self, prompt=None, messages=None, **kwargs):
        """Handle different calling patterns DSPy might use"""
        if prompt is None and messages is not None:
            # Handle messages format
            if isinstance(messages, list) and len(messages) > 0:
                prompt = messages[-1].get('content', '') if isinstance(messages[-1], dict) else str(messages[-1])
            else:
                prompt = str(messages)
        elif prompt is None:
            prompt = ""
            
        result = self.basic_request(prompt, **kwargs)
        return [result]  # DSPy expects a list
    
    def generate(self, prompt, **kwargs):
        """Alternative method that DSPy might call"""
        return self.__call__(prompt, **kwargs)
    
    def request(self, prompt, **kwargs):
        """Another method DSPy might expect"""
        return self.basic_request(prompt, **kwargs)

# Configure DSPy
try:
    claude = CustomAnthropic(api_key=ANTHROPIC_API_KEY)
    dspy.settings.configure(lm=claude)
except Exception as e:
    raise ValueError(f"Failed to configure DSPy: {str(e)}")

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Lease End DSPy API is running."}

@app.post("/generate-reply")
async def generate_reply(request: Request):
    data = await request.json()
    comment = data.get("comment", "")
    postId = data.get("postId", "")

    try:
        # Simple DSPy pipeline for demonstration
        class SimpleReply(dspy.Signature):
            """Generate a helpful reply to a car lease comment."""
            comment = dspy.InputField(desc="The original comment about car leasing")
            postId = dspy.InputField(desc="The post ID for context")
            reply = dspy.OutputField(desc="A helpful and informative reply")

        reply_module = dspy.Predict(SimpleReply)
        result = reply_module(comment=comment, postId=postId)
        
        # Handle both string and list responses
        reply_text = result.reply
        if isinstance(reply_text, list):
            reply_text = reply_text[0] if reply_text else "No reply generated"
            
        return {
            "postId": postId,
            "reply": reply_text
        }
    except Exception as e:
        print(f"Error in generate_reply: {e}")
        return {
            "postId": postId,
            "reply": "Sorry, I encountered an error while generating a reply.",
            "error": str(e)
        }