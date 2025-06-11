from fastapi import FastAPI, Request
import os
from anthropic import Anthropic
import dspy
from pydantic import BaseModel

# Load API key from environment variable (set in Railway dashboard)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")

# Custom Anthropic LM for DSPy with better response handling
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
            if isinstance(messages, list) and len(messages) > 0:
                prompt = messages[-1].get('content', '') if isinstance(messages[-1], dict) else str(messages[-1])
            else:
                prompt = str(messages)
        elif prompt is None:
            prompt = ""
            
        result = self.basic_request(prompt, **kwargs)
        return [result]  # DSPy expects a list
    
    def generate(self, prompt, **kwargs):
        return self.__call__(prompt, **kwargs)
    
    def request(self, prompt, **kwargs):
        return self.basic_request(prompt, **kwargs)

# Configure DSPy
try:
    claude = CustomAnthropic(api_key=ANTHROPIC_API_KEY)
    dspy.settings.configure(lm=claude)
except Exception as e:
    raise ValueError(f"Failed to configure DSPy: {str(e)}")

# DSPy Signatures with fallback handling
class CommentClassification(dspy.Signature):
    """Classify a social media comment for automotive lease company"""
    comment = dspy.InputField(desc="The customer comment to classify")
    comment_age = dspy.InputField(desc="How old the comment is")
    
    category = dspy.OutputField(desc="primary_inquiry, complaint, praise, spam, general")
    urgency = dspy.OutputField(desc="low, medium, high")
    requires_response = dspy.OutputField(desc="true or false")

class LeaseResponseGeneration(dspy.Signature):
    """Generate professional response to automotive lease inquiry"""
    comment = dspy.InputField(desc="Original customer comment")
    category = dspy.InputField(desc="Classified category")
    urgency = dspy.InputField(desc="Urgency level")
    
    reply = dspy.OutputField(desc="Professional, helpful response")

# Request/Response models
class CommentRequest(BaseModel):
    comment: str
    postId: str
    created_time: str = ""
    memory_context: str = ""

class ProcessedComment(BaseModel):
    postId: str
    original_comment: str
    category: str
    urgency: str
    requires_response: bool
    reply: str
    should_post: bool
    needs_review: bool

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Lease End DSPy API with Classification + Response"}

@app.post("/process-comment", response_model=ProcessedComment)
async def process_comment(request: CommentRequest):
    """Full comment processing: classification + response generation"""
    try:
        # Calculate comment age
        comment_age = "recent"
        if request.created_time:
            from datetime import datetime, timezone
            try:
                created = datetime.fromisoformat(request.created_time.replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                age_hours = (now - created).total_seconds() / 3600
                
                if age_hours < 2:
                    comment_age = "recent"
                elif age_hours < 24:
                    comment_age = "hours_old"
                else:
                    comment_age = "days_old"
            except:
                comment_age = "unknown"
        
        # Step 1: Classify the comment with fallback
        try:
            classifier = dspy.Predict(CommentClassification)
            classification = classifier(
                comment=request.comment,
                comment_age=comment_age
            )
            
            category = getattr(classification, 'category', 'general')
            urgency = getattr(classification, 'urgency', 'medium')
            requires_response_str = getattr(classification, 'requires_response', 'true')
            requires_response = requires_response_str.lower() == "true"
            
        except Exception as e:
            print(f"Classification error: {e}")
            # Fallback classification logic
            comment_lower = request.comment.lower()
            if any(word in comment_lower for word in ['complaint', 'terrible', 'awful', 'hate', 'sucks']):
                category = "complaint"
                urgency = "high"
                requires_response = True
            elif any(word in comment_lower for word in ['thank', 'great', 'love', 'awesome']):
                category = "praise"
                urgency = "low"
                requires_response = True
            elif '?' in request.comment:
                category = "primary_inquiry"
                urgency = "medium"
                requires_response = True
            else:
                category = "general"
                urgency = "low"
                requires_response = False
        
        # Step 2: Generate response if needed
        reply_text = ""
        if requires_response:
            try:
                responder = dspy.Predict(LeaseResponseGeneration)
                response = responder(
                    comment=request.comment,
                    category=category,
                    urgency=urgency
                )
                reply_text = getattr(response, 'reply', '')
                
            except Exception as e:
                print(f"Response generation error: {e}")
                # Fallback response generation
                claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)
                
                prompt = f"""You are a professional customer service representative for a car lease company called "Lease End."

Customer comment: "{request.comment}"
Comment category: {category}
Urgency: {urgency}

Generate a helpful, professional response. Be friendly, informative, and address their concern directly. Keep it concise but helpful.

Response:"""

                claude_response = claude_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                reply_text = claude_response.content[0].text.strip()
        
        # Decision logic for automation
        should_post = (
            requires_response and 
            urgency != "high" and
            comment_age != "days_old" and
            category != "complaint"
        )
        
        needs_review = (
            requires_response and 
            (urgency == "high" or comment_age == "days_old" or category == "complaint")
        )
        
        return ProcessedComment(
            postId=request.postId,
            original_comment=request.comment,
            category=category,
            urgency=urgency,
            requires_response=requires_response,
            reply=reply_text,
            should_post=should_post,
            needs_review=needs_review
        )
        
    except Exception as e:
        print(f"Error processing comment: {e}")
        return ProcessedComment(
            postId=request.postId,
            original_comment=request.comment,
            category="error",
            urgency="low",
            requires_response=False,
            reply="Thank you for your comment. We appreciate your feedback.",
            should_post=False,
            needs_review=True
        )

# Keep the simple endpoint for backwards compatibility
@app.post("/generate-reply")
async def generate_reply(request: Request):
    """Simple reply generation"""
    data = await request.json()
    comment = data.get("comment", "")
    postId = data.get("postId", "")

    try:
        claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        
        prompt = f"""You are a professional customer service representative for a car lease company.

Customer comment: "{comment}"

Generate a helpful, professional response:"""

        response = claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            "postId": postId,
            "reply": response.content[0].text.strip()
        }
        
    except Exception as e:
        return {
            "postId": postId,
            "reply": "Thank you for your comment. We appreciate your feedback.",
            "error": str(e)
        }

# Railway-specific server startup
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
