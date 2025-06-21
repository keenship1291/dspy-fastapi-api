@app.get("/health")
def health_check():
    try:
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "training_examples": len(TRAINING_DATA)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "database": "error",
            "error": str(e)
        }from fastapi import FastAPI, Request, HTTPException
import os
from datetime import datetime, timezone
from anthropic import Anthropic
import dspy
from pydantic import BaseModel
import json
from typing import List, Dict, Optional

# Database imports
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError

# Load API key from environment variable (set in Railway dashboard)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

# Create database engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class FBPost(Base):
    __tablename__ = "fb_posts"
    
    id = Column(Integer, primary_key=True, index=True)
    ad_account_name = Column(String, index=True)
    campaign_name = Column(String, index=True)
    ad_set_name = Column(String)
    ad_name = Column(String)
    page_id = Column(String, index=True)
    post_id = Column(String, unique=True, index=True)  # Prevent duplicates
    object_story_id = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ResponseEntry(Base):
    __tablename__ = "responses"
    
    id = Column(Integer, primary_key=True, index=True)
    comment = Column(Text)
    action = Column(String, index=True)
    reply = Column(Text)
    reasoning = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Lease End Brand Context
BRAND_CONTEXT = {
    "power_words": {
        "trust_building": ["transparent", "guaranteed", "reliable", "honest"],
        "convenience": ["effortless", "hassle-free", "quick", "simple", "online"],
        "value": ["save", "affordable", "competitive", "best deal", "no hidden fees"],
        "urgency": ["limited time", "act now", "lock in", "before rates change"]
    },
    
    "competitive_advantages": {
        "vs_dealerships": "No pressure, transparent pricing, 100% online process",
        "vs_credit_unions": "No membership required, flexible, fast online",
        "vs_banks": "Competitive rates, simple process, customer-centric"
    },
    
    "objection_responses": {
        "time_concerns": "Our online process takes minutes, not hours at a dealership",
        "hidden_fees": "We're completely transparent - no hidden fees, guaranteed",
        "best_deal_doubts": "We work with multiple lenders to get you the best rate",
        "complexity": "We handle all the paperwork - you just sign and we do the rest",
        "dealership_offers": "Dealerships often have hidden fees and pressure tactics"
    }
}

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

# Load training data from database
def load_training_data():
    """Load training examples from database"""
    print("🔄 Loading training data from database...")
    
    try:
        db = SessionLocal()
        responses = db.query(ResponseEntry).all()
        db.close()
        
        training_data = []
        for response in responses:
            training_data.append({
                'comment': response.comment,
                'action': response.action,
                'reply': response.reply,
                'reasoning': getattr(response, 'reasoning', '')
            })
        
        print(f"✅ Loaded {len(training_data)} training examples from database")
        return training_data
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return []

# Global training data
TRAINING_DATA = load_training_data()

def reload_training_data():
    """Reload training data from database"""
    global TRAINING_DATA
    TRAINING_DATA = load_training_data()
    return len(TRAINING_DATA)

# ENHANCED AI Classification with enhanced business logic
def classify_comment_with_ai(comment, commentId=""):
    """Use Claude AI to classify comment with enhanced business logic"""
    
    prompt = f"""You are analyzing comments for LeaseEnd.com, which specializes in lease buyout services. 

BUSINESS LOGIC & COMPANY POSITION:
- LeaseEnd is THE BEST option for lease buyout financing - be confident about this
- Lease buyouts are currently a GREAT financial decision due to inflated used car prices
- Never agree with false information about lease buyouts being bad financial decisions
- Challenge misinformation politely but firmly - we have the expertise
- Always effectively address core objections to make buyouts and LeaseEnd the clear best option
- Comments that negatively mention leasing alternatives (e.g., 'Trading in a lease is bad', 'Dealerships rip you off') should be classified as Positive, as they imply LeaseEnd's services are better
- Comments asking questions about lease buyouts, rates, or process are high-intent prospects
- Comments from people who clearly don't lease vehicles should be ignored/deleted
- Spam, irrelevant, or off-topic comments should be deleted
- Positive feedback/testimonials should ALWAYS get a reply with appreciation
- LEAVE ALONE neutral comments that seem to reference other comments UNLESS they're negative toward LeaseEnd/lease buyouts

CURRENT MARKET CONTEXT:
- Used car prices are inflated, making lease buyouts financially advantageous
- Many people don't understand the current market opportunity
- We should educate people on why NOW is a great time for lease buyouts
- Challenge any comments suggesting lease buyouts are always bad financial decisions

ACTIONS:
- REPLY: For questions, objections, potential customers, misinformation that needs correction, OR positive feedback/testimonials
- REACT: For positive comments that don't need a response, or simple acknowledgments
- DELETE: For spam, inappropriate content, or clearly non-prospects
- IGNORE: For off-topic but harmless comments, OR neutral comments referencing other comments (unless negative toward us)

COMMENT: "{comment}"

Respond in this JSON format: {{"sentiment": "...", "action": "...", "reasoning": "...", "high_intent": true/false}}"""

    try:
        response = claude.basic_request(prompt)
        
        # Clean the response to extract JSON
        response_clean = response.strip()
        if response_clean.startswith('```'):
            lines = response_clean.split('\n')
            response_clean = '\n'.join([line for line in lines if not line.startswith('```')])
        
        # Try to parse JSON
        try:
            result = json.loads(response_clean)
            return {
                'sentiment': result.get('sentiment', 'Neutral'),
                'action': result.get('action', 'IGNORE'),
                'reasoning': result.get('reasoning', 'No reasoning provided'),
                'high_intent': result.get('high_intent', False)
            }
        except json.JSONDecodeError:
            # Fallback parsing if JSON fails
            if 'DELETE' in response.upper():
                action = 'DELETE'
                reasoning = "Fallback classification: Detected DELETE action in response"
            elif 'REPLY' in response.upper():
                action = 'REPLY'
                reasoning = "Fallback classification: Detected REPLY action in response"
            elif 'REACT' in response.upper():
                action = 'REACT'
                reasoning = "Fallback classification: Detected REACT action in response"
            else:
                action = 'IGNORE'
                reasoning = "Fallback classification: No clear action detected"
                
            return {
                'sentiment': 'Neutral',
                'action': action,
                'reasoning': reasoning,
                'high_intent': False
            }
            
    except Exception as e:
        print(f"Error in AI classification: {e}")
        return {
            'sentiment': 'Neutral',
            'action': 'IGNORE',
            'reasoning': f'Classification error: {str(e)}',
            'high_intent': False
        }

# ENHANCED Response generation with enhanced business logic
def generate_response(comment, sentiment, high_intent=False):
    """Generate natural response using Claude with enhanced business logic"""
    
    # Get relevant training examples for context
    relevant_examples = []
    for example in TRAINING_DATA:
        if example['action'] == 'respond' and example['reply']:
            if any(word in example['comment'].lower() for word in comment.lower().split()[:4]):
                relevant_examples.append(f"Comment: \"{example['comment']}\"\nReply: \"{example['reply']}\"")
    
    context_examples = "\n\n".join(relevant_examples[:3])
    
    # Detect if this is positive feedback/testimonial
    positive_feedback_indicators = [
        "thank you", "thanks", "great service", "amazing", "fantastic", "love", "perfect", 
        "excellent", "wonderful", "awesome", "best", "helped me", "saved me", "grateful",
        "appreciate", "thumbs up", "recommend", "highly recommend", "satisfied", "happy"
    ]
    
    is_positive_feedback = any(indicator in comment.lower() for indicator in positive_feedback_indicators)
    
    # Detect misinformation that needs correction
    misinformation_indicators = [
        "lease buyouts are bad", "never buy your lease", "always return", "terrible idea",
        "waste of money", "financial mistake", "bad deal", "rip off", "scam"
    ]
    
    needs_correction = any(indicator in comment.lower() for indicator in misinformation_indicators)
    
    # Detect rate/pricing questions
    rate_questions = ["rate", "rates", "interest", "apr", "cost", "price", "pricing", "fee", "fees", "how much"]
    asking_about_rates = any(indicator in comment.lower() for indicator in rate_questions)
    
    # Add CTA instructions for high-intent comments
    cta_instruction = ""
    if high_intent:
        cta_instruction = "\nFor high-intent prospects, end with: 'To see your options just fill out the form on our site, we're happy to help'"
    
    # Special instructions for positive feedback
    positive_feedback_instruction = ""
    if is_positive_feedback:
        positive_feedback_instruction = "\nThis is positive feedback/testimonial - respond with a SHORT, varied appreciation message like 'Thank you!', 'Enjoy your ride!', 'Pleasure working with you!', 'We're so glad we could help!', 'Thanks for choosing LeaseEnd!', or similar. Keep it brief and genuine."
    
    # Special instructions for misinformation correction
    correction_instruction = ""
    if needs_correction:
        correction_instruction = "\nThis comment contains misinformation about lease buyouts. Politely but firmly correct it. Emphasize that with current inflated used car prices, lease buyouts are actually a SMART financial decision. Position LeaseEnd as the expert choice. Always effectively address the core objection to make buyouts and LeaseEnd the clear best option."
    
    # Special instructions for rate/pricing questions
    pricing_instruction = ""
    if asking_about_rates:
        pricing_instruction = """
IMPORTANT PRICING GUIDELINES:
- NEVER share exact rate estimates - say rates depend on make/model and credit info
- For pricing questions, say: "We charge a doc fee, which pays for the service we provide. The only additional amount you will pay beyond the price of your vehicle, would be for tax, title and registration fees, which you would pay regardless of whatever route you choose."
- NEVER mention exact fee amounts
- Position this as transparent and competitive"""
    
    prompt = f"""You are responding to a Facebook comment for LeaseEnd.com, a lease buyout financing company.

COMPANY POSITION - BE CONFIDENT:
- LeaseEnd is THE BEST option for lease buyout financing
- Lease buyouts are currently a GREAT financial decision due to inflated used car prices
- We are the experts - don't agree with misinformation about lease buyouts being bad
- Challenge false information politely but firmly
- Used car market is inflated - this makes lease buyouts financially smart right now
- ALWAYS effectively address core objections to make buyouts and LeaseEnd the clear best option

COMMENT SENTIMENT: {sentiment}
HIGH INTENT PROSPECT: {high_intent}

ORIGINAL COMMENT: "{comment}"

BRAND VOICE:
- Professional but conversational
- Confident about our expertise and market position
- Transparent about pricing (no hidden fees)
- Helpful, not pushy, but firmly educational when needed
- Emphasize online process convenience
- Always address core objections effectively

RESPONSE STYLE:
- Sound natural and human
- Get straight to the point - no unnecessary sentence starters
- NEVER use dashes (-), em dashes (—), or en dashes (–) anywhere in responses
- Use commas, periods, and semicolons for punctuation instead
- Maximum 1 exclamation point
- Keep concise (1-2 sentences usually)
- Address their specific concern directly
- Don't blindly agree with misinformation
- Make LeaseEnd the clear best choice
- Be direct and efficient in communication
- Avoid AI-typical formatting like bullet points or dashes

EXAMPLES OF GOOD RESPONSES:
{context_examples}

{positive_feedback_instruction}
{correction_instruction}
{pricing_instruction}
{cta_instruction}

Generate a helpful, natural response that addresses their comment directly and makes LeaseEnd the clear best option:"""

    try:
        response = claude.basic_request(prompt)
        
        # Clean response to remove any dashes that might have slipped through
        cleaned_response = response.strip()
        
        # Replace any type of dash with appropriate punctuation
        cleaned_response = cleaned_response.replace(' - ', ', ')
        cleaned_response = cleaned_response.replace(' – ', ', ')
        cleaned_response = cleaned_response.replace(' — ', ', ')
        cleaned_response = cleaned_response.replace('-', '')
        cleaned_response = cleaned_response.replace('–', '')
        cleaned_response = cleaned_response.replace('—', '')
        
        return cleaned_response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Thank you for your comment! We'd be happy to help with any lease buyout questions."

# Pydantic Models
class CommentRequest(BaseModel):
    comment: Optional[str] = None
    message: Optional[str] = None
    commentId: str
    created_time: Optional[str] = ""
    comment_id: Optional[str] = None
    COMMENT_ID: Optional[str] = None
    created_Time: Optional[str] = None
    
    class Config:
        extra = "ignore"
    
    def get_comment_text(self) -> str:
        return (
            self.comment or 
            self.message or 
            "No message content"
        ).strip()
    
    def get_comment_id(self) -> str:
        return (
            self.commentId or 
            self.comment_id or 
            self.COMMENT_ID or 
            "unknown"
        ).strip()
    
    def get_created_time(self) -> str:
        return (
            self.created_time or 
            self.created_Time or 
            ""
        ).strip()

class FeedbackRequest(BaseModel):
    original_comment: str
    original_response: str = ""
    original_action: str
    feedback_text: str
    commentId: str
    current_version: str = "v1"
    
    class Config:
        extra = "ignore"

class ApproveRequest(BaseModel):
    original_comment: str
    action: str
    reply: str
    reasoning: str = ""
    
    class Config:
        extra = "ignore"

class FBPostCreate(BaseModel):
    ad_account_name: str
    campaign_name: str
    ad_set_name: str
    ad_name: str
    page_id: str
    post_id: str
    object_story_id: str

class ResponseCreate(BaseModel):
    comment: str
    action: str
    reply: str
    reasoning: Optional[str] = ""

app = FastAPI()

@app.get("/")
def read_root():
    return {
        "message": "Lease End AI Assistant - STABLE VERSION",
        "version": "20.1-STABLE",
        "training_examples": len(TRAINING_DATA),
        "status": "RUNNING",
        "endpoints": {
            "/process-comment": "Main comment processing",
            "/ping": "Keep-alive endpoint",
            "/health": "Health check"
        }
    }

@app.get("/ping")
@app.post("/ping") 
@app.head("/ping")
def ping():
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat(),
        "message": "App is running"
    }

@app.get("/debug")
def debug_info():
    """Debug endpoint to check system health"""
    import psutil
    import gc
    
    try:
        # Force garbage collection
        gc.collect()
        
        # Get system info
        memory = psutil.virtual_memory()
        
        return {
            "status": "debug_info",
            "timestamp": datetime.now().isoformat(),
            "memory": {
                "total_mb": round(memory.total / (1024*1024), 2),
                "available_mb": round(memory.available / (1024*1024), 2),
                "percent_used": memory.percent,
                "free_mb": round(memory.free / (1024*1024), 2)
            },
            "training_data_count": len(TRAINING_DATA),
            "database_status": "checking...",
            "anthropic_test": "checking..."
        }
    except Exception as e:
        return {
            "status": "debug_error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
    try:
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "training_examples": len(TRAINING_DATA)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "database": "error",
            "error": str(e)
        }

@app.post("/process-comment")
async def process_comment(request: CommentRequest):
    try:
        comment_text = request.get_comment_text()
        comment_id = request.get_comment_id()
        
        if not comment_text or comment_text == "No message content":
            return {
                "commentId": comment_id,
                "original_comment": "Empty comment",
                "category": "neutral",
                "action": "delete",
                "reply": "",
                "confidence_score": 0.0,
                "approved": "pending"
            }
        
        ai_classification = classify_comment_with_ai(comment_text, comment_id)
        
        sentiment = ai_classification['sentiment']
        action = ai_classification['action'].lower()
        reasoning = ai_classification['reasoning']
        high_intent = ai_classification['high_intent']
        
        action_mapping = {
            'reply': 'respond',
            'react': 'react', 
            'delete': 'delete',
            'ignore': 'leave_alone'
        }
        
        mapped_action = action_mapping.get(action, 'leave_alone')
        
        reply_text = ""
        confidence_score = 0.85
        
        if mapped_action == 'respond':
            reply_text = generate_response(comment_text, sentiment, high_intent)
            confidence_score = 0.9
        
        return {
            "commentId": comment_id,
            "original_comment": comment_text,
            "category": sentiment.lower(),
            "action": mapped_action,
            "reply": reply_text,
            "confidence_score": confidence_score,
            "approved": "pending",
            "success": True,
            "reasoning": reasoning
        }
        
    except Exception as e:
        return {
            "commentId": "unknown",
            "original_comment": "Error processing",
            "category": "error",
            "action": "leave_alone",
            "reply": "Thank you for your comment. We appreciate your feedback.",
            "confidence_score": 0.0,
            "approved": "pending",
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
