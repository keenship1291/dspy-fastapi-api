from fastapi import FastAPI, Request, HTTPException
import os
from datetime import datetime, timezone, timedelta
from anthropic import Anthropic
import dspy
from pydantic import BaseModel
import json
from typing import List, Dict, Optional
import uuid
import re

# Database imports
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, text
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
    post_id = Column(String, unique=True, index=True)
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
    completed_at = Column(DateTime)

class BatchJob(Base):
    __tablename__ = "batch_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True)
    status = Column(String, default="processing")
    total_comments = Column(Integer, default=0)
    processed_comments = Column(Integer, default=0)
    results = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

# Create tables
Base.metadata.create_all(bind=engine)

# Centralized LeaseEnd Response Guidelines
LEASEEND_GUIDELINES = """
COMPANY POSITION - BE CONFIDENT:
- LeaseEnd is THE BEST option for lease buyout financing
- Lease buyouts are SMART financial decisions - find the most compelling argument for this specific comment
- We are the experts - don't agree with misinformation about lease buyouts being bad
- Challenge false information politely but firmly
- Monthly car payments are going to continue rising - use this reality when relevant
- ALWAYS effectively address core objections to make buyouts and LeaseEnd the clear best option

BRAND VOICE:
- Professional but conversational
- Confident about our expertise and market position
- Transparent about pricing (no hidden fees)
- Helpful, not pushy, but firmly educational when needed
- Emphasize online process convenience
- Always address core objections effectively

CTA GUIDELINES:
- For urgent customers: Include phone (844) 679-1188 and website options
- For general interest: Soft website CTA like "check out our site to see your options"
- Keep CTAs natural and helpful, not pushy

CRITICAL NUMERICAL VALUE RESTRICTIONS:
- NEVER include any dollar amounts ($500, $1000, etc.)
- NEVER include any percentages (5%, 10%, 3.5%, etc.)  
- NEVER include specific interest rates (4.5% APR, 6% interest, etc.)
- NEVER include specific fee amounts ($299 doc fee, $150 processing, etc.)
- Use qualitative terms instead: "competitive rates", "affordable", "low fees", "great deal"
- If asked about specific numbers, redirect to: "Rates depend on your specific vehicle and credit profile"

PRICING GUIDELINES:
- NEVER share exact rate estimates - say rates depend on make/model and credit info
- For pricing questions, explain our transparent doc fee approach without specific amounts
- Position this as transparent and competitive
"""

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
        return [result]
    
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

def filter_numerical_values(text):
    """Remove any numerical values (dollars, percentages, rates) from response text"""
    
    # Remove dollar amounts ($X, $X.XX, $X,XXX, etc.)
    text = re.sub(r'\$[\d,]+(?:\.\d{2})?', '', text)
    
    # Remove percentages (X%, X.X%, XX.XX%, etc.)
    text = re.sub(r'\b\d+(?:\.\d+)?%', '', text)
    
    # Remove APR/interest rate patterns (X.X% APR, X% interest, etc.)
    text = re.sub(r'\b\d+(?:\.\d+)?\s*%?\s*(?:APR|apr|interest|rate)', '', text)
    
    # Remove standalone numbers that might be rates (like "4.5" or "6.2")
    text = re.sub(r'\b\d+\.\d+\b(?=\s*(?:rate|APR|interest|%|percent))', '', text)
    
    # Clean up any double spaces or awkward spacing created by removals
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,.!?])', r'\1', text)  # Remove space before punctuation
    
    return text.strip()

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def classify_comment_with_ai(comment, commentId=""):
    """Use Claude AI to classify comment with enhanced business logic"""
    
    prompt = f"""You are analyzing comments for LeaseEnd.com, which specializes in lease buyout services. 

BUSINESS LOGIC & COMPANY POSITION:
- LeaseEnd is THE BEST option for lease buyout financing - be confident about this
- Lease buyouts are SMART financial decisions - find the most compelling argument for each situation
- Never agree with false information about lease buyouts being bad financial decisions
- Challenge misinformation politely but firmly - we have the expertise
- Always effectively address core objections to make buyouts and LeaseEnd the clear best option
- Comments that negatively mention leasing alternatives (e.g., 'Trading in a lease is bad', 'Dealerships rip you off') should be classified as Positive, as they imply LeaseEnd's services are better
- Comments asking questions about lease buyouts, rates, or process are high-intent prospects
- Comments from people who clearly don't lease vehicles should be ignored/deleted
- Spam, irrelevant, or off-topic comments should be deleted
- Positive feedback/testimonials should ALWAYS get a reply with appreciation
- LEAVE ALONE neutral comments that seem to reference other comments UNLESS they're negative toward LeaseEnd/lease buyouts

CORE PRINCIPLE TO UNDERSTAND:
- Monthly car payments are going to continue rising - this economic reality makes lease buyouts smart timing

TAGGING DETECTION LOGIC:
- If comment starts with a name (like "John Smith check this out" or "@Sarah this might help"), the person is tagging someone
- For tagged comments: LEAVE ALONE unless it's very negative toward LeaseEnd or lease buyouts
- For very negative tagged comments toward LeaseEnd/lease buyouts: DELETE
- Tagged comments are usually people sharing with friends, not direct engagement with us

ACTIONS:
- REPLY: For questions, objections, potential customers, misinformation that needs correction, OR positive feedback/testimonials
- REACT: For positive comments that don't need a response, or simple acknowledgments
- DELETE: For spam, inappropriate content, clearly non-prospects, very negative tagged comments about LeaseEnd/lease buyouts, OR brief negative comments like "scam", "ripoff", "terrible"
- LEAVE_ALONE: For off-topic but harmless comments, neutral comments referencing other comments, OR tagged comments (unless very negative toward us)

COMMENT: "{comment}"

Respond in this JSON format: {{"sentiment": "...", "action": "...", "reasoning": "...", "high_intent": true/false}}"""

    try:
        response = claude.basic_request(prompt)
        
        print(f"🔍 DEBUG - Raw AI response for '{comment[:20]}...': {response[:200]}...")
        
        # Clean the response to extract JSON
        response_clean = response.strip()
        if response_clean.startswith('```'):
            lines = response_clean.split('\n')
            response_clean = '\n'.join([line for line in lines if not line.startswith('```')])
        
        # Try to parse JSON
        try:
            result = json.loads(response_clean)
            print(f"✅ Successfully parsed JSON: {result}")
            return {
                'sentiment': result.get('sentiment', 'Neutral'),
                'action': result.get('action', 'IGNORE'),
                'reasoning': result.get('reasoning', 'No reasoning provided'),
                'high_intent': result.get('high_intent', False)
            }
        except json.JSONDecodeError as json_error:
            print(f"❌ JSON parsing failed: {json_error}")
            print(f"❌ Attempted to parse: {response_clean[:200]}...")
            
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
                action = 'LEAVE_ALONE'
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
            'action': 'LEAVE_ALONE',
            'reasoning': f'Classification error: {str(e)}',
            'high_intent': False
        }

# Enhanced Response generation with numerical value filtering and no dash removal
def generate_response(comment, sentiment, high_intent=False):
    """Generate natural response using Claude with enhanced business logic and no numerical values"""
    
    # Get relevant training examples for context
    relevant_examples = []
    for example in TRAINING_DATA:
        if example['action'] == 'respond' and example['reply']:
            if any(word in example['comment'].lower() for word in comment.lower().split()[:4]):
                relevant_examples.append(f"Comment: \"{example['comment']}\"\nReply: \"{example['reply']}\"")
    
    context_examples = "\n\n".join(relevant_examples[:3])
    
    # Detect if this is a potential customer vs just making statements
    customer_indicators = [
        "how much", "what are", "can i", "should i", "interested", "looking", 
        "want to", "need", "help me", "my lease", "my car", "rates", "process",
        "qualify", "apply", "cost", "price", "how do", "when can", "where do"
    ]
    
    statement_indicators = [
        "i think", "i believe", "in my opinion", "that's wrong", "that's not true",
        "never", "always", "people should", "anyone who", "dealerships are",
        "banks are", "leasing is", "buying is", "worst", "terrible", "bad idea"
    ]
    
    # Detect urgency/personalized help needs
    urgent_indicators = [
        "urgent", "asap", "need help now", "time sensitive", "deadline", "expires", 
        "confused", "don't understand", "complicated", "help me", "call me", "speak to someone"
    ]
    
    is_potential_customer = any(indicator in comment.lower() for indicator in customer_indicators)
    is_making_statement = any(indicator in comment.lower() for indicator in statement_indicators)
    needs_personal_help = any(indicator in comment.lower() for indicator in urgent_indicators)
    
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
    
    # CTA logic: Only for potential customers, not for people making statements
    cta_instruction = ""
    if is_potential_customer and high_intent and not is_making_statement:
        if needs_personal_help:
            cta_instruction = "\nFor this customer who seems to need immediate or personalized help, offer both options: 'Feel free to give us a call at (844) 679-1188 for immediate help, or check out our site to see your options.' Keep it natural and helpful."
        else:
            cta_instruction = "\nFor this potential customer, you may end with a soft CTA like: 'Feel free to check out our site to see your options' or 'We'd be happy to help if you have questions.' Keep it natural and not pushy."
    elif is_making_statement:
        cta_instruction = "\nThis person is making a statement or argument. DO NOT include any call to action. Simply address their point, correct any misinformation, and provide helpful information. No sales pitch needed."
    
    # Special instructions for positive feedback
    positive_feedback_instruction = ""
    if is_positive_feedback:
        positive_feedback_instruction = "\nThis is positive feedback/testimonial - respond with a SHORT, varied appreciation message like 'Thank you!', 'Enjoy your ride!', 'Pleasure working with you!', 'We're so glad we could help!', 'Thanks for choosing LeaseEnd!', or similar. Keep it brief and genuine."
    
    # Special instructions for misinformation correction
    correction_instruction = ""
    if needs_correction:
        correction_instruction = "\nThis comment contains misinformation about lease buyouts. Politely but firmly correct it. Find the most compelling, relevant argument for why lease buyouts make sense. Remember that monthly car payments are going to continue rising. Position LeaseEnd as the expert choice. Always effectively address the core objection to make buyouts and LeaseEnd the clear best option."
    
    # Special instructions for rate/pricing questions
    pricing_instruction = ""
    if asking_about_rates:
        pricing_instruction = """
IMPORTANT PRICING GUIDELINES:
- NEVER share exact rate estimates - say rates depend on make/model and credit info
- For pricing questions, say: "We charge a doc fee, which pays for the service we provide. The only additional amount you will pay beyond the price of your vehicle, would be for tax, title and registration fees, which you would pay regardless of whatever route you choose."
- NEVER mention exact fee amounts or percentages
- Position this as transparent and competitive"""
    
    prompt = f"""You are responding to a Facebook comment for LeaseEnd.com, a lease buyout financing company.

COMPANY POSITION - BE CONFIDENT:
- LeaseEnd is THE BEST option for lease buyout financing
- Lease buyouts are SMART financial decisions - find the most compelling argument for this specific comment
- We are the experts - don't agree with misinformation about lease buyouts being bad
- Challenge false information politely but firmly
- Monthly car payments are going to continue rising - use this reality to support lease buyouts when relevant
- ALWAYS effectively address core objections to make buyouts and LeaseEnd the clear best option

COMMENT SENTIMENT: {sentiment}
HIGH INTENT PROSPECT: {high_intent}
POTENTIAL CUSTOMER: {is_potential_customer}
MAKING STATEMENT: {is_making_statement}

ORIGINAL COMMENT: "{comment}"

BRAND VOICE:
- Professional but conversational
- Confident about our expertise and market position
- Transparent about pricing (no hidden fees)
- Helpful, not pushy, but firmly educational when needed
- Emphasize online process convenience
- Always address core objections effectively

CRITICAL NUMERICAL VALUE RESTRICTIONS:
- NEVER include any dollar amounts ($500, $1000, etc.)
- NEVER include any percentages (5%, 10%, 3.5%, etc.)  
- NEVER include specific interest rates (4.5% APR, 6% interest, etc.)
- NEVER include specific fee amounts ($299 doc fee, $150 processing, etc.)
- Use qualitative terms instead: "competitive rates", "affordable", "low fees", "great deal"
- If asked about specific numbers, redirect to: "Rates depend on your specific vehicle and credit profile"

RESPONSE STYLE:
- Sound natural and human
- Get straight to the point - no unnecessary sentence starters
- NEVER use ALL CAPS text - it's unprofessional and looks like shouting
- Use commas, periods, and semicolons for punctuation
- Maximum 1 exclamation point
- Keep concise (1-2 sentences usually)
- Address their specific concern directly
- Don't blindly agree with misinformation
- Make LeaseEnd the clear best choice
- Be direct and efficient in communication
- Avoid AI-typical formatting like bullet points

EXAMPLES OF GOOD RESPONSES:
{context_examples}

{positive_feedback_instruction}
{correction_instruction}
{pricing_instruction}
{cta_instruction}

Generate a helpful, natural response that addresses their comment directly, makes LeaseEnd the clear best option, and contains NO numerical values whatsoever:"""

    try:
        response = claude.basic_request(prompt)
        
        # Apply numerical value filter but keep response formatting intact
        cleaned_response = filter_numerical_values(response.strip())
        
        return cleaned_response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Thank you for your comment! We'd be happy to help with any lease buyout questions."

# Pydantic Models
class CommentRequest(BaseModel):
    comment: Optional[str] = None
    message: Optional[str] = None
    commentId: Optional[str] = None
    postId: Optional[str] = None
    created_time: Optional[str] = ""
    memory_context: Optional[str] = ""
    
    def get_comment_text(self) -> str:
        return (self.comment or self.message or "No message content").strip()
    
    def get_comment_id(self) -> str:
        return (self.commentId or self.postId or "unknown").strip()

class Comment(BaseModel):
    comment: Optional[str] = None
    message: Optional[str] = None
    commentId: str
    created_time: Optional[str] = ""
    
    def get_comment_text(self) -> str:
        return (self.comment or self.message or "No message content").strip()

class BatchCommentRequest(BaseModel):
    comments: List[Comment]
    batch_id: Optional[str] = None

class ProcessedComment(BaseModel):
    commentId: str  # Changed from postId to commentId
    original_comment: str
    category: str
    action: str
    reply: str
    confidence_score: float
    approved: str
    reasoning: str

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
    created_time: Optional[str] = ""
    
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
        "message": "Lease End AI Assistant - FULL VERSION",
        "version": "27.1-CONCISE",
        "training_examples": len(TRAINING_DATA),
        "status": "RUNNING",
        "features": ["Batch Processing", "Smart CTA", "Professional Tone", "Phone Support", "No Numerical Values", "Natural Arguments", "Centralized Guidelines"],
        "endpoints": {
            "/process-comment": "Single comment processing (legacy)",
            "/process-batch": "Batch comment processing (new)",
            "/process-feedback": "Human feedback processing",
            "/approve-response": "Approve responses for training",
            "/fb-posts": "Get all FB posts",
            "/fb-posts/add": "Add new FB post",
            "/responses": "Get all response data", 
            "/responses/add": "Add new response data",
            "/reload-training-data": "Reload training from database",
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

@app.get("/health")
def health_check():
    try:
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

# Database Endpoints - MINIMAL FIX VERSION
@app.get("/fb-posts")
async def get_fb_posts():
    """Get all FB posts from database"""
    db = SessionLocal()
    try:
        posts = db.query(FBPost).all()
        
        return {
            "success": True,
            "count": len(posts),
            "data": [
                {
                    "Ad account name": post.ad_account_name,
                    "Campaign name": post.campaign_name,
                    "Ad set name": post.ad_set_name,
                    "Ad name": post.ad_name,
                    "Page ID": post.page_id,
                    "Post Id": post.post_id,
                    "Object Story ID": post.object_story_id
                }
                for post in posts
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

@app.post("/fb-posts/add")
async def add_fb_post(post: FBPostCreate):
    """Add new FB post to database"""
    try:
        db = SessionLocal()
        
        db_post = FBPost(
            ad_account_name=post.ad_account_name,
            campaign_name=post.campaign_name,
            ad_set_name=post.ad_set_name,
            ad_name=post.ad_name,
            page_id=post.page_id,
            post_id=post.post_id,
            object_story_id=post.object_story_id
        )
        
        db.add(db_post)
        db.commit()
        db.close()
        
        return {
            "success": True,
            "message": "FB post added successfully",
            "post_id": post.post_id,
            "status": "new"
        }
        
    except IntegrityError:
        db.rollback()
        db.close()
        return {
            "success": True,
            "message": f"Post {post.post_id} already exists (duplicate skipped)",
            "post_id": post.post_id,
            "status": "duplicate_skipped"
        }
    except Exception as e:
        db.rollback()
        db.close()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.delete("/fb-posts/clear-recent")
async def clear_recent_posts():
    """Delete FB posts created in the past 24 hours"""
    try:
        db = SessionLocal()
        
        # Calculate 24 hours ago
        twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
        
        # Find posts created in the past 24 hours
        recent_posts = db.query(FBPost).filter(FBPost.created_at >= twenty_four_hours_ago).all()
        deleted_count = len(recent_posts)
        
        # Delete them
        db.query(FBPost).filter(FBPost.created_at >= twenty_four_hours_ago).delete()
        db.commit()
        db.close()
        
        return {
            "success": True,
            "message": f"Deleted {deleted_count} posts created in the past 24 hours",
            "deleted_count": deleted_count,
            "cutoff_time": twenty_four_hours_ago.isoformat()
        }
        
    except Exception as e:
        db.rollback()
        db.close()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/responses")
async def get_responses():
    """Get all response training data"""
    db = SessionLocal()
    try:
        responses = db.query(ResponseEntry).all()
        
        return {
            "success": True,
            "count": len(responses),
            "data": [
                {
                    "comment": resp.comment,
                    "action": resp.action,
                    "reply": resp.reply
                }
                for resp in responses
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

@app.post("/responses/add")
async def add_response(response: ResponseCreate):
    """Add new response training data"""
    db = SessionLocal()
    try:
        db_response = ResponseEntry(
            comment=response.comment,
            action=response.action,
            reply=response.reply,
            reasoning=response.reasoning
        )
        
        db.add(db_response)
        db.commit()
        
        new_count = reload_training_data()
        
        return {
            "success": True,
            "message": "Response added successfully",
            "new_training_count": new_count
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

@app.post("/reload-training-data")
async def reload_training_data_endpoint():
    """Manually reload training data from database"""
    try:
        new_count = reload_training_data()
        return {
            "success": True,
            "message": "Training data reloaded from database",
            "total_examples": new_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading training data: {str(e)}")

# NEW: Batch Processing Endpoint
@app.post("/process-batch")
async def process_batch(request: BatchCommentRequest):
    """Process multiple comments in a batch and store results"""
    try:
        job_id = str(uuid.uuid4())
        
        db = SessionLocal()
        batch_job = BatchJob(
            job_id=job_id,
            status="processing",
            total_comments=len(request.comments),
            processed_comments=0
        )
        db.add(batch_job)
        db.commit()
        
        results = []
        processed_count = 0
        
        for comment_data in request.comments:
            try:
                comment_text = comment_data.get_comment_text()
                comment_id = comment_data.commentId
                
                if not comment_text or comment_text == "No message content":
                    result = {
                        "commentId": comment_id,
                        "original_comment": "Empty comment",
                        "category": "neutral",
                        "action": "delete",
                        "reply": "",
                        "confidence_score": 0.0,
                        "approved": "pending"
                    }
                else:
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
                    
                    result = {
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
                
                results.append(result)
                processed_count += 1
                
                batch_job.processed_comments = processed_count
                db.commit()
                
            except Exception as comment_error:
                result = {
                    "commentId": comment_data.commentId,
                    "original_comment": "Error processing",
                    "category": "error",
                    "action": "leave_alone",
                    "reply": "Thank you for your comment. We appreciate your feedback.",
                    "confidence_score": 0.0,
                    "approved": "pending",
                    "success": False,
                    "error": str(comment_error)
                }
                results.append(result)
                processed_count += 1
        
        batch_job.status = "completed"
        batch_job.completed_at = datetime.utcnow()
        batch_job.results = json.dumps(results)
        db.commit()
        db.close()
        
        return {
            "job_id": job_id,
            "status": "completed",
            "total_comments": len(request.comments),
            "processed_comments": processed_count,
            "results": results
        }
        
    except Exception as e:
        try:
            batch_job.status = "failed"
            batch_job.results = json.dumps({"error": str(e)})
            db.commit()
            db.close()
        except:
            pass
            
        return {
            "job_id": job_id if 'job_id' in locals() else "unknown",
            "status": "failed",
            "total_comments": len(request.comments) if request.comments else 0,
            "processed_comments": 0,
            "error": str(e)
        }

# ORIGINAL: Single Comment Processing (backward compatibility)
@app.post("/process-comment", response_model=ProcessedComment)
async def process_comment(request: CommentRequest):
    """Core comment processing using AI classification"""
    try:
        comment_text = request.get_comment_text()
        comment_id = request.get_comment_id()
        
        # Use AI to classify the comment
        ai_classification = classify_comment_with_ai(comment_text, comment_id)
        
        sentiment = ai_classification['sentiment']
        action = ai_classification['action'].lower()
        reasoning = ai_classification['reasoning']
        high_intent = ai_classification['high_intent']
        
        # Map actions to our system
        action_mapping = {
            'reply': 'respond',
            'react': 'react', 
            'delete': 'delete',
            'leave_alone': 'leave_alone'
        }
        
        mapped_action = action_mapping.get(action, 'leave_alone')
        
        # Generate response only if action is 'respond'
        reply_text = ""
        confidence_score = 0.85
        
        if mapped_action == 'respond':
            reply_text = generate_response(comment_text, sentiment, high_intent)
            confidence_score = 0.9
        
        return ProcessedComment(
            commentId=comment_id,  # Changed from postId
            original_comment=comment_text,
            category=sentiment.lower(),
            action=mapped_action,
            reply=reply_text,
            confidence_score=confidence_score,
            approved="pending",
            reasoning=reasoning
        )
        
    except Exception as e:
        print(f"Error processing comment: {e}")
        return ProcessedComment(
            commentId=request.get_comment_id(),
            original_comment=request.get_comment_text(),
            category="error",
            action="leave_alone",
            reply="Thank you for your comment. We appreciate your feedback.",
            confidence_score=0.0,
            approved="pending",
            reasoning="Error occurred during processing"
        )

@app.post("/process-feedback")
async def process_feedback(request: FeedbackRequest):
    """Use human feedback to improve responses"""
    try:
        # Clean and validate input data
        original_comment = request.original_comment.strip()
        original_response = request.original_response.strip() if request.original_response else "No response was generated"
        original_action = request.original_action.strip().lower()
        feedback_text = request.feedback_text.strip()
        
        print(f"🔄 Processing feedback for commentId: {request.commentId}")
        print(f"📝 Feedback: {feedback_text[:100]}...")
        
        # Enhanced prompt that includes human feedback
        feedback_prompt = f"""You are improving a response based on human feedback for LeaseEnd.com.

ORIGINAL COMMENT: "{original_comment}"
YOUR ORIGINAL RESPONSE: "{original_response}"
YOUR ORIGINAL ACTION: "{original_action}"

HUMAN FEEDBACK: "{feedback_text}"

{LEASEEND_GUIDELINES}

Generate an IMPROVED response that incorporates the human feedback while maintaining our company position and brand voice.

Respond in this JSON format: {{"sentiment": "...", "action": "REPLY/REACT/DELETE/IGNORE", "reply": "...", "reasoning": "Explain why this action and response make sense for this comment based on analysis and human feedback", "confidence": 0.85}}"""

        improved_response = claude.basic_request(feedback_prompt)
        
        # Parse the improved response
        try:
            response_clean = improved_response.strip()
            if response_clean.startswith('```'):
                lines = response_clean.split('\n')
                response_clean = '\n'.join([line for line in lines if not line.startswith('```')])
            
            if not response_clean.startswith('{'):
                json_start = response_clean.find('{')
                json_end = response_clean.rfind('}') + 1
                if json_start != -1 and json_end != 0:
                    response_clean = response_clean[json_start:json_end]
            
            result = json.loads(response_clean)
            
            # Map actions to our system
            action_mapping = {
                'reply': 'respond',
                'react': 'react', 
                'delete': 'delete',
                'leave_alone': 'leave_alone'
            }
            
            # Clean response and apply numerical value filter
            improved_action = action_mapping.get(result.get('action', 'ignore').lower(), 'leave_alone')
            improved_reply = filter_numerical_values(result.get('reply', ''))  # Apply filter to reply
            
            # Calculate new version number
            try:
                current_version_num = int(request.current_version.replace('v', '')) if request.current_version.startswith('v') else 1
            except:
                current_version_num = 1
            new_version = f"v{current_version_num + 1}"
            
            return {
                "commentId": request.commentId,
                "original_comment": original_comment,
                "category": result.get('sentiment', 'neutral').lower(),
                "action": improved_action,
                "reply": improved_reply,
                "confidence_score": float(result.get('confidence', 0.85)),
                "approved": "pending",
                "feedback_text": feedback_text,  # Return the feedback_text
                "version": new_version,
                "reasoning": result.get('reasoning', 'Applied human feedback to improve response'),
                "feedback_processed": True,
                "success": True
            }
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            new_version_num = int(request.current_version.replace('v', '')) + 1 if request.current_version.startswith('v') else 2
            return {
                "commentId": request.commentId,
                "original_comment": original_comment,
                "category": "neutral",
                "action": "leave_alone",
                "reply": "Thank you for your comment. We appreciate your feedback.",
                "confidence_score": 0.5,
                "approved": "pending",
                "feedback_text": feedback_text,
                "version": f"v{new_version_num}",
                "reasoning": "JSON parsing failed during feedback processing - applied safe fallback response",
                "error": f"JSON parsing failed: {str(e)}",
                "success": False
            }
            
    except Exception as e:
        print(f"❌ Feedback processing error: {e}")
        new_version_num = int(request.current_version.replace('v', '')) + 1 if request.current_version.startswith('v') else 2
        return {
            "commentId": request.commentId,
            "original_comment": request.original_comment,
            "category": "error",
            "action": "leave_alone",
            "reply": "Thank you for your comment. We appreciate your feedback.",
            "confidence_score": 0.0,
            "approved": "pending",
            "feedback_text": feedback_text,
            "version": f"v{new_version_num}",
            "reasoning": "Error occurred during feedback processing - applied safe fallback response",
            "error": str(e),
            "success": False
        }

@app.post("/approve-response")
async def approve_response(request: ApproveRequest):
    """Approve and store a response for training"""
    db = SessionLocal()
    try:
        # Parse created_time if provided, otherwise use current time
        comment_created_at = datetime.utcnow()  # Default fallback
        if request.created_time:
            try:
                # Try to parse the created_time string
                comment_created_at = datetime.fromisoformat(request.created_time.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                # If parsing fails, use current time
                comment_created_at = datetime.utcnow()
        
        # Store the approved response as training data with the ACTUAL action passed
        response_entry = ResponseEntry(
            comment=request.original_comment,
            action=request.action,  # Use the action from the request, not hardcoded "respond"
            reply=request.reply,
            reasoning=request.reasoning,
            created_at=comment_created_at  # Use the comment's original timestamp
        )
        db.add(response_entry)
        db.commit()
        
        # Reload training data
        global TRAINING_DATA
        TRAINING_DATA = load_training_data()
        
        return {
            "status": "approved",
            "message": f"Response approved and added to training data with action: {request.action}",
            "training_examples": len(TRAINING_DATA),
            "action_stored": request.action,
            "comment_timestamp": comment_created_at.isoformat(),
            "created_time_received": request.created_time or "not provided"
        }
        
    except Exception as e:
        db.rollback()
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        db.close()

@app.get("/batch-status/{job_id}")
def get_batch_status(job_id: str):
    """Get status of a batch processing job"""
    db = SessionLocal()
    try:
        batch_job = db.query(BatchJob).filter(BatchJob.job_id == job_id).first()
        
        if not batch_job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        response = {
            "job_id": batch_job.job_id,
            "status": batch_job.status,
            "total_comments": batch_job.total_comments,
            "processed_comments": batch_job.processed_comments,
            "created_at": batch_job.created_at.isoformat(),
        }
        
        if batch_job.completed_at:
            response["completed_at"] = batch_job.completed_at.isoformat()
        
        if batch_job.status == "completed" and batch_job.results:
            response["results"] = json.loads(batch_job.results)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        return {
            "job_id": job_id,
            "status": "error",
            "error": str(e)
        }
    finally:
        db.close()

@app.get("/stats")
async def get_stats():
    """Get training data statistics"""
    action_counts = {}
    
    for example in TRAINING_DATA:
        action = example.get('action', 'unknown')
        action_counts[action] = action_counts.get(action, 0) + 1
    
    return {
        "total_training_examples": len(TRAINING_DATA),
        "action_distribution": action_counts,
        "data_structure": {
            "type": "PostgreSQL Database",
            "tables": ["fb_posts", "responses", "batch_jobs"],
            "benefits": ["Simple appends", "No duplicates", "Fast queries", "Batch tracking"]
        },
        "supported_actions": {
            "respond": "Generate helpful response (with smart CTA for high-intent)",
            "react": "Add thumbs up or heart reaction", 
            "delete": "Remove spam/inappropriate/non-prospect content",
            "leave_alone": "Ignore harmless off-topic comments"
        },
        "features": {
            "smart_cta": "Phone number for urgent customers, website for general interest",
            "professional_tone": "Natural responses with proper formatting",
            "batch_processing": "Process multiple comments efficiently",
            "no_numerical_values": "Removes all dollar amounts, percentages, and rates"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
