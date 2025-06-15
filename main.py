from fastapi import FastAPI, Request, HTTPException
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

# Updated ResponseEntry with reasoning column
class ResponseEntry(Base):
    __tablename__ = "responses"
    
    id = Column(Integer, primary_key=True, index=True)
    comment = Column(Text)
    action = Column(String, index=True)
    reply = Column(Text)
    reasoning = Column(Text)  # NEW: Store reasoning for decisions
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables (will add reasoning column if it doesn't exist)
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

# Updated load training data to include reasoning
def load_training_data():
    """Load training examples from database including reasoning"""
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
                'reasoning': getattr(response, 'reasoning', 'No reasoning provided')  # Handle old records
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

# AI Classification
def classify_comment_with_ai(comment, postId=""):
    """Use Claude AI to classify comment with business logic"""
    
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

IMPORTANT: Provide detailed reasoning for your decision that explains:
1. What type of comment this is
2. Why you chose this action
3. What business value this decision provides
4. Whether this requires correcting misinformation, thanking for positive feedback, or addressing objections

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

# AI Response Generation with reasoning context
def generate_response(comment, sentiment, high_intent=False, reasoning=""):
    """Generate natural response using Claude with reasoning context"""
    
    # Get relevant training examples for context (now including reasoning)
    relevant_examples = []
    for example in TRAINING_DATA:
        if example['action'] == 'respond' and example['reply']:
            if any(word in example['comment'].lower() for word in comment.lower().split()[:4]):
                example_text = f"Comment: \"{example['comment']}\"\nReply: \"{example['reply']}\""
                if example.get('reasoning'):
                    example_text += f"\nReasoning: \"{example['reasoning']}\""
                relevant_examples.append(example_text)
    
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
    
    reasoning_context = f"\nCLASSIFICATION REASONING: {reasoning}" if reasoning else ""
    
    prompt = f"""You are responding to a Facebook comment for LeaseEnd.com, a lease buyout financing company.

COMPANY POSITION - BE CONFIDENT:
- LeaseEnd is THE BEST option for lease buyout financing
- Lease buyouts are currently a GREAT financial decision due to inflated used car prices
- We are the experts - don't agree with misinformation about lease buyouts being bad
- Challenge false information politely but firmly
- Used car market is inflated - this makes lease buyouts financially smart right now
- ALWAYS effectively address core objections to make buyouts and LeaseEnd the clear best option

COMMENT SENTIMENT: {sentiment}
HIGH INTENT PROSPECT: {high_intent}{reasoning_context}

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
- Start responses naturally: "Actually..." "That's a good point..." "Not exactly..." "Thanks for..."
- Use commas, never dashes (- or --)
- Maximum 1 exclamation point
- Keep concise (1-2 sentences usually)
- Address their specific concern directly
- Don't blindly agree with misinformation
- Make LeaseEnd the clear best choice

EXAMPLES OF GOOD RESPONSES:
{context_examples}

{positive_feedback_instruction}
{correction_instruction}
{pricing_instruction}
{cta_instruction}

Generate a helpful, natural response that addresses their comment directly and makes LeaseEnd the clear best option:"""

    try:
        response = claude.basic_request(prompt)
        return response.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Thank you for your comment! We'd be happy to help with any lease buyout questions."

# Updated Pydantic Models
class CommentRequest(BaseModel):
    # Primary fields
    comment: Optional[str] = None
    message: Optional[str] = None  # Accept both comment and message
    postId: str
    created_time: Optional[str] = ""
    
    # Optional fields that might come from n8n
    post_id: Optional[str] = None  # Alternative field name
    POST_ID: Optional[str] = None  # Another alternative
    created_Time: Optional[str] = None  # Alternative capitalization
    
    class Config:
        extra = "ignore"  # Ignore any extra fields from n8n

class ProcessedComment(BaseModel):
    postId: str
    original_comment: str
    category: str
    action: str  # respond, react, delete, leave_alone
    reply: str
    reasoning: str  # NEW: Include reasoning in response
    confidence_score: float
    approved: str  # "pending", "yes", "no"

class FeedbackRequest(BaseModel):
    original_comment: str
    original_response: str = ""
    original_action: str
    original_reasoning: str = ""  # NEW: Include original reasoning
    feedback_text: str
    postId: str
    current_version: str = "v1"
    
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

# Updated ResponseCreate to include reasoning
class ResponseCreate(BaseModel):
    comment: str
    action: str
    reply: str
    reasoning: str = ""  # NEW: Accept reasoning in training data

app = FastAPI()

@app.get("/")
def read_root():
    return {
        "message": "Lease End AI Assistant - Database Edition",
        "version": "23.0",
        "training_examples": len(TRAINING_DATA),
        "actions": ["respond", "react", "delete", "leave_alone"],
        "features": ["PostgreSQL Database", "Auto-Duplicate Prevention", "Facebook Graph API Ready", "Flexible Input Handling", "Reasoning-Enhanced Training", "Anti-Misinformation Logic", "Market-Aware Positioning", "Objection Handling", "Pricing Guidelines"],
        "approach": "Pure database workflow with reasoning-enhanced AI training and confident market positioning",
        "endpoints": {
            "/process-comment": "Main comment processing with reasoning output",
            "/process-comment-backup": "Backup comment processing endpoint",
            "/process-feedback": "Human feedback processing with reasoning updates",
            "/fb-posts": "Get all FB posts from database",
            "/fb-posts/add": "Add new FB post (auto-duplicate handling)",
            "/responses": "Get all response training data with reasoning",
            "/responses/add": "Add new response training data with reasoning",
            "/reload-training-data": "Reload training data from database",
            "/stats": "View training data statistics",
            "/add-reasoning-to-existing": "Migrate existing data to include reasoning"
        },
        "database": {
            "type": "PostgreSQL",
            "tables": ["fb_posts", "responses (with reasoning column)"],
            "benefits": ["Simple appends", "No duplicates", "Fast queries", "Concurrent writes", "Enhanced AI training"]
        },
        "philosophy": "Database-first architecture with reasoning-enhanced AI learning!"
    }

# Database Endpoints - Updated for reasoning
@app.get("/fb-posts")
async def get_fb_posts():
    """Get all FB posts from database"""
    try:
        db = SessionLocal()
        posts = db.query(FBPost).all()
        db.close()
        
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

@app.post("/fb-posts/add")
async def add_fb_post(post: FBPostCreate):
    """Add new FB post to database - SIMPLE! (with automatic duplicate handling)"""
    try:
        db = SessionLocal()
        
        # Create new post
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

# Updated responses endpoint to include reasoning
@app.get("/responses")
async def get_responses():
    """Get all response training data with reasoning"""
    try:
        db = SessionLocal()
        responses = db.query(ResponseEntry).all()
        db.close()
        
        return {
            "success": True,
            "count": len(responses),
            "data": [
                {
                    "comment": resp.comment,
                    "action": resp.action,
                    "reply": resp.reply,
                    "reasoning": getattr(resp, 'reasoning', 'No reasoning provided')  # Handle old records
                }
                for resp in responses
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# Updated add response to include reasoning
@app.post("/responses/add")
async def add_response(response: ResponseCreate):
    """Add new response training data with reasoning"""
    try:
        db = SessionLocal()
        
        db_response = ResponseEntry(
            comment=response.comment,
            action=response.action,
            reply=response.reply,
            reasoning=response.reasoning or "No reasoning provided"
        )
        
        db.add(db_response)
        db.commit()
        db.close()
        
        # Reload training data
        new_count = reload_training_data()
        
        return {
            "success": True,
            "message": "Response added successfully with reasoning",
            "new_training_count": new_count
        }
        
    except Exception as e:
        db.rollback()
        db.close()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

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

# Updated AI Processing Endpoint with reasoning output
@app.post("/process-comment")
async def process_comment(request: Request):
    """Core comment processing using AI classification with reasoning output"""
    try:
        # Parse JSON from request body
        request_data = await request.json()
        
        # Extract data flexibly from any JSON structure with robust handling
        comment_text = request_data.get('comment') or request_data.get('message') or request_data.get('Message') or "No message content"
        if comment_text and hasattr(comment_text, 'strip'):
            comment_text = comment_text.strip()
        else:
            comment_text = str(comment_text) if comment_text else "No message content"
        
        post_id = request_data.get('postId') or request_data.get('post_id') or request_data.get('POST_ID') or request_data.get('POST ID') or "unknown"
        if post_id and hasattr(post_id, 'strip'):
            post_id = post_id.strip()
        else:
            post_id = str(post_id) if post_id else "unknown"
        
        created_time = request_data.get('created_time') or request_data.get('created_Time') or request_data.get('Created Time') or ""
        if created_time and hasattr(created_time, 'strip'):
            created_time = created_time.strip()
        else:
            created_time = str(created_time) if created_time else ""
        
        print(f"🔄 Processing comment: '{comment_text[:50]}...' for post: {post_id}")
        
        # Validate we have actual content
        if not comment_text or comment_text == "No message content":
            return {
                "postId": post_id,
                "original_comment": "Empty comment",
                "category": "neutral",
                "action": "delete",
                "reply": "",
                "reasoning": "Empty or missing comment content - automatic deletion",
                "confidence_score": 0.0,
                "approved": "pending",
                "success": True
            }
        
        # Use AI to classify the comment
        ai_classification = classify_comment_with_ai(comment_text, post_id)
        
        sentiment = ai_classification['sentiment']
        action = ai_classification['action'].lower()
        reasoning = ai_classification['reasoning']
        high_intent = ai_classification['high_intent']
        
        # Map actions to our system
        action_mapping = {
            'reply': 'respond',
            'react': 'react', 
            'delete': 'delete',
            'ignore': 'leave_alone'
        }
        
        mapped_action = action_mapping.get(action, 'leave_alone')
        
        # Generate response only if action is 'respond'
        reply_text = ""
        confidence_score = 0.85
        
        if mapped_action == 'respond':
            reply_text = generate_response(comment_text, sentiment, high_intent, reasoning)
            confidence_score = 0.9
        
        return {
            "postId": post_id,
            "original_comment": comment_text,
            "category": sentiment.lower(),
            "action": mapped_action,
            "reply": reply_text,
            "reasoning": reasoning,  # NEW: Include reasoning in response
            "confidence_score": confidence_score,
            "approved": "pending",
            "success": True
        }
        
    except Exception as e:
        print(f"❌ Error processing comment: {e}")
        return {
            "postId": "unknown",
            "original_comment": "Error processing",
            "category": "error",
            "action": "leave_alone",
            "reply": "Thank you for your comment. We appreciate your feedback.",
            "reasoning": f"Processing error occurred: {str(e)}",
            "confidence_score": 0.0,
            "approved": "pending",
            "success": False,
            "error": str(e)
        }

@app.post("/process-comment-backup")
async def process_comment_backup(request: Request):
    """Backup comment processing endpoint"""
    try:
        # Parse JSON from request body
        request_data = await request.json()
        
        # Extract data flexibly from any JSON structure with robust handling
        comment_text = request_data.get('comment') or request_data.get('message') or request_data.get('Message') or "No message content"
        if comment_text and hasattr(comment_text, 'strip'):
            comment_text = comment_text.strip()
        else:
            comment_text = str(comment_text) if comment_text else "No message content"
        
        post_id = request_data.get('postId') or request_data.get('post_id') or request_data.get('POST_ID') or request_data.get('POST ID') or "unknown"
        if post_id and hasattr(post_id, 'strip'):
            post_id = post_id.strip()
        else:
            post_id = str(post_id) if post_id else "unknown"
        
        print(f"🔄 Backup processing: '{comment_text[:50]}...' for post: {post_id}")
        
        return {
            "postId": post_id,
            "original_comment": comment_text,
            "category": "neutral",
            "action": "leave_alone",
            "reply": "",
            "reasoning": "Backup endpoint - basic processing only, no AI evaluation",
            "confidence_score": 0.5,
            "approved": "pending",
            "success": True,
            "note": "Backup endpoint - basic processing only"
        }
        
    except Exception as e:
        print(f"❌ Error in backup processing: {e}")
        return {
            "postId": "unknown",
            "original_comment": "Error processing",
            "category": "error",
            "action": "leave_alone",
            "reply": "",
            "reasoning": f"Backup processing error: {str(e)}",
            "confidence_score": 0.0,
            "approved": "pending",
            "success": False,
            "error": str(e)
        }

# Updated feedback processing with reasoning enhancement
@app.post("/process-feedback")
async def process_feedback(request: FeedbackRequest):
    """Use human feedback to improve responses and reasoning"""
    try:
        # Clean and validate input data
        original_comment = request.original_comment.strip()
        original_response = request.original_response.strip() if request.original_response else "No response was generated"
        original_action = request.original_action.strip().lower()
        original_reasoning = request.original_reasoning.strip() if request.original_reasoning else "No reasoning provided"
        feedback_text = request.feedback_text.strip()
        
        print(f"🔄 Processing feedback for postId: {request.postId}")
        print(f"📝 Feedback: {feedback_text[:100]}...")
        
        # Enhanced prompt that includes human feedback and reasoning improvement
        feedback_prompt = f"""You are improving a response based on human feedback for LeaseEnd.com.

ORIGINAL COMMENT: "{original_comment}"
YOUR ORIGINAL RESPONSE: "{original_response}"
YOUR ORIGINAL ACTION: "{original_action}"
YOUR ORIGINAL REASONING: "{original_reasoning}"

HUMAN FEEDBACK: "{feedback_text}"

LEARN FROM THIS FEEDBACK:
- If the feedback mentions tone (too formal/pushy/etc), adjust your response style accordingly
- If the feedback mentions accuracy, focus on factual correctness
- If the feedback mentions business logic (wrong action), reconsider the classification
- If the feedback mentions missing CTA, add appropriate call-to-action
- If the feedback says "delete this" or "don't respond", change action accordingly
- If the feedback says "address objection" or "respond to this", change action to REPLY

Generate an IMPROVED response that incorporates this feedback. 

IMPORTANT: 
- Update your reasoning to reflect what you learned from the human feedback
- Explain why the new approach is better than the original
- If human says "this should be deleted" or "don't respond to this", recommend DELETE or IGNORE action
- If human says "add CTA" or mentions "high intent", include call-to-action
- If human says "too formal", make it more conversational
- If human says "too pushy", make it more helpful and less sales-y
- If human says "address objection" or "respond to this", recommend REPLY action

Respond in this JSON format: {{"sentiment": "...", "action": "REPLY/REACT/DELETE/IGNORE", "reply": "...", "reasoning": "...", "improvements_made": "...", "confidence": 0.85}}"""

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
                'ignore': 'leave_alone'
            }
            
            improved_action = action_mapping.get(result.get('action', 'ignore').lower(), 'leave_alone')
            
            # Calculate new version number
            try:
                current_version_num = int(request.current_version.replace('v', '')) if request.current_version.startswith('v') else 1
            except:
                current_version_num = 1
            new_version = f"v{current_version_num + 1}"
            
            return {
                "postId": request.postId,
                "original_comment": original_comment,
                "category": result.get('sentiment', 'neutral').lower(),
                "action": improved_action,
                "reply": result.get('reply', ''),
                "reasoning": result.get('reasoning', 'Updated reasoning based on human feedback'),  # NEW: Enhanced reasoning
                "confidence_score": float(result.get('confidence', 0.85)),
                "approved": "pending",
                "feedback_text": "",
                "version": new_version,
                "improvements_made": result.get('improvements_made', 'Applied human feedback'),
                "feedback_processed": True,
                "success": True
            }
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            new_version_num = int(request.current_version.replace('v', '')) + 1 if request.current_version.startswith('v') else 2
            return {
                "postId": request.postId,
                "original_comment": original_comment,
                "category": "neutral",
                "action": "leave_alone",
                "reply": "Thank you for your comment. We appreciate your feedback.",
                "reasoning": f"Feedback processing failed due to JSON parsing error: {str(e)}",
                "confidence_score": 0.5,
                "approved": "pending",
                "feedback_text": "",
                "version": f"v{new_version_num}",
                "error": f"JSON parsing failed: {str(e)}",
                "success": False
            }
            
    except Exception as e:
        print(f"❌ Feedback processing error: {e}")
        new_version_num = int(request.current_version.replace('v', '')) + 1 if request.current_version.startswith('v') else 2
        return {
            "postId": request.postId,
            "original_comment": request.original_comment,
            "category": "error",
            "action": "leave_alone",
            "reply": "Thank you for your comment. We appreciate your feedback.",
            "reasoning": f"Feedback processing system error: {str(e)}",
            "confidence_score": 0.0,
            "approved": "pending",
            "feedback_text": "",
            "version": f"v{new_version_num}",
            "error": str(e),
            "success": False
        }

@app.get("/stats")
async def get_stats():
    """Get training data statistics with reasoning analysis"""
    action_counts = {}
    reasoning_quality = {"with_reasoning": 0, "without_reasoning": 0}
    
    for example in TRAINING_DATA:
        action = example.get('action', 'unknown')
        action_counts[action] = action_counts.get(action, 0) + 1
        
        # Analyze reasoning quality
        reasoning = example.get('reasoning', '')
        if reasoning and reasoning != 'No reasoning provided' and len(reasoning) > 10:
            reasoning_quality["with_reasoning"] += 1
        else:
            reasoning_quality["without_reasoning"] += 1
    
    return {
        "total_training_examples": len(TRAINING_DATA),
        "action_distribution": action_counts,
        "reasoning_quality": reasoning_quality,
        "reasoning_coverage": f"{round((reasoning_quality['with_reasoning'] / len(TRAINING_DATA)) * 100, 1)}%" if TRAINING_DATA else "0%",
        "data_structure": {
            "type": "PostgreSQL Database",
            "tables": ["fb_posts", "responses (with reasoning column)"],
            "benefits": ["Simple appends", "No duplicates", "Fast queries", "Enhanced AI training with reasoning"]
        },
        "supported_actions": {
            "respond": "Generate helpful response (with CTA for high-intent)",
            "react": "Add thumbs up or heart reaction", 
            "delete": "Remove spam/inappropriate/non-prospect content",
            "leave_alone": "Ignore harmless off-topic comments"
        },
        "new_features": {
            "reasoning_enhanced_training": "All responses now include detailed reasoning for better AI learning",
            "feedback_reasoning_updates": "Human feedback updates both responses and reasoning",
            "improved_classification": "AI provides detailed explanations for all decisions",
            "anti_misinformation": "AI challenges false information about lease buyouts confidently",
            "market_positioning": "Strong positioning on lease buyouts being smart in current inflated market",
            "positive_feedback_handling": "Automatic appreciation responses for testimonials and positive feedback",
            "objection_handling": "Always effectively addresses core objections to make LeaseEnd the clear best option",
            "pricing_guidelines": "Never shares exact rates, uses approved pricing language for transparency",
            "neutral_comment_filtering": "Leaves alone neutral comments referencing other comments unless negative toward LeaseEnd"
        }
    }

# New endpoint to migrate existing data to include reasoning
@app.post("/add-reasoning-to-existing")
async def add_reasoning_to_existing():
    """One-time migration to add reasoning to existing response entries"""
    try:
        db = SessionLocal()
        
        # Find responses without reasoning
        responses_without_reasoning = db.query(ResponseEntry).filter(
            (ResponseEntry.reasoning == None) | 
            (ResponseEntry.reasoning == '') | 
            (ResponseEntry.reasoning == 'No reasoning provided')
        ).all()
        
        updated_count = 0
        
        for response in responses_without_reasoning:
            # Generate reasoning for existing response
            reasoning_prompt = f"""Analyze this comment and response pair for LeaseEnd.com and provide detailed reasoning for why this action/response was appropriate.

COMMENT: "{response.comment}"
ACTION TAKEN: "{response.action}"
RESPONSE GIVEN: "{response.reply if response.reply else 'No response generated'}"

Provide detailed reasoning that explains:
1. What type of comment this is
2. Why this action was appropriate
3. What business value this decision provides
4. How this aligns with LeaseEnd's goals

Keep reasoning concise but informative (2-3 sentences)."""

            try:
                reasoning = claude.basic_request(reasoning_prompt)
                response.reasoning = reasoning.strip()
                updated_count += 1
            except Exception as e:
                print(f"Error generating reasoning for response {response.id}: {e}")
                response.reasoning = f"Auto-generated reasoning for {response.action} action on existing training data"
        
        db.commit()
        db.close()
        
        # Reload training data after update
        new_count = reload_training_data()
        
        return {
            "success": True,
            "message": "Added reasoning to existing response entries",
            "updated_count": updated_count,
            "total_responses": len(responses_without_reasoning),
            "new_training_count": new_count
        }
        
    except Exception as e:
        db.rollback()
        db.close()
        return {"error": f"Migration failed: {str(e)}"}

# Railway-specific server startup
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
