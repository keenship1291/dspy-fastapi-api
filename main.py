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

class ResponseEntry(Base):
    __tablename__ = "responses"
    
    id = Column(Integer, primary_key=True, index=True)
    comment = Column(Text)
    action = Column(String, index=True)
    reply = Column(Text)
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
                'reply': response.reply
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

BUSINESS LOGIC:
- Comments that negatively mention leasing alternatives (e.g., 'Trading in a lease is bad', 'Dealerships rip you off') should be classified as Positive, as they imply LeaseEnd's services are better
- Comments asking questions about lease buyouts, rates, or process are high-intent prospects
- Comments from people who clearly don't lease vehicles should be ignored/deleted
- Spam, irrelevant, or off-topic comments should be deleted

Analyze this comment and classify its sentiment as Positive, Neutral, or Negative. Then recommend one action: REPLY, REACT, DELETE, or IGNORE.

ACTIONS:
- REPLY: For questions, objections, or potential customers (generate helpful response)
- REACT: For positive comments, praise, or simple acknowledgments  
- DELETE: For spam, inappropriate content, or clearly non-prospects
- IGNORE: For off-topic but harmless comments

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
            elif 'REPLY' in response.upper():
                action = 'REPLY'
            elif 'REACT' in response.upper():
                action = 'REACT'
            else:
                action = 'IGNORE'
                
            return {
                'sentiment': 'Neutral',
                'action': action,
                'reasoning': 'Fallback classification',
                'high_intent': False
            }
            
    except Exception as e:
        print(f"Error in AI classification: {e}")
        return {
            'sentiment': 'Neutral',
            'action': 'IGNORE',
            'reasoning': 'Classification error',
            'high_intent': False
        }

# AI Response Generation
def generate_response(comment, sentiment, high_intent=False):
    """Generate natural response using Claude"""
    
    # Get relevant training examples for context
    relevant_examples = []
    for example in TRAINING_DATA:
        if example['action'] == 'respond' and example['reply']:
            if any(word in example['comment'].lower() for word in comment.lower().split()[:4]):
                relevant_examples.append(f"Comment: \"{example['comment']}\"\nReply: \"{example['reply']}\"")
    
    context_examples = "\n\n".join(relevant_examples[:3])
    
    # Add CTA instructions for high-intent comments
    cta_instruction = ""
    if high_intent:
        cta_instruction = "\nFor high-intent prospects, end with: 'To see your options just fill out the form on our site, we're happy to help'"
    
    prompt = f"""You are responding to a Facebook comment for LeaseEnd.com, a lease buyout financing company.

COMMENT SENTIMENT: {sentiment}
HIGH INTENT PROSPECT: {high_intent}

ORIGINAL COMMENT: "{comment}"

BRAND VOICE:
- Professional but conversational
- Transparent about pricing (no hidden fees)
- Helpful, not pushy
- Emphasize online process convenience

RESPONSE STYLE:
- Sound natural and human
- Start responses naturally: "Actually..." "That's a good point..." "Not exactly..."
- Use commas, never dashes (- or --)
- Maximum 1 exclamation point
- Keep concise (1-2 sentences usually)
- Address their specific concern directly

EXAMPLES OF GOOD RESPONSES:
{context_examples}

{cta_instruction}

Generate a helpful, natural response that addresses their comment directly:"""

    try:
        response = claude.basic_request(prompt)
        return response.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Thank you for your comment! We'd be happy to help with any lease buyout questions."

# Pydantic Models
class CommentRequest(BaseModel):
    comment: str
    postId: str
    created_time: str = ""
    memory_context: str = ""

class ProcessedComment(BaseModel):
    postId: str
    original_comment: str
    category: str
    action: str  # respond, react, delete, leave_alone
    reply: str
    confidence_score: float
    approved: str  # "pending", "yes", "no"

class FeedbackRequest(BaseModel):
    original_comment: str
    original_response: str = ""
    original_action: str
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

class ResponseCreate(BaseModel):
    comment: str
    action: str
    reply: str

app = FastAPI()

@app.get("/")
def read_root():
    return {
        "message": "Lease End AI Assistant - Database Edition",
        "version": "17.0",
        "training_examples": len(TRAINING_DATA),
        "actions": ["respond", "react", "delete", "leave_alone"],
        "features": ["PostgreSQL Database", "Simple Appends", "No CSV Complexity"],
        "approach": "Database for everything - fast, simple, reliable",
        "endpoints": {
            "/process-comment": "Initial comment processing",
            "/process-feedback": "Human feedback processing",
            "/fb-posts": "Get all FB posts",
            "/fb-posts/add": "Add new FB post (simple!)",
            "/responses": "Get all response data",
            "/responses/add": "Add new response data",
            "/stats": "View training data statistics"
        },
        "database": {
            "type": "PostgreSQL",
            "tables": ["fb_posts", "responses"],
            "benefits": ["Simple appends", "No duplicates", "Fast queries", "Concurrent writes"]
        },
        "philosophy": "Databases for data, not CSV files!"
    }

# Database Endpoints - Super Simple!
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
    """Add new FB post to database - SIMPLE!"""
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
            "post_id": post.post_id
        }
        
    except IntegrityError:
        db.rollback()
        db.close()
        return {
            "success": False,
            "message": f"Post {post.post_id} already exists (duplicate prevented)"
        }
    except Exception as e:
        db.rollback()
        db.close()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/responses")
async def get_responses():
    """Get all response training data"""
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
                    "reply": resp.reply
                }
                for resp in responses
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/responses/add")
async def add_response(response: ResponseCreate):
    """Add new response training data"""
    try:
        db = SessionLocal()
        
        db_response = ResponseEntry(
            comment=response.comment,
            action=response.action,
            reply=response.reply
        )
        
        db.add(db_response)
        db.commit()
        db.close()
        
        # Reload training data
        new_count = reload_training_data()
        
        return {
            "success": True,
            "message": "Response added successfully",
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

# Original AI Processing Endpoints (unchanged)
@app.post("/process-comment", response_model=ProcessedComment)
async def process_comment(request: CommentRequest):
    """Core comment processing using AI classification"""
    try:
        # Use AI to classify the comment
        ai_classification = classify_comment_with_ai(request.comment, request.postId)
        
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
            reply_text = generate_response(request.comment, sentiment, high_intent)
            confidence_score = 0.9
        
        return ProcessedComment(
            postId=request.postId,
            original_comment=request.comment,
            category=sentiment.lower(),
            action=mapped_action,
            reply=reply_text,
            confidence_score=confidence_score,
            approved="pending"
        )
        
    except Exception as e:
        print(f"Error processing comment: {e}")
        return ProcessedComment(
            postId=request.postId,
            original_comment=request.comment,
            category="error",
            action="leave_alone",
            reply="Thank you for your comment. We appreciate your feedback.",
            confidence_score=0.0,
            approved="pending"
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
        
        print(f"🔄 Processing feedback for postId: {request.postId}")
        print(f"📝 Feedback: {feedback_text[:100]}...")
        
        # Enhanced prompt that includes human feedback
        feedback_prompt = f"""You are improving a response based on human feedback for LeaseEnd.com.

ORIGINAL COMMENT: "{original_comment}"
YOUR ORIGINAL RESPONSE: "{original_response}"
YOUR ORIGINAL ACTION: "{original_action}"

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
- If human says "this should be deleted" or "don't respond to this", recommend DELETE or IGNORE action
- If human says "add CTA" or mentions "high intent", include call-to-action
- If human says "too formal", make it more conversational
- If human says "too pushy", make it more helpful and less sales-y
- If human says "address objection" or "respond to this", recommend REPLY action

Respond in this JSON format: {{"sentiment": "...", "action": "REPLY/REACT/DELETE/IGNORE", "reply": "...", "improvements_made": "...", "confidence": 0.85}}"""

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
            "confidence_score": 0.0,
            "approved": "pending",
            "feedback_text": "",
            "version": f"v{new_version_num}",
            "error": str(e),
            "success": False
        }

@app.post("/migrate-from-github")
async def migrate_from_github():
    """One-time migration from GitHub CSV to database"""
    try:
        import requests
        import csv
        import base64
        
        # GitHub file URL (you might need to add GITHUB_TOKEN back temporarily)
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            return {"error": "Need GITHUB_TOKEN environment variable for migration"}
        
        headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        # Get CSV from GitHub
        url = "https://api.github.com/repos/dten111213/dspy-fastapi-api/contents/active_fb_post_id.csv"
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            return {"error": f"Failed to fetch GitHub CSV: {response.status_code}"}
        
        # Decode CSV content
        csv_content = base64.b64decode(response.json()['content']).decode('utf-8')
        
        # Parse CSV
        csv_reader = csv.DictReader(csv_content.strip().split('\n'))
        
        db = SessionLocal()
        migrated_count = 0
        duplicate_count = 0
        
        for row in csv_reader:
            try:
                # Create new post
                db_post = FBPost(
                    ad_account_name=row.get('Ad account name', ''),
                    campaign_name=row.get('Campaign name', ''),
                    ad_set_name=row.get('Ad set name', ''),
                    ad_name=row.get('Ad name', ''),
                    page_id=row.get('Page ID', ''),
                    post_id=row.get('Post Id', ''),
                    object_story_id=row.get('Object Story ID', '')
                )
                
                db.add(db_post)
                db.commit()
                migrated_count += 1
                
            except IntegrityError:
                db.rollback()
                duplicate_count += 1
                continue
        
        db.close()
        
        return {
            "success": True,
            "message": "Migration completed",
            "migrated_count": migrated_count,
            "duplicate_count": duplicate_count,
            "total_processed": migrated_count + duplicate_count
        }
        
    except Exception as e:
        return {"error": f"Migration failed: {str(e)}"}

@app.post("/deduplicate-posts")
async def deduplicate_posts():
    """Remove duplicate posts, keeping only one per Object Story ID"""
    try:
        db = SessionLocal()
        
        # Get all posts grouped by object_story_id
        from sqlalchemy import func
        
        # Find duplicate object_story_ids
        duplicate_groups = db.query(
            FBPost.object_story_id,
            func.count(FBPost.id).label('count'),
            func.min(FBPost.id).label('keep_id')
        ).group_by(FBPost.object_story_id).having(func.count(FBPost.id) > 1).all()
        
        total_duplicates_removed = 0
        
        for group in duplicate_groups:
            object_story_id = group.object_story_id
            keep_id = group.keep_id
            duplicate_count = group.count - 1  # -1 because we keep one
            
            # Delete all duplicates except the one we want to keep
            deleted = db.query(FBPost).filter(
                FBPost.object_story_id == object_story_id,
                FBPost.id != keep_id
            ).delete()
            
            total_duplicates_removed += deleted
            
            print(f"Removed {deleted} duplicates for Object Story ID: {object_story_id}")
        
        db.commit()
        
        # Get final count
        remaining_count = db.query(FBPost).count()
        db.close()
        
        return {
            "success": True,
            "message": "Deduplication completed",
            "duplicates_removed": total_duplicates_removed,
            "remaining_posts": remaining_count,
            "duplicate_groups_processed": len(duplicate_groups)
        }
        
    except Exception as e:
        db.rollback()
        db.close()
        return {"error": f"Deduplication failed: {str(e)}"}

@app.get("/duplicate-stats")
async def get_duplicate_stats():
    """Check how many duplicates exist based on Object Story ID"""
    try:
        db = SessionLocal()
        
        from sqlalchemy import func
        
        # Get duplicate statistics
        total_posts = db.query(FBPost).count()
        
        duplicate_groups = db.query(
            FBPost.object_story_id,
            func.count(FBPost.id).label('count')
        ).group_by(FBPost.object_story_id).having(func.count(FBPost.id) > 1).all()
        
        total_duplicates = sum(group.count - 1 for group in duplicate_groups)
        unique_posts = total_posts - total_duplicates
        
        db.close()
        
        return {
            "success": True,
            "total_posts": total_posts,
            "unique_posts": unique_posts,
            "duplicate_posts": total_duplicates,
            "duplicate_groups": len(duplicate_groups),
            "duplicate_details": [
                {
                    "object_story_id": group.object_story_id,
                    "duplicate_count": group.count
                }
                for group in duplicate_groups[:10]  # Show first 10 examples
            ]
        }
        
    except Exception as e:
        return {"error": f"Failed to get duplicate stats: {str(e)}"}

@app.post("/migrate-responses-from-github")
async def migrate_responses_from_github():
    """One-time migration of response data from GitHub CSV to database"""
    try:
        import requests
        import csv
        import base64
        
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            return {"error": "Need GITHUB_TOKEN environment variable for migration"}
        
        headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        # Get CSV from GitHub
        url = "https://api.github.com/repos/dten111213/dspy-fastapi-api/contents/response_database.csv"
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            return {"error": f"Failed to fetch GitHub CSV: {response.status_code}"}
        
        # Decode CSV content
        csv_content = base64.b64decode(response.json()['content']).decode('utf-8')
        
        # Parse CSV
        csv_reader = csv.DictReader(csv_content.strip().split('\n'))
        
        db = SessionLocal()
        migrated_count = 0
        
        for row in csv_reader:
            try:
                # Create new response
                db_response = ResponseEntry(
                    comment=row.get('comment', ''),
                    action=row.get('action', ''),
                    reply=row.get('reply', '')
                )
                
                db.add(db_response)
                db.commit()
                migrated_count += 1
                
            except Exception as e:
                db.rollback()
                print(f"Error migrating response: {e}")
                continue
        
        db.close()
        
        # Reload training data after migration
        new_count = reload_training_data()
        
        return {
            "success": True,
            "message": "Response migration completed",
            "migrated_count": migrated_count,
            "new_training_count": new_count
        }
        
    except Exception as e:
        return {"error": f"Response migration failed: {str(e)}"}
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
            "tables": ["fb_posts", "responses"],
            "benefits": ["Simple appends", "No duplicates", "Fast queries"]
        },
        "supported_actions": {
            "respond": "Generate helpful response (with CTA for high-intent)",
            "react": "Add thumbs up or heart reaction", 
            "delete": "Remove spam/inappropriate/non-prospect content",
            "leave_alone": "Ignore harmless off-topic comments"
        }
    }

# Railway-specific server startup
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
