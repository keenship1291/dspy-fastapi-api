from fastapi import FastAPI, Request, HTTPException
import os
from datetime import datetime, timezone
from anthropic import Anthropic
import dspy
from pydantic import BaseModel
import json
from typing import List, Dict, Optional
import uuid

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

class BatchJob(Base):
    __tablename__ = "batch_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True)
    status = Column(String, default="processing")  # processing, completed, failed
    total_comments = Column(Integer, default=0)
    processed_comments = Column(Integer, default=0)
    results = Column(Text)  # JSON string of results
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

# Create tables
Base.metadata.create_all(bind=engine)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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

def classify_comment_with_ai(comment, commentId=""):
    """Use Claude AI to classify comment"""
    
    prompt = f"""You are analyzing comments for LeaseEnd.com, which specializes in lease buyout services. 

BUSINESS LOGIC & COMPANY POSITION:
- LeaseEnd is THE BEST option for lease buyout financing - be confident about this
- Lease buyouts are currently a GREAT financial decision due to inflated used car prices
- Never agree with false information about lease buyouts being bad financial decisions
- Challenge misinformation politely but firmly - we have the expertise
- Always effectively address core objections to make buyouts and LeaseEnd the clear best option

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
            'reasoning': f'Classification error: {str(e)}',
            'high_intent': False
        }

def generate_response(comment, sentiment, high_intent=False):
    """Generate natural response using Claude"""
    
    prompt = f"""You are responding to a Facebook comment for LeaseEnd.com, a lease buyout financing company.

COMPANY POSITION - BE CONFIDENT:
- LeaseEnd is THE BEST option for lease buyout financing
- Lease buyouts are currently a GREAT financial decision due to inflated used car prices
- We are the experts - don't agree with misinformation about lease buyouts being bad
- Challenge false information politely but firmly
- Used car market is inflated - this makes lease buyouts financially smart right now

COMMENT SENTIMENT: {sentiment}
HIGH INTENT PROSPECT: {high_intent}

ORIGINAL COMMENT: "{comment}"

RESPONSE STYLE:
- Sound natural and human
- Get straight to the point
- NEVER use dashes (-), em dashes (—), or en dashes (–) anywhere in responses
- Use commas, periods, and semicolons for punctuation instead
- Maximum 1 exclamation point
- Keep concise (1-2 sentences usually)
- Address their specific concern directly
- Make LeaseEnd the clear best choice

Generate a helpful, natural response:"""

    try:
        response = claude.basic_request(prompt)
        
        # Clean response to remove any dashes
        cleaned_response = response.strip()
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

class BatchStatusResponse(BaseModel):
    job_id: str
    status: str
    total_comments: int
    processed_comments: int
    results: Optional[List[Dict]] = None

app = FastAPI()

@app.get("/")
def read_root():
    return {
        "message": "Lease End AI Assistant - BATCH VERSION",
        "version": "21.0-BATCH",
        "training_examples": len(TRAINING_DATA),
        "status": "RUNNING",
        "endpoints": {
            "/process-batch": "Batch comment processing",
            "/batch-status/{job_id}": "Check batch status",
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

@app.post("/process-batch")
async def process_batch(request: BatchCommentRequest):
    """Process multiple comments in a batch and store results"""
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create batch job record
        db = SessionLocal()
        batch_job = BatchJob(
            job_id=job_id,
            status="processing",
            total_comments=len(request.comments),
            processed_comments=0
        )
        db.add(batch_job)
        db.commit()
        
        # Process all comments
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
                    # Classify comment
                    ai_classification = classify_comment_with_ai(comment_text, comment_id)
                    
                    sentiment = ai_classification['sentiment']
                    action = ai_classification['action'].lower()
                    reasoning = ai_classification['reasoning']
                    high_intent = ai_classification['high_intent']
                    
                    # Map actions
                    action_mapping = {
                        'reply': 'respond',
                        'react': 'react', 
                        'delete': 'delete',
                        'ignore': 'leave_alone'
                    }
                    
                    mapped_action = action_mapping.get(action, 'leave_alone')
                    
                    # Generate response if needed
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
                
                # Update progress
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
        
        # Mark job as completed and store results
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
        # Mark job as failed
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

@app.get("/batch-status/{job_id}")
def get_batch_status(job_id: str):
    """Get status of a batch processing job"""
    try:
        db = SessionLocal()
        batch_job = db.query(BatchJob).filter(BatchJob.job_id == job_id).first()
        db.close()
        
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

# Keep the original single comment endpoint for backward compatibility
@app.post("/process-comment")
async def process_comment(request: Comment):
    """Process a single comment (legacy endpoint)"""
    batch_request = BatchCommentRequest(comments=[request])
    result = await process_batch(batch_request)
    
    if result.get("results") and len(result["results"]) > 0:
        return result["results"][0]
    else:
        return {
            "commentId": request.commentId,
            "original_comment": "Error processing",
            "category": "error",
            "action": "leave_alone",
            "reply": "Thank you for your comment.",
            "confidence_score": 0.0,
            "approved": "pending",
            "success": False
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
