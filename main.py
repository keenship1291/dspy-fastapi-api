from fastapi import FastAPI, Request, HTTPException
import os
import csv
from datetime import datetime, timezone
from anthropic import Anthropic
import dspy
from pydantic import BaseModel
import requests
import json
from typing import List, Dict, Optional
import base64

# Load API key from environment variable (set in Railway dashboard)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")

# Facebook Config (simplified - token is now unlimited)
FACEBOOK_APP_ID = os.getenv("FACEBOOK_APP_ID")
FACEBOOK_APP_SECRET = os.getenv("FACEBOOK_APP_SECRET") 
FACEBOOK_PAGE_ID = os.getenv("FACEBOOK_PAGE_ID")
FACEBOOK_PAGE_TOKEN = os.getenv("FACEBOOK_PAGE_TOKEN")

# GitHub Configuration - Only for reading CSV files
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Personal Access Token
GITHUB_OWNER = os.getenv("GITHUB_OWNER", "dten213")  # Your GitHub username
GITHUB_REPO = os.getenv("GITHUB_REPO", "dspy-fastapi-api")  # Your repo name
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")  # Branch to read from

# CSV File Paths
RESPONSE_DATABASE_CSV = "response_database.csv"
ACTIVE_FB_POST_ID_CSV = "active_fb_post_id.csv"

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

# Smart CSV sync: Only sync if files don't exist locally
def smart_csv_sync():
    """Only sync from GitHub if local files don't exist"""
    files_to_sync = []
    
    for csv_file in [RESPONSE_DATABASE_CSV, ACTIVE_FB_POST_ID_CSV]:
        if not os.path.exists(csv_file):
            files_to_sync.append(csv_file)
    
    if files_to_sync:
        print(f"🔄 Missing local files, syncing from GitHub: {files_to_sync}")
        sync_csv_from_github()
    else:
        print("✅ Local CSV files exist, skipping GitHub sync")

# Smart startup sync
print("🚀 Starting up with smart CSV sync...")
smart_csv_sync()

# GitHub Integration Functions
def download_file_from_github(file_path):
    """Download a file from GitHub repository"""
    try:
        if not GITHUB_TOKEN:
            print(f"⚠️ No GitHub token found, cannot download {file_path}")
            return None
        
        # GitHub API headers
        headers = {
            'Authorization': f'token {GITHUB_TOKEN}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        # Get file from GitHub
        get_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{file_path}"
        response = requests.get(get_url, headers=headers)
        
        if response.status_code == 200:
            file_data = response.json()
            # Decode base64 content
            content = base64.b64decode(file_data['content']).decode('utf-8')
            print(f"✅ Downloaded {file_path} from GitHub ({len(content)} characters)")
            return content
        elif response.status_code == 404:
            print(f"📝 File {file_path} not found in GitHub, will create new")
            return None
        else:
            print(f"❌ Error downloading {file_path}: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading from GitHub: {e}")
        return None

def sync_csv_from_github():
    """Download and sync CSV files from GitHub to Railway on startup"""
    print("🔄 Syncing CSV files from GitHub...")
    
    files_synced = 0
    
    for csv_file in [RESPONSE_DATABASE_CSV, ACTIVE_FB_POST_ID_CSV]:
        try:
            # Download from GitHub
            github_content = download_file_from_github(csv_file)
            
            if github_content:
                # Write to local Railway file
                with open(csv_file, 'w', encoding='utf-8') as file:
                    file.write(github_content)
                
                # Count lines for verification
                lines = github_content.count('\n')
                print(f"✅ Synced {csv_file}: {lines} lines from GitHub → Railway")
                files_synced += 1
            else:
                # Create empty file with headers if not found in GitHub
                if csv_file == RESPONSE_DATABASE_CSV:
                    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow(['comment', 'action', 'reply'])
                    print(f"📝 Created new {csv_file} with headers")
                elif csv_file == ACTIVE_FB_POST_ID_CSV:
                    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow(['Ad account name', 'Campaign name', 'Ad set name', 'Ad name', 'Page ID', 'Post Id', 'Object Story ID'])
                    print(f"📝 Created new {csv_file} with headers")
                
        except Exception as e:
            print(f"❌ Error syncing {csv_file}: {e}")
    
    print(f"🎯 GitHub sync complete: {files_synced} files synced from GitHub")
    return files_synced
def commit_file_to_github(file_path, content, commit_message):
    """Commit a file to GitHub repository"""
    try:
        if not GITHUB_TOKEN:
            print("⚠️ No GitHub token found, skipping GitHub commit")
            return False
        
        # GitHub API headers
        headers = {
            'Authorization': f'token {GITHUB_TOKEN}',
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json'
        }
        
        # Get current file SHA (required for updating existing files)
        get_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{file_path}"
        get_response = requests.get(get_url, headers=headers)
        
        sha = None
        if get_response.status_code == 200:
            sha = get_response.json().get('sha')
            print(f"📋 Found existing file SHA: {sha[:7]}...")
        elif get_response.status_code == 404:
            print(f"📝 Creating new file: {file_path}")
        else:
            print(f"❌ Error getting file info: {get_response.status_code}")
            return False
        
        # Encode content to base64
        content_encoded = base64.b64encode(content.encode('utf-8')).decode('utf-8')
        
        # Prepare commit data
        commit_data = {
            'message': commit_message,
            'content': content_encoded,
            'branch': GITHUB_BRANCH
        }
        
        # Add SHA if updating existing file
        if sha:
            commit_data['sha'] = sha
        
        # Commit to GitHub
        put_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{file_path}"
        put_response = requests.put(put_url, headers=headers, json=commit_data)
        
        if put_response.status_code in [200, 201]:
            commit_sha = put_response.json().get('commit', {}).get('sha', 'unknown')
            print(f"✅ Successfully committed to GitHub: {commit_sha[:7]}...")
            return True
        else:
            print(f"❌ Failed to commit to GitHub: {put_response.status_code}")
            print(f"Response: {put_response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error committing to GitHub: {e}")
        return False

def read_csv_as_string(file_path):
    """Read CSV file and return as string"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"⚠️ File not found: {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None
# CSV Management Functions - Read Only
def read_response_database():
    """Read training data from GitHub CSV or create empty structure"""
    try:
        # Try to download from GitHub first
        github_content = download_file_from_github(RESPONSE_DATABASE_CSV)
        
        if github_content:
            # Parse CSV content from GitHub
            lines = github_content.strip().split('\n')
            if len(lines) > 1:  # Has header + data
                reader = csv.DictReader(lines)
                data = list(reader)
                print(f"✅ Loaded {len(data)} training examples from GitHub")
                return data
        
        # Return empty list if no data
        print("📝 No training data found in GitHub, starting with empty dataset")
        return []
        
    except Exception as e:
        print(f"❌ Error reading response database from GitHub: {e}")
        return []

def read_active_fb_post_ids():
    """Read active FB post IDs from GitHub CSV"""
    try:
        # Try to download from GitHub
        github_content = download_file_from_github(ACTIVE_FB_POST_ID_CSV)
        
        if github_content:
            # Parse CSV content from GitHub
            lines = github_content.strip().split('\n')
            if len(lines) > 1:  # Has header + data
                reader = csv.DictReader(lines)
                data = list(reader)
                print(f"✅ Loaded {len(data)} FB post IDs from GitHub")
                return data
        
        # Return empty list if no data
        print("📝 No FB post data found in GitHub, starting with empty dataset")
        return []
        
    except Exception as e:
        print(f"❌ Error reading FB post IDs from GitHub: {e}")
        return []

def load_training_data():
    """Load training examples from response_database.csv"""
    print("🔄 Loading training data from CSV...")
    
    try:
        data = read_response_database()
        print(f"✅ Loaded {len(data)} training examples from response_database.csv")
        return data
    except Exception as e:
        print(f"❌ Error loading response_database.csv: {e}")
        return []

# Global training data
TRAINING_DATA = load_training_data()

def reload_training_data():
    """Reload training data from GitHub"""
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

# Pydantic Models - Removed append models since n8n handles writing
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

app = FastAPI()

@app.get("/")
def read_root():
    return {
        "message": "Lease End AI Assistant - Core AI System",
        "version": "16.0",
        "training_examples": len(TRAINING_DATA),
        "actions": ["respond", "react", "delete", "leave_alone"],
        "features": ["Pure AI Classification", "GitHub CSV Reading", "n8n Direct CSV Writing"],
        "approach": "FastAPI for processing, n8n for data management",
        "endpoints": {
            "/process-comment": "Initial comment processing",
            "/process-feedback": "Human feedback processing",
            "/test-feedback": "Debug endpoint for testing n8n integration",
            "/stats": "View training data statistics",
            "/generate-reply": "Backwards compatibility endpoint",
            "/response-database": "Read response database from GitHub",
            "/active-fb-posts": "Read active FB post IDs from GitHub",
            "/reload-training-data": "Refresh training data from GitHub"
        },
        "csv_files": {
            "response_database": {
                "file": "response_database.csv",
                "fields": ["comment", "action", "reply"],
                "managed_by": "n8n workflows → GitHub"
            },
            "active_fb_posts": {
                "file": "active_fb_post_id.csv", 
                "fields": ["Ad account name", "Campaign name", "Ad set name", "Ad name", "Page ID", "Post Id", "Object Story ID"],
                "managed_by": "n8n workflows → GitHub"
            }
        },
        "data_flow": "GitHub CSV ← n8n workflows | FastAPI reads from GitHub CSV",
        "philosophy": "FastAPI for AI processing, n8n for data management, GitHub as single source of truth."
    }

# Read-Only CSV Database Endpoints
@app.get("/response-database")
async def get_response_database():
    """Get all entries from response_database.csv (GitHub)"""
    try:
        data = read_response_database()
        return {
            "success": True,
            "count": len(data),
            "data": data,
            "source": "GitHub CSV"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading response database: {str(e)}")

@app.get("/active-fb-posts")
async def get_active_fb_posts():
    """Get all entries from active_fb_post_id.csv (GitHub)"""
    try:
        data = read_active_fb_post_ids()
        return {
            "success": True,
            "count": len(data),
            "data": data,
            "source": "GitHub CSV"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading active FB post IDs: {str(e)}")

@app.post("/reload-training-data")
async def reload_training_data_endpoint():
    """Manually reload training data from GitHub"""
    try:
        new_count = reload_training_data()
        return {
            "success": True,
            "message": "Training data reloaded from GitHub",
            "total_examples": new_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading training data: {str(e)}")

@app.get("/download-csv/{filename}")
async def download_csv(filename: str):
    """Download current CSV files from Railway"""
    if filename not in ["response_database.csv", "active_fb_post_id.csv"]:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    try:
        if not os.path.exists(filename):
            raise HTTPException(status_code=404, detail="File not found")
            
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Count lines for reference
        lines = content.count('\n')
        
        return {
            "filename": filename,
            "content": content,
            "line_count": lines,
            "message": f"Current {filename} with {lines} lines from Railway"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

@app.post("/sync-from-github")
async def sync_from_github_endpoint():
    """Manually trigger sync from GitHub"""
    try:
        files_synced = sync_csv_from_github()
        # Reload training data after sync
        new_count = reload_training_data()
        
        return {
            "success": True,
            "message": "Successfully synced from GitHub",
            "files_synced": files_synced,
            "new_training_count": new_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error syncing from GitHub: {str(e)}")

# Original Endpoints (unchanged)
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

@app.post("/test-feedback")
async def test_feedback(request: dict):
    """Simple endpoint to test what n8n is actually sending"""
    print("🔍 Test endpoint received data:")
    print(f"Request data: {request}")
    
    return {
        "status": "success",
        "message": "Test endpoint working",
        "received_data": request,
        "data_types": {key: str(type(value)) for key, value in request.items()},
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate-reply")
async def generate_reply(request: Request):
    """Simple reply generation for backwards compatibility"""
    data = await request.json()
    comment = data.get("comment", "")
    postId = data.get("postId", "")

    try:
        ai_classification = classify_comment_with_ai(comment, postId)
        
        if ai_classification['action'].lower() == 'reply':
            reply = generate_response(comment, ai_classification['sentiment'], ai_classification['high_intent'])
        else:
            reply = "Thank you for your comment."
        
        return {
            "postId": postId,
            "reply": reply,
            "action": ai_classification['action'].lower()
        }
        
    except Exception as e:
        return {
            "postId": postId,
            "reply": "Thank you for your comment. We appreciate your feedback.",
            "action": "ignore",
            "error": str(e)
        }

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
            "fields": ["comment", "action", "reply"],
            "field_count": 3,
            "source": "response_database.csv (GitHub)",
            "managed_by": "n8n workflows"
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
