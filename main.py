from fastapi import FastAPI, Request
import os
import csv
from datetime import datetime, timezone
from anthropic import Anthropic
import dspy
from pydantic import BaseModel

# Load API key from environment variable (set in Railway dashboard)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")

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

# Load training data from CSV
def load_training_data():
    """Load training examples from CSV file"""
    training_examples = []
    
    try:
        with open('training_data.csv', 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                training_examples.append({
                    'comment': row['comment'],
                    'category': row['category'],
                    'urgency': row['urgency'],
                    'requires_response': row['requires_response'].lower() == 'true',
                    'reply': row['reply']
                })
        print(f"✅ Loaded {len(training_examples)} training examples")
    except FileNotFoundError:
        print("⚠️ No training_data.csv found, using fallback classification")
        training_examples = []
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        training_examples = []
    
    return training_examples

# Global training data
TRAINING_DATA = load_training_data()

# Customer Avatar Detection
def identify_customer_avatar(comment):
    """Identify customer avatar based on comment content"""
    comment_lower = comment.lower()
    
    # Anna (Busy Professional) - time/efficiency focused
    anna_keywords = ["busy", "quick", "time", "efficient", "fast", "hassle", "simple", "easy"]
    
    # Mike (Cost-Conscious) - price/value focused  
    mike_keywords = ["cost", "price", "rate", "save", "money", "cheap", "expensive", "fee", "payment"]
    
    # Sarah (Family-Oriented) - stability/simplicity focused
    sarah_keywords = ["family", "kids", "reliable", "stable", "safe", "secure", "trust", "worry"]
    
    anna_score = sum(1 for word in anna_keywords if word in comment_lower)
    mike_score = sum(1 for word in mike_keywords if word in comment_lower)
    sarah_score = sum(1 for word in sarah_keywords if word in comment_lower)
    
    if mike_score > anna_score and mike_score > sarah_score:
        return "mike_cost_conscious"
    elif sarah_score > anna_score and sarah_score > mike_score:
        return "sarah_family"
    elif anna_score > 0:
        return "anna_professional"
    else:
        return "general"

# Enhanced classification using training data + brand context
def smart_classify_comment(comment, comment_age="recent"):
    """Classify comment using training data patterns and brand context"""
    comment_lower = comment.lower()
    
    # First, check training data for similar patterns
    best_match = None
    max_similarity = 0
    
    for example in TRAINING_DATA:
        example_lower = example['comment'].lower()
        # Simple similarity check (could be enhanced with embeddings)
        common_words = set(comment_lower.split()) & set(example_lower.split())
        if len(common_words) >= 2:  # At least 2 words in common
            similarity = len(common_words)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = example
    
    # If we found a good match, use its classification
    if best_match and max_similarity >= 2:
        return {
            'category': best_match['category'],
            'urgency': best_match['urgency'],
            'requires_response': best_match['requires_response']
        }
    
    # Fallback: Brand-aligned classification logic
    if any(word in comment_lower for word in ['rate', 'cost', 'price', 'expensive', 'fee', 'payment', 'money']):
        return {'category': 'price_objection', 'urgency': 'medium', 'requires_response': True}
    elif any(word in comment_lower for word in ['dealership', 'bank', 'credit union', 'better', 'vs', 'compare']):
        return {'category': 'competitor_comparison', 'urgency': 'medium', 'requires_response': True}
    elif any(word in comment_lower for word in ['process', 'how', 'steps', 'time', 'quick', 'easy', 'hassle']):
        return {'category': 'process_concern', 'urgency': 'medium', 'requires_response': True}
    elif any(word in comment_lower for word in ['terrible', 'awful', 'bad', 'worst', 'hate', 'scam']):
        return {'category': 'complaint', 'urgency': 'high', 'requires_response': True}
    elif '?' in comment or any(word in comment_lower for word in ['can i', 'do you', 'how do', 'what']):
        return {'category': 'primary_inquiry', 'urgency': 'medium', 'requires_response': True}
    else:
        return {'category': 'general', 'urgency': 'low', 'requires_response': False}

# Brand-aligned response generation
def generate_brand_aligned_response(comment, avatar, category, urgency):
    """Generate response using training data + brand alignment"""
    comment_lower = comment.lower()
    
    # First, try to find matching response from training data
    best_match = None
    max_similarity = 0
    
    for example in TRAINING_DATA:
        if example['category'] == category:
            example_lower = example['comment'].lower()
            common_words = set(comment_lower.split()) & set(example_lower.split())
            similarity = len(common_words)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = example
    
    # If we found a good match, use it as base
    if best_match and max_similarity >= 1:
        base_response = best_match['reply']
        
        # Enhance with avatar-specific messaging
        if avatar == "anna_professional" and category == "process_concern":
            return f"{base_response} Our 100% online process saves you time - no dealership visits required!"
        elif avatar == "mike_cost_conscious" and category == "price_objection":
            return f"{base_response} No hidden fees, guaranteed - you'll know exactly what you're paying upfront."
        elif avatar == "sarah_family" and any(word in comment_lower for word in ["worry", "decision", "family"]):
            return f"{base_response} We're here to guide you through every step with no pressure."
        else:
            return base_response
    
    # Fallback: Brand-aligned response templates
    templates = {
        "price_objection": {
            "mike_cost_conscious": "We work with multiple lenders to get you the most competitive rates available. No hidden fees, guaranteed - you'll know exactly what you're paying upfront.",
            "general": "Our transparent pricing means no surprises. We shop multiple lenders to find you the best rate, often beating dealership offers."
        },
        "process_concern": {
            "anna_professional": "Our 100% online process takes just minutes to complete. No dealership visits, no pressure, no wasted time - we handle everything digitally.",
            "sarah_family": "We've made the process simple and stress-free. Our team guides you through every step, so you can make the right decision for your family with confidence.",
            "general": "Skip the dealership hassle! Our online process is quick, transparent, and completely pressure-free."
        },
        "competitor_comparison": {
            "general": "Unlike dealerships with hidden fees and pressure tactics, we offer transparent pricing and a no-pressure, 100% online experience. Many customers save both time and money with us."
        },
        "primary_inquiry": {
            "mike_cost_conscious": "Great question! We specialize in competitive lease buyout financing with transparent pricing. Let us shop multiple lenders to find you the best rate - no hidden fees guaranteed.",
            "anna_professional": "We can help! Our streamlined online process gets you pre-approved quickly, often with better rates than dealerships. No time wasted, no pressure.",
            "sarah_family": "We're here to help make this decision easy for you. Our transparent process and dedicated support team ensure you get the best deal without any stress or pressure.",
            "general": "We'd love to help! Our transparent, no-pressure process often gets customers better rates than dealerships, all handled online for your convenience."
        },
        "complaint": {
            "general": "We understand your concern, and transparency is core to everything we do. We're here to make the lease buyout process stress-free and honest - no hidden fees, no pressure, guaranteed."
        }
    }
    
    # Get appropriate template
    category_templates = templates.get(category, templates["primary_inquiry"])
    response = category_templates.get(avatar, category_templates.get("general", "Thank you for your interest! We're here to help make your lease buyout transparent and hassle-free."))
    
    # Add urgency elements
    if urgency == "high":
        urgency_phrases = [
            " Don't wait - lease-end dates approach quickly!",
            " Lock in your rate before it changes!",
            " Get your free quote today!"
        ]
        response += urgency_phrases[0]
    
    return response

# Request/Response models
class CommentRequest(BaseModel):
    comment: str
    postId: str
    created_time: str = ""
    memory_context: str = ""

class ProcessedComment(BaseModel):
    postId: str
    original_comment: str
    customer_avatar: str
    category: str
    urgency: str
    requires_response: bool
    reply: str
    should_post: bool
    needs_review: bool
    confidence_score: float

app = FastAPI()

@app.get("/")
def read_root():
    return {
        "message": "Lease End AI Assistant - Production Ready",
        "version": "2.0",
        "training_examples": len(TRAINING_DATA),
        "features": ["Customer Avatar Detection", "Brand-Aligned Responses", "Training Data Integration"]
    }

@app.post("/process-comment", response_model=ProcessedComment)
async def process_comment(request: CommentRequest):
    """Enhanced comment processing with brand alignment and training data"""
    try:
        # Calculate comment age
        comment_age = "recent"
        if request.created_time:
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
        
        # Identify customer avatar
        avatar = identify_customer_avatar(request.comment)
        
        # Use smart classification (training data + brand context)
        classification = smart_classify_comment(request.comment, comment_age)
        
        category = classification['category']
        urgency = classification['urgency']
        requires_response = classification['requires_response']
        
        # Generate brand-aligned response if needed
        reply_text = ""
        confidence_score = 0.8  # Base confidence
        
        if requires_response:
            reply_text = generate_brand_aligned_response(request.comment, avatar, category, urgency)
            
            # Adjust confidence based on training data match
            comment_lower = request.comment.lower()
            for example in TRAINING_DATA:
                example_lower = example['comment'].lower()
                common_words = set(comment_lower.split()) & set(example_lower.split())
                if len(common_words) >= 2:
                    confidence_score = min(0.95, confidence_score + (len(common_words) * 0.05))
                    break
        
        # Enhanced decision logic with brand context
        should_post = (
            requires_response and 
            urgency != "high" and
            comment_age != "days_old" and
            category not in ["complaint"] and
            confidence_score > 0.75
        )
        
        needs_review = (
            requires_response and 
            (urgency == "high" or comment_age == "days_old" or category == "complaint" or confidence_score <= 0.75)
        )
        
        return ProcessedComment(
            postId=request.postId,
            original_comment=request.comment,
            customer_avatar=avatar,
            category=category,
            urgency=urgency,
            requires_response=requires_response,
            reply=reply_text,
            should_post=should_post,
            needs_review=needs_review,
            confidence_score=confidence_score
        )
        
    except Exception as e:
        print(f"Error processing comment: {e}")
        return ProcessedComment(
            postId=request.postId,
            original_comment=request.comment,
            customer_avatar="general",
            category="error",
            urgency="low",
            requires_response=False,
            reply="Thank you for your comment. We appreciate your feedback and will get back to you soon.",
            should_post=False,
            needs_review=True,
            confidence_score=0.0
        )

# Keep backwards compatibility
@app.post("/generate-reply")
async def generate_reply(request: Request):
    """Simple reply generation for backwards compatibility"""
    data = await request.json()
    comment = data.get("comment", "")
    postId = data.get("postId", "")

    try:
        # Use enhanced processing
        avatar = identify_customer_avatar(comment)
        classification = smart_classify_comment(comment)
        
        if classification['requires_response']:
            reply = generate_brand_aligned_response(comment, avatar, classification['category'], classification['urgency'])
        else:
            reply = "Thank you for your comment."
        
        return {
            "postId": postId,
            "reply": reply
        }
        
    except Exception as e:
        return {
            "postId": postId,
            "reply": "Thank you for your comment. We appreciate your feedback.",
            "error": str(e)
        }

# Debug endpoint for testing
@app.post("/debug-comment")
async def debug_comment(request: CommentRequest):
    """Debug endpoint to see all classification details"""
    avatar = identify_customer_avatar(request.comment)
    classification = smart_classify_comment(request.comment)
    
    return {
        "comment": request.comment,
        "customer_avatar": avatar,
        "classification": classification,
        "brand_context": "Customer avatar and brand-aligned classification applied"
    }

# Railway-specific server startup
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
