# facebook_auth.py
import os
import requests
from datetime import datetime

class FacebookTokenManager:
    def __init__(self):
        self.app_id = os.getenv("FACEBOOK_APP_ID")
        self.app_secret = os.getenv("FACEBOOK_APP_SECRET") 
        self.page_id = os.getenv("FACEBOOK_PAGE_ID")
        
    def refresh_page_token(self):
        """Refresh the Facebook page token automatically"""
        try:
            current_token = os.getenv("FACEBOOK_PAGE_TOKEN")
            
            # Step 1: Exchange current token for long-lived user token
            user_token_url = f"https://graph.facebook.com/oauth/access_token"
            user_params = {
                "grant_type": "fb_exchange_token",
                "client_id": self.app_id,
                "client_secret": self.app_secret,
                "fb_exchange_token": current_token
            }
            
            user_response = requests.get(user_token_url, params=user_params)
            user_data = user_response.json()
            
            if "access_token" not in user_data:
                return {"error": "Failed to get user token", "details": user_data}
            
            long_lived_user_token = user_data["access_token"]
            
            # Step 2: Get new page token using long-lived user token
            page_token_url = f"https://graph.facebook.com/{self.page_id}"
            page_params = {
                "fields": "access_token",
                "access_token": long_lived_user_token
            }
            
            page_response = requests.get(page_token_url, params=page_params)
            page_data = page_response.json()
            
            if "access_token" not in page_data:
                return {"error": "Failed to get page token", "details": page_data}
            
            new_page_token = page_data["access_token"]
            
            return {
                "success": True,
                "new_token": new_page_token,
                "old_token": current_token[:20] + "...",
                "refreshed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def check_token_status(self):
        """Check when Facebook token expires"""
        try:
            current_token = os.getenv("FACEBOOK_PAGE_TOKEN")
            
            # Check token info
            debug_url = "https://graph.facebook.com/debug_token"
            params = {
                "input_token": current_token,
                "access_token": f"{self.app_id}|{self.app_secret}"
            }
            
            response = requests.get(debug_url, params=params)
            data = response.json()
            
            if "data" in data:
                expires_at = data["data"].get("expires_at", 0)
                is_valid = data["data"].get("is_valid", False)
                
                if expires_at == 0:
                    expiry_info = "Never expires"
                    days_until_expiry = "N/A"
                else:
                    expiry_date = datetime.fromtimestamp(expires_at)
                    days_until_expiry = (expiry_date - datetime.now()).days
                    expiry_info = f"Expires on {expiry_date.strftime('%Y-%m-%d %H:%M:%S')}"
                
                return {
                    "is_valid": is_valid,
                    "expiry_info": expiry_info,
                    "days_until_expiry": days_until_expiry,
                    "needs_refresh": days_until_expiry < 7 if isinstance(days_until_expiry, int) else False
                }
            else:
                return {"error": "Could not check token status", "details": data}
                
        except Exception as e:
            return {"error": str(e)}
