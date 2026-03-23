import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

def get_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL or SUPABASE_KEY not found in .env")
    return create_client(url, key)

# ─────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────
def sign_up(email, password):
    supabase = get_supabase()
    try:
        response = supabase.auth.sign_up({
            "email":    email,
            "password": password
        })
        return {"success": True, "user": response.user}
    except Exception as e:
        return {"success": False, "error": str(e)}

def sign_in(email, password):
    supabase = get_supabase()
    try:
        response = supabase.auth.sign_in_with_password({
            "email":    email,
            "password": password
        })
        return {"success": True, "user": response.user, "session": response.session}
    except Exception as e:
        return {"success": False, "error": str(e)}

def sign_out(jwt):
    supabase = get_supabase()
    try:
        supabase.auth.sign_out()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ─────────────────────────────────────────────
# Conversations
# ─────────────────────────────────────────────
def create_conversation(user_id, title, jwt):
    supabase = get_supabase()
    supabase.postgrest.auth(jwt)
    try:
        response = supabase.table("conversations").insert({
            "user_id": user_id,
            "title":   title
        }).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        print(f"Error creating conversation: {e}")
        return None

def get_conversations(user_id, jwt):
    supabase = get_supabase()
    supabase.postgrest.auth(jwt)
    try:
        response = supabase.table("conversations") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("updated_at", desc=True) \
            .execute()
        return response.data
    except Exception as e:
        print(f"Error fetching conversations: {e}")
        return []

# ─────────────────────────────────────────────
# Messages
# ─────────────────────────────────────────────
def save_message(conversation_id, role, content, sources, jwt):
    supabase = get_supabase()
    supabase.postgrest.auth(jwt)
    try:
        response = supabase.table("messages").insert({
            "conversation_id": conversation_id,
            "role":            role,
            "content":         content,
            "sources":         sources
        }).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        print(f"Error saving message: {e}")
        return None

def get_messages(conversation_id, jwt):
    supabase = get_supabase()
    supabase.postgrest.auth(jwt)
    try:
        response = supabase.table("messages") \
            .select("*") \
            .eq("conversation_id", conversation_id) \
            .order("created_at", desc=False) \
            .execute()
        return response.data
    except Exception as e:
        print(f"Error fetching messages: {e}")
        return []

# ─────────────────────────────────────────────
# Test connection
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Supabase connection...")
    supabase = get_supabase()
    print("✅ Connected to Supabase successfully")

    print("\nTesting sign up...")
    result = sign_up("test@licensebot.com", "Test1234!")
    if result["success"]:
        print(f"✅ Sign up successful: {result['user'].email}")
    else:
        print(f"Sign up result: {result['error']}")

    print("\nTesting sign in...")
    result = sign_in("test@licensebot.com", "Test1234!")
    if result["success"]:
        print(f"✅ Sign in successful: {result['user'].email}")
    else:
        print(f"Sign in result: {result['error']}")