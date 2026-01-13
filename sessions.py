"""
Session management module.
Handles user sessions and conversation history.
"""

import os
import json
import uuid
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict
from observability import get_logger, metrics

logger = get_logger(__name__)

# Sessions storage directory
SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "data", "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


@dataclass 
class Session:
    """Represents a chat session."""
    id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[Message] = field(default_factory=list)
    is_active: bool = True
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": len(self.messages),
            "is_active": self.is_active
        }
    
    def to_full_dict(self) -> dict:
        data = self.to_dict()
        data["messages"] = [asdict(m) for m in self.messages]
        return data


class SessionManager:
    """Manages user sessions and conversations."""
    
    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._load_sessions()
    
    def _load_sessions(self):
        """Load existing sessions from disk."""
        try:
            for filename in os.listdir(SESSIONS_DIR):
                if filename.endswith(".json"):
                    filepath = os.path.join(SESSIONS_DIR, filename)
                    with open(filepath, "r") as f:
                        data = json.load(f)
                        messages = [Message(**m) for m in data.get("messages", [])]
                        session = Session(
                            id=data["id"],
                            user_id=data["user_id"],
                            title=data["title"],
                            created_at=data["created_at"],
                            updated_at=data["updated_at"],
                            messages=messages,
                            is_active=data.get("is_active", True)
                        )
                        self._sessions[session.id] = session
            logger.info(f"Loaded {len(self._sessions)} sessions from disk")
        except Exception as e:
            logger.error(f"Failed to load sessions: {str(e)}")
    
    def _save_session(self, session: Session):
        """Save a session to disk."""
        try:
            filepath = os.path.join(SESSIONS_DIR, f"{session.id}.json")
            with open(filepath, "w") as f:
                json.dump(session.to_full_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save session {session.id}: {str(e)}")
    
    def create_session(self, user_id: str, title: str = None) -> Session:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat() + "Z"
        
        session = Session(
            id=session_id,
            user_id=user_id,
            title=title or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            created_at=now,
            updated_at=now,
            messages=[],
            is_active=True
        )
        
        self._sessions[session_id] = session
        self._save_session(session)
        
        metrics.active_sessions.inc()
        logger.info(f"Created new session {session_id} for user {user_id}")
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self._sessions.get(session_id)
    
    def get_user_sessions(self, user_id: str) -> list[Session]:
        """Get all sessions for a user, sorted by updated_at descending."""
        sessions = [s for s in self._sessions.values() if s.user_id == user_id]
        return sorted(sessions, key=lambda s: s.updated_at, reverse=True)
    
    def add_message(self, session_id: str, role: str, content: str) -> Optional[Message]:
        """Add a message to a session."""
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found")
            return None
        
        message = Message(role=role, content=content)
        session.messages.append(message)
        session.updated_at = datetime.utcnow().isoformat() + "Z"
        
        # Update title from first user message if still default
        if role == "user" and len(session.messages) == 1:
            session.title = content[:50] + ("..." if len(content) > 50 else "")
        
        self._save_session(session)
        return message
    
    def get_messages(self, session_id: str) -> list[Message]:
        """Get all messages for a session."""
        session = self.get_session(session_id)
        return session.messages if session else []
    
    def delete_session(self, session_id: str, user_id: str) -> bool:
        """Delete a session (only if user owns it)."""
        session = self.get_session(session_id)
        if not session or session.user_id != user_id:
            return False
        
        # Remove from memory
        del self._sessions[session_id]
        
        # Remove from disk
        filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
        
        metrics.active_sessions.dec()
        logger.info(f"Deleted session {session_id}")
        return True
    
    def rename_session(self, session_id: str, user_id: str, new_title: str) -> bool:
        """Rename a session."""
        session = self.get_session(session_id)
        if not session or session.user_id != user_id:
            return False
        
        session.title = new_title
        session.updated_at = datetime.utcnow().isoformat() + "Z"
        self._save_session(session)
        return True


# Global session manager
session_manager = SessionManager()
