# --- Centralized Database Configuration ---
from lab import WORKSPACE_DIR


db = None  # This will hold the aiosqlite connection
DATABASE_FILE_NAME = f"{WORKSPACE_DIR}/llmlab.sqlite3"
DATABASE_URL = f"sqlite+aiosqlite:///{DATABASE_FILE_NAME}"
