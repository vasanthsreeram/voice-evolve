"""Interactive login — run once to create the .session file.

  uv run python login.py

Telegram will send a code to your account; enter it when prompted.
"""
import os
from telethon import TelegramClient
from dotenv import load_dotenv

load_dotenv()

API_ID = int(os.environ["TG_API_ID"])
API_HASH = os.environ["TG_API_HASH"]
PHONE = os.environ["TG_PHONE"]
SESSION = os.environ.get("TG_SESSION", "telecall")

with TelegramClient(SESSION, API_ID, API_HASH) as client:
    me = client.loop.run_until_complete(client.get_me())
    print(f"Logged in as: {me.first_name} (id={me.id}, phone=+{me.phone})")
