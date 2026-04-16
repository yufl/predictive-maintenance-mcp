import os

from dotenv import load_dotenv

load_dotenv()
GITHUB_TOKEN=os.getenv("GITHUB_TOKEN")
ZHIPU_SECRET_KEY=os.getenv("ZHIPU_SECRET_KEY")