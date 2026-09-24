import os
from dotenv import load_dotenv
load_dotenv()

for _key in ("OPENAI_API_BASE", "OPENAI_API_KEY"):
    _value = os.getenv(_key)
    if _value is not None:
        os.environ[_key] = _value
