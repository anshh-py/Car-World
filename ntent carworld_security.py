[1mdiff --git a/backend/app/modules/chat/routes.py b/backend/app/modules/chat/routes.py[m
[1mindex 41edc0d..687f888 100644[m
[1m--- a/backend/app/modules/chat/routes.py[m
[1m+++ b/backend/app/modules/chat/routes.py[m
[36m@@ -3,7 +3,7 @@[m [mfrom __future__ import annotations[m
 import asyncio[m
 import logging[m
 import math[m
[31m-import re[m
[32m+[m[32mimport sys[m
 import time[m
 from collections import defaultdict, deque[m
 [m
[36m@@ -11,12 +11,17 @@[m [mfrom fastapi import APIRouter, Depends, HTTPException, Request, status[m
 from groq import BadRequestError[m
 from starlette.concurrency import run_in_threadpool[m
 [m
[31m-from app.core.config import settings[m
[32m+[m[32mfrom app.core.config import PROJECT_ROOT, settings[m
 from app.modules.auth.dependencies import get_optional_current_user[m
 from app.modules.auth.schemas import AuthenticatedUser[m
 from app.modules.chat.agent import get_agent[m
 from app.modules.chat.schemas import ChatRequest, ChatResponse[m
 [m
[32m+[m[32mproject_root_string = str(PROJECT_ROOT)[m
[32m+[m[32mif project_root_string not in sys.path:[m
[32m+[m[32m    sys.path.insert(0, project_root_string)[m
[32m+[m[32mfrom carworld_security import redact_sensitive_text, validate_model_output[m
[32m+[m
 logger = logging.getLogger(__name__)[m
 router = APIRouter(tags=["chat"])[m
 _request_times: dict[str, deque[float]] = defaultdict(deque)[m
[36m@@ -70,10 +75,7 @@[m [mdef _safe_provider_error_fields(error: BadRequestError) -> tuple[str, str, str,[m
     def safe_text(value: object, limit: int = 240) -> str:[m
         if not isinstance(value, str):[m
             return ""[m
[31m-        value = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~-]+", "Bearer [REDACTED]", value)[m
[31m-        value = re.sub(r"\bgsk_[A-Za-z0-9_-]+\b", "[REDACTED]", value)[m
[31m-        value = re.sub(r"\btvly-[A-Za-z0-9_-]+\b", "[REDACTED]", value)[m
[31m-        return value[:limit][m
[32m+[m[32m        return redact_sensitive_text(value)[:limit][m
 [m
     return ([m
         safe_text(details.get("type")),[m
[36m@@ -99,9 +101,7 @@[m [masync def chat([m
     try:[m
         agent = get_agent()[m
         replies = await run_in_threadpool(lambda: list(agent.run(conversation)))[m
[31m-        reply = replies[-1].strip() if replies else ""[m
[31m-        if not reply:[m
[31m-            raise RuntimeError("The agent returned an empty response.")[m
[32m+[m[32m        reply = validate_model_output(replies[-1] if replies else None)[m
         return ChatResponse(reply=reply)[m
     except BadRequestError as error:[m
         provider_type, provider_code, provider_param, message, generation_reason = ([m
