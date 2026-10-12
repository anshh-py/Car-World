from typing import Any

import httpx
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings
from app.modules.auth.schemas import AuthenticatedUser

bearer_scheme = HTTPBearer(auto_error=False)


class InvalidSupabaseToken(Exception):
	pass


class SupabaseAuthUnavailable(Exception):
	pass


async def request_supabase_user(access_token: str) -> httpx.Response:
	if not settings.supabase_url or not settings.supabase_anon_key:
		raise SupabaseAuthUnavailable("Supabase Auth is not configured.")

	url = settings.supabase_url.rstrip("/") + "/auth/v1/user"
	try:
		async with httpx.AsyncClient(timeout=5.0) as client:
			return await client.get(
				url,
				headers={
					"apikey": settings.supabase_anon_key,
					"Authorization": f"Bearer {access_token}",
				},
			)
	except httpx.RequestError as error:
		raise SupabaseAuthUnavailable("Supabase Auth could not be reached.") from error


async def verify_supabase_access_token(access_token: str) -> AuthenticatedUser:
	response = await request_supabase_user(access_token)
	if response.status_code in {400, 401, 403}:
		raise InvalidSupabaseToken("The Supabase access token is invalid or expired.")
	if response.status_code != 200:
		raise SupabaseAuthUnavailable("Supabase Auth could not verify the access token.")

	try:
		user: Any = response.json()
	except ValueError as error:
		raise InvalidSupabaseToken("Supabase Auth returned an invalid user response.") from error

	if not isinstance(user, dict) or not isinstance(user.get("id"), str) or not user["id"]:
		raise InvalidSupabaseToken("Supabase Auth returned an invalid user response.")

	return AuthenticatedUser(
		id=user["id"],
		email=user.get("email") if isinstance(user.get("email"), str) else None,
		role=user.get("role") if isinstance(user.get("role"), str) else None,
	)


async def get_current_user(
	credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> AuthenticatedUser:
	if credentials is None:
		raise HTTPException(
			status_code=401,
			detail="Bearer token required",
			headers={"WWW-Authenticate": "Bearer"},
		)

	access_token = credentials.credentials
	parts = access_token.split(".")
	if len(parts) != 3 or any(not part for part in parts):
		raise HTTPException(
			status_code=401,
			detail="Invalid bearer token",
			headers={"WWW-Authenticate": "Bearer"},
		)

	try:
		return await verify_supabase_access_token(access_token)
	except InvalidSupabaseToken as error:
		raise HTTPException(
			status_code=401,
			detail="Invalid or expired Supabase access token",
			headers={"WWW-Authenticate": "Bearer"},
		) from error
	except SupabaseAuthUnavailable as error:
		raise HTTPException(
			status_code=503,
			detail="Supabase Auth verification is unavailable",
		) from error


async def get_optional_current_user(
	credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> AuthenticatedUser | None:
	"""Verify a supplied bearer token, while allowing requests with no token."""
	if credentials is None:
		return None
	return await get_current_user(credentials)
