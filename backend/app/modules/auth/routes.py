from fastapi import APIRouter, Depends

from app.core.config import settings
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.schemas import AuthenticatedUser, SupabaseAuthConfigRead

router = APIRouter(tags=["auth"])


@router.get("/config", response_model=SupabaseAuthConfigRead)
async def read_supabase_auth_config() -> SupabaseAuthConfigRead:
	return SupabaseAuthConfigRead(
		url=settings.supabase_url,
		anon_key=settings.supabase_anon_key,
	)


@router.get("/me", response_model=AuthenticatedUser)
async def read_current_user(
	current_user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
	return current_user