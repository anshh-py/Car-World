from pydantic import BaseModel


class SupabaseAuthConfigRead(BaseModel):
	url: str | None
	anon_key: str | None


class AuthenticatedUser(BaseModel):
	id: str
	email: str | None = None
	role: str | None = None