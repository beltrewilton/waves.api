from pydantic import BaseModel
from datetime import datetime


class Scope(BaseModel):
    id: int | None = None
    user_id: int | None = None
    name: str | None = None
    check: int | None = None

    class Config:
        orm_mode = True


class User(BaseModel):
    id: int | None = None
    username: str | None = None
    password: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    is_active: bool | None = None
    date_joined: datetime | None = None
    last_login: datetime | None = None
    pic: str | None = None
    scope: list[Scope] = []

    class Config:
        orm_mode = True
