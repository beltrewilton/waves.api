import sys
import os
from datetime import timedelta, datetime
from fastapi import APIRouter, Depends, HTTPException, Security, status, Header
from sqlalchemy.orm import Session
from typing import Optional
from dbms.database import get_db
import query.user_query as user_query
import schemas.users as schemas
import modules.user_models as models
from query.user_query import Token
from query.user_query import create_access_token, logout_token
from query.user_query import ACCESS_TOKEN_EXPIRE_MINUTES, ACCESS_TOKEN_EXPIRE_SECONDS
from query.user_query import validate_permissions
from dbms.Query import Query

router = APIRouter(prefix='/users', tags=['users'])

gettrace = getattr(sys, 'gettrace', None)
# is debug mode :-) ?
if gettrace():
    path = os.getcwd() + '/dbms/query.sql'
    print('Debugging :-* ')
else:
    path = os.getcwd() + '/dbms/query.sql'
    print('Run normally.')

query = Query(path)


@router.post("/add", response_model=schemas.User)
async def create(user: schemas.User, db: Session = Depends(get_db)):
    try:
        return user_query.create_user(user, db, query)
    except Exception as ex:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ex))


@router.post("/add_scope", response_model=schemas.User)
async def add_scope(user: schemas.User, db: Session = Depends(get_db)):
    try:
        return user_query.add_scopes_to_user(user, db, query)
    except Exception as ex:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ex))


@router.post("/add_stores", response_model=schemas.User)
async def add_stores(user: schemas.User, db: Session = Depends(get_db)):
    try:
        return user_query.add_stores_to_user(user, db, query)
    except Exception as ex:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ex))


@router.post("/delete_scope", response_model=schemas.User)
async def delete_scope(user: schemas.User, db: Session = Depends(get_db)):
    try:
        return user_query.delete_scopes_from_user(user, db, query)
    except Exception as ex:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ex))


@router.post("/delete_stores", response_model=schemas.User)
async def delete_stores(user: schemas.User, db: Session = Depends(get_db)):
    try:
        return user_query.delete_stores_from_user(user, db, query)
    except Exception as ex:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ex))


@router.get("/scopes", response_model=list[schemas.Scope])
async def get(
        db: Session = Depends(get_db),
        token_info: models.User = Security(dependency=validate_permissions, scopes=["member"])
):
    scopes = user_query.get_scopes(db, query)
    return scopes


@router.get("/", response_model=list[schemas.User])
async def get(
        db: Session = Depends(get_db),
        token_info: models.User = Security(dependency=validate_permissions, scopes=["member"])
):
    users = user_query.get_users(db, token_info)
    return users


@router.get('/{username}', response_model=schemas.User)
async def get_user(username: str, db: Session = Depends(get_db),
                   token_info: models.User = Security(dependency=validate_permissions, scopes=["member"])):
    user = user_query.get_user(username, db, query)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.post("/token", response_model=Token)
async def login_for_access_token(username: str, password: str,  db: Session = Depends(get_db)):
    user = user_query.authenticate_user(username, password, db, query)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    scopes = [s.name for s in user.scope]
    # stores = [s.name for s in user.stores]
    # access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token_expires = timedelta(seconds=ACCESS_TOKEN_EXPIRE_SECONDS)
    access_token = create_access_token(
        data={"sub": user.username, "scopes": scopes, "is_active": user.is_active},
        expires_delta=access_token_expires,
    )
    return {
            "username": username,
            "first_name": user.first_name,
            "access_token": access_token,
            "token_type": "bearer",
            "scopes": scopes,
            "pic": user.pic,
            'dateupdate': datetime.now().isoformat()
        }


@router.post("/logout")
async def __logout_token(Authorization: Optional[str] = Header(None),):
    try:
        token: str = Authorization.split(" ")[1]
        info = logout_token(token)

        return {
            "username": info['user']['sub'],
            "access_token": info['token_expired'],
            "token_type": "bearer",
            "scopes": info['user']['scopes'],
            "stores": info['user']['stores'],
            'dateupdate': datetime.now().isoformat()
        }
    except Exception as ex:
        raise HTTPException(status_code=401, detail=ex)



# @router.post("/token", response_model=Token)
# async def login_for_access_token(user: schemas.User,):
#     user = authenticate_user(fake_users_db, user.username, user.password)
#     if not user:
#         raise HTTPException(status_code=400, detail="Incorrect username or password")
#     access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     access_token = create_access_token(
#         data={"sub": user.username, "scopes": user.scopes},
#         expires_delta=access_token_expires,
#     )
#     return {"access_token": access_token, "token_type": "bearer", "scopes": user.scopes}