from sqlalchemy.orm import Session
import modules.user_models as models
from datetime import datetime, timedelta
from typing import List
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, ValidationError
from dbms.database import get_db

from dbms.database import get_cursor
from schemas.users import User, Scope

from dbms.Query import Query


# openssl rand -hex 32
SECRET_KEY = "1f0cf1b58b6207323d9fb963b3b6ce85c1f725a474713ae8054b2969be23c0d0"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60*5
ACCESS_TOKEN_EXPIRE_SECONDS = 60*60*10


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
            "me": "Read information about the current user.",
            "items": "Read items.",
            "sales": "Can see sales and products.",
    },
)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


class Token(BaseModel):
    username: str | None = None
    first_name: str | None = None
    access_token: str
    token_type: str
    scopes: List[str] = []
    pic: str | None = None
    dateupdate: str | None = None


class TokenData(BaseModel):
    username: str | None = None
    scopes: List[str] = []


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def validate_user(
    security_scopes: SecurityScopes, token: str = Depends(oauth2_scheme),
):
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"

    credentials_expired = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")

        if username is None:
            raise credentials_expired

        token_scopes = payload.get("scopes", [])
        user_active  = payload.get("is_active")
        token_data = TokenData(scopes=token_scopes, username=username)
    except (JWTError, ValidationError) as ex:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(ex),
            headers={"WWW-Authenticate": authenticate_value},
        )

    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not enough permissions",
                headers={"WWW-Authenticate": authenticate_value},
            )

    return {
        'username': username,
        'user_active': user_active,
        'token_scopes': token_scopes,
        'dateupdate': datetime.now().isoformat()
    }


async def validate_permissions(token_info: models.User = Security(validate_user)):
    if not token_info['user_active']:
        raise HTTPException(status_code=400, detail="Inactive user")
    return token_info


def create_user(user: User, db: Session, query: Query):
    user.password = get_password_hash(user.password)
    sql_raw_insert_user = query.INSERT_USER
    sql_raw_insert_scopes = query.INSERT_SCOPES
    sql_raw_insert_u_stores = query.INSERT_USER_STORES
    cur = get_cursor(db)
    data = (user.username, user.password, user.first_name, user.last_name, user.pic)
    cur.execute(sql_raw_insert_user, data)
    user_id = cur.fetchone()['id']
    user.id = user_id
    cur.connection.commit()


    for s in user.scope:
        data = (s.name, user_id)
        cur.execute(sql_raw_insert_scopes, data)
        cur.connection.commit()

    # for t in user.stores:
    #     store_id: int = selected_store(t.name, db, query)
    #     data = (user_id, store_id)
    #     cur.execute(sql_raw_insert_u_stores, data)
    #     cur.connection.commit()

    return user


def add_scopes_to_user(user: User, db: Session, query: Query):
    sql_raw_insert_scopes = query.INSERT_SCOPES
    cur = get_cursor(db)
    for s in user.scope:
        data = (s.name, user.id)
        cur.execute(sql_raw_insert_scopes, data)
        cur.connection.commit()

    return True


# def add_stores_to_user(user: User, db: Session, query: Query):
#     sql_raw_insert_u_stores = query.INSERT_USER_STORES
#     cur = get_cursor(db)
#     for t in user.stores:
#         store_id: int = selected_store(t.name, db, query)
#         data = (user.id, store_id)
#         cur.execute(sql_raw_insert_u_stores, data)
#         cur.connection.commit()
#
#     return True


def delete_scopes_from_user(user: User, db: Session, query: Query):
    sql_raw_delete_scopes = query.DELETE_SCOPES
    cur = get_cursor(db)
    for s in user.scope:
        data = (s.name, user.id)
        cur.execute(sql_raw_delete_scopes, data)
        cur.connection.commit()

    return True


# def delete_stores_from_user(user: User, db: Session, query: Query):
#     sql_raw_delete_u_stores = query.DELETE_USER_STORES
#     cur = get_cursor(db)
#     for t in user.stores:
#         store_id: int = selected_store(t.name, db, query)
#         data = (user.id, store_id)
#         cur.execute(sql_raw_delete_u_stores, data)
#         cur.connection.commit()
#
#     return True


def get_scopes(db: Session, query: Query):
    sql_raw_scopes = query.SELECT_SCOPES

    cur = get_cursor(db)
    cur.execute(sql_raw_scopes)
    resp = cur.fetchall()

    listscope = []
    for r in resp:
        scope = Scope()
        scope.id = r['id']
        scope.name = r['name']
        scope.check = 0
        listscope.append(scope)

    return listscope


def get_user(username: str, db: Session, query: Query):
    sql_raw_user = query.SELECT_USER
    sql_raw_user_store = query.SELECT_USER_STORE
    sql_raw_user_scope = query.SELECT_USER_SCOPES

    data = (username,)
    cur = get_cursor(db)
    cur.execute(sql_raw_user, data)
    resp = cur.fetchall()
    user = User()
    for r in resp:
        user.id = r['id']
        user.username = r['username']
        user.password = r['password']
        user.first_name = r['first_name']
        user.last_name = r['last_name']
        user.is_active = r['is_active']
        user.date_joined = r['date_joined']
        user.last_login = r['last_login']
        user.pic = r['pic']

        # data = (user.id,)
        # cur.execute(sql_raw_user_store, data)
        # stor = cur.fetchall()
        # storlist = []
        # for s in stor:
        #     store = Store()
        #     store.id = s['id']
        #     store.name = s['name']
        #     store.company_id = s['company_id']
        #     store.slogan = s['slogan']
        #     store.logo = s['logo']
        #     store.address = s['address']
        #     storlist.append(store)

        data = (user.id,)
        cur.execute(sql_raw_user_scope, data)
        sco = cur.fetchall()
        scolist = []
        for s in sco:
            scope = Scope()
            scope.name = s['scope']
            scolist.append(scope)

        # user.stores = storlist
        user.scope = scolist

    return user


def get_users(db: Session, token_info: dict):
    # res = db.query(models.User, models.Scopes).filter(models.User.id == models.Scopes.user_id).all()
    # if 'sales.filter.user' in token_info['token_scopes']:
    #     res = db.query(models.User).all()
    # else:
    #     res = db.query(models.User).filter(models.User.username == token_info['username']).all()
    res = db.query(models.User).all()
    return res


def authenticate_user(username: str, password: str, db: Session, query: Query):
    user = get_user(username, db, query)
    if not user:
        return False
    if not verify_password(password, user.password):
        return False
    return user


def logout_token(token: str):
    expire = datetime.utcnow() + timedelta(seconds=1)
    token_decoded = None

    try:
        token_decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except Exception as ex:
        token_decoded = {"sub": "dummy", "scopes": [], "stores": [], "is_active": "1", "exp": 0}

    token_decoded.update({"exp": expire})
    token_expired = jwt.encode(token_decoded, key=SECRET_KEY, algorithm=ALGORITHM)
    print(token_expired)
    return {'user': token_decoded, 'token_expired': token_expired}
