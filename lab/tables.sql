CREATE TABLE users
(
    id            serial primary key,
    uname          VARCHAR(40) not null,
    creation_date  DATE default CURRENT_DATE
);

CREATE TABLE users_videos_t
(
    id            serial primary key,
    user_id        INTEGER,
    video_id       INTEGER
);

CREATE TABLE videos_t
(
    id            serial primary key,
    title          VARCHAR(40) not null,
    url            VARCHAR(200) not null,
    audio          BYTEA,
    transcript     JSON,
    creation_date  DATE default CURRENT_DATE
);