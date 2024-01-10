DROP TABLE users;
DROP TABLE scopes;
DROP TABLE scope_list;

CREATE TABLE users (
  id            serial primary key,
  username varchar(20) DEFAULT NULL,
  password varchar(200) DEFAULT NULL,
  first_name varchar(20) DEFAULT NULL,
  last_name varchar(30) DEFAULT NULL,
  is_active int DEFAULT 1,
  date_joined TIMESTAMP DEFAULT CURRENT_DATE,
  last_login TIMESTAMP DEFAULT NULL,
  pic text,
  UNIQUE (username)
);

CREATE TABLE scopes (
  id            serial primary key,
  name varchar(50) DEFAULT NULL,
  user_id int DEFAULT NULL
);

CREATE TABLE scope_list (
   id           serial primary key,
  name varchar(200) DEFAULT NULL
);

INSERT INTO scopes (name, user_id) VALUES('member', 1);
INSERT INTO scope_list (name) VALUES('member');