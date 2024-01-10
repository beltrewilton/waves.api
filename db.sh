psql postgres -c "CREATE USER wilton WITH SUPERUSER PASSWORD '123456';"
psql postgres -U wilton -c 'DROP DATABASE IF EXISTS waves_db;'
psql postgres -U wilton -c 'CREATE DATABASE waves_db;'
psql waves_db -U wilton -f tables.sql
