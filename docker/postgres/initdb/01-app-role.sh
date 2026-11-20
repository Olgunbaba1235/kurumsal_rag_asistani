#!/bin/sh
# Veritabanı ilk oluşturulduğunda bir kez çalışır (postgres resmi imajı).
# Uygulama rolü: LOGIN, superuser DEĞİL, RLS'i atlayamaz (NOBYPASSRLS).
# Tablo yetkileri Alembic migration'ında verilir.
set -eu

psql -v ON_ERROR_STOP=1 \
     --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
     --set app_user="$APP_DB_USER" --set app_password="$APP_DB_PASSWORD" <<'EOSQL'
CREATE EXTENSION IF NOT EXISTS vector;
CREATE ROLE :"app_user" LOGIN PASSWORD :'app_password' NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS;
GRANT CONNECT ON DATABASE :"DBNAME" TO :"app_user";
EOSQL
