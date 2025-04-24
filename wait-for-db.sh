#!/bin/sh
# wait-for-db.sh

# 1) figure out host & port: either args ($1/$2) or env-vars
if [ -n "$1" ] && [ -n "$2" ]; then
  host="$1"
  port="$2"
  shift 2
elif [ -n "$DB_HOST" ] && [ -n "$DB_PORT" ]; then
  host="$DB_HOST"
  port="$DB_PORT"
else
  echo >&2 "Error: no host/port provided (args or DB_HOST/DB_PORT)"
  exit 1
fi

# 2) the command to run once the DB is up
cmd="$@"
if [ -z "$cmd" ]; then
  echo >&2 "Error: no command provided to execute after DB is up"
  exit 1
fi

echo "⏱ Waiting for database at $host:$port …"
while ! nc -z "$host" "$port"; do
  sleep 1
done

echo "✅ $host:$port is up — executing: $cmd"
exec $cmd
