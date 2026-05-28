#!/usr/bin/env bash
# LUCIDOTA PostgreSQL runtime contract after OS refresh.
# Prefer official distro binaries and the system cluster on 5432.

export PATH="/usr/lib/postgresql/16/bin:/usr/bin:/usr/sbin:/bin:/sbin:$PATH"
export PGHOST="/var/run/postgresql"
export PGPORT="5432"
export PGUSER="${USER}"
export PGDATABASE="lucidota_state"

export DATABASE_URL="postgresql:///lucidota_state"
export LUCIDOTA_STATE_DSN="postgresql:///lucidota_state"
export LUCIDOTA_GO_STORAGE_DSN="postgresql:///lucidota_storage"
export KORPUS_DATABASE_URL="postgresql:///lucidota_storage"
