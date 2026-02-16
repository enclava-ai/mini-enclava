#!/usr/bin/env sh

set -eu

# Run migrations before starting the app.
/usr/local/bin/migrate.sh

# Launch application server.
exec uvicorn app.main:app ${UVICORN_ARGS}
