#!/bin/sh
#nohup  xinference-local --log-level DEBUG  --host 0.0.0.0 --port 9997 2>&1 &
#nohup xinference-supervisor -H 192.168.1.88 --port 9997  --log-level DEBUG > supervisor.log 2>&1 &
docker compose -f docker-compose.supervisor.yaml up -d
