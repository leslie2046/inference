#!/bin/sh
#nohup xinference-worker -e "http://192.168.1.88:9997" -H 127.0.0.1 --log-level DEBUG > sensevoice.log 2>&1 &
docker compose -f docker-compose.worker.yaml up -d
