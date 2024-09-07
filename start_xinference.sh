#!/bin/sh
#nohup  xinference-local --log-level DEBUG  --host 0.0.0.0 --port 9997 2>&1 &
nohup xinference-supervisor -H 0.0.0.0 --log-level DEBUG > supervisor.log 2>&1 &
sleep 10s
nohup xinference-worker -e "http://127.0.0.1:9997" -H 127.0.0.1 --log-level DEBUG > worker.log 2>&1 &
