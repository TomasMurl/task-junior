#!/bin/bash

python3 -m aiohttp.web -H 0.0.0.0 -P 8080 predict:init_func &
nginx -c /home/nginx.conf &
wait

exit $?
