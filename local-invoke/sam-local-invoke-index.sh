sam local invoke IndexEmbeddingsFunction \
    --profile sandbox-us-west-2-sts \
    -e ./local-invoke/events-and-env-vars/index/event.json \
    -n ./local-invoke/events-and-env-vars/index/env-vars.json