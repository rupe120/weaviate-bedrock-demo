from aws_lambda_powertools.utilities.typing import LambdaContext
from aws_lambda_powertools.logging import correlation_paths
from aws_lambda_powertools import Logger
from aws_lambda_powertools import Tracer
from aws_lambda_powertools import Metrics
from aws_lambda_powertools.metrics import MetricUnit
from aws_lambda_powertools.utilities import parameters
import json
import urllib
from index_file import index_s3_file

tracer = Tracer()
logger = Logger()
metrics = Metrics(namespace="Powertools")

# Enrich logging with contextual information from Lambda
@logger.inject_lambda_context(correlation_id_path=correlation_paths.API_GATEWAY_REST)
# Adding tracer
# See: https://awslabs.github.io/aws-lambda-powertools-python/latest/core/tracer/
@tracer.capture_lambda_handler
# ensures metrics are flushed upon request completion/failure and capturing ColdStart metric
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    # Log the incoming event
    logger.info("event:")
    logger.info(json.dumps(event))

    # Here is an example of a lambda handler event coming from an S3 trigger:
    # The code below also handles the scenario where the event is coming from an SQS Queue.
    # {
    #     "Records": [
    #         {
    #             "eventVersion": "2.1",
    #             "eventSource": "aws:s3",
    #             "awsRegion": "us-east-1",
    #             "eventTime": "2021-08-31T18:37:33.000Z",
    #             "eventName": "ObjectCreated:Put",
    #             "userIdentity": {
    #                 "principalId": "AWS:xxxxxxxxxxxx:xxxxxxxxxxxxxxxxxxxxx"
    #             },
    #             "requestParameters": {
    #                 "sourceIPAddress": "xxx.xxx.xxx.xxx"
    #             },
    #             "responseElements": {
    #                 "x-amz-request-id": "xxxxxxxxxxxxxxxx",
    #                 "x-amz-id-2": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    #             },
    #             "s3": {
    #                 "s3SchemaVersion": "1.0",
    #                 "configurationId": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    #                 "bucket": {
    #                     "name": "my-bucket",
    #                     "ownerIdentity": {
    #                         "principalId": "xxxxxxxxxxxx"
    #                     },
    #                     "arn": "arn:aws:s3:::my-bucket"
    #                 },
    #                 "object": {
    #                     "key": "my-file.txt",
    #                     "size": 123456,
    #                     "eTag": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    #                     "sequencer": "xxxxxxxxxxxxxxxxxxxxxx"
    #                 }
    #             }
    #         }
    #     ]
    # }
    
    status_messages = []
    # If there is a messageId attribute on the record process it as an SQS message
    for record in event["Records"]:
        if "messageId" in record:
            # Log the incoming SQS message
            logger.debug(record)
            if "body" not in record:
                # Log that there is nothing to process and return successfully
                logger.info("No record body to process")
                continue
            
            sqs_record_body = json.loads(record["body"])
            if "Records" not in sqs_record_body:
            # Log that there is nothing to process and return successfully
                logger.info("No record body to process")
                continue
            for s3_record in sqs_record_body["Records"]:
                s3bucket = s3_record["s3"]["bucket"]["name"]
                s3key    = urllib.parse.unquote(s3_record["s3"]["object"]["key"])
                
                status_message = index_s3_file(s3bucket, s3key)
                status_messages.append(status_message)
        else:
            # Log the incoming S3 message
            logger.debug(record)

            s3bucket = record["s3"]["bucket"]["name"]
            s3key    = record["s3"]["object"]["key"]
            
            status_message = index_s3_file(s3bucket, s3key)
            status_messages.append(status_message)
        
    return {
        "statusCode": 200,
        "body": json.dumps({
            "messages": status_messages
        })
    }
