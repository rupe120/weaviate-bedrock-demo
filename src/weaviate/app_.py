from aws_lambda_powertools.event_handler import APIGatewayRestResolver
from aws_lambda_powertools.utilities.typing import LambdaContext
from aws_lambda_powertools.logging import correlation_paths
from aws_lambda_powertools import Logger
from aws_lambda_powertools import Tracer
from aws_lambda_powertools import Metrics
from aws_lambda_powertools.metrics import MetricUnit
from aws_lambda_powertools.utilities import parameters
from aws_lambda_powertools.event_handler.exceptions import (
    BadRequestError,
    InternalServerError
)
from aws_lambda_powertools.event_handler.openapi.params import Query 
from aws_lambda_powertools.shared.types import Annotated  
import json
import boto3
import traceback
import os

app = APIGatewayRestResolver()
tracer = Tracer()
logger = Logger()
metrics = Metrics(namespace="Powertools")

def query_weaviate(semantic_search, weaviate_class_name, weaviate_class_properties, query_limit, weaviate_client, similarity_threshold, bedrock_client):
    global diagnostics

    # Get embeddings for the semantic_search from Bedrock
    embedding_response = bedrock_client.invoke_model(
        body=json.dumps({"inputText": semantic_search}),
        modelId='amazon.titan-embed-text-v1',
        accept='application/json',
        contentType='application/json'
    )
    embedding_response_body = json.loads(embedding_response['body'].read())

    embedding = embedding_response_body["embedding"]
    
    weaviate_query_response = (
        weaviate_client.query
        .get(weaviate_class_name, weaviate_class_properties)
        .with_additional(['certainty'])
        .with_near_vector({"vector": embedding, "certainty": similarity_threshold})
        .with_limit(query_limit)
        .do()
    )
    
    if "data" not in weaviate_query_response or "Get" not in weaviate_query_response["data"] or weaviate_class_name not in weaviate_query_response["data"]["Get"]:
        print(f"weaviate_query_response: {json.dumps(weaviate_query_response, indent=2)}")
        raise InternalServerError(f"Failed to execute Weaviate query")
    
    return weaviate_query_response["data"]["Get"][weaviate_class_name]


def query(weaviate_client, domain, search_text, max_results, similarity_threshold):
    if not search_text or "" == search_text.strip():
        return json.dumps({"result":"","urls":[]})
    
    weaviate_class_name = domain # The domain is really the class name in Weaviate

    # If the class name starts with DomainLookup_ perform a lookup against DomainLookup_[domain] to get the actual domain
    if weaviate_class_name.startswith("DomainLookup_"):
        print("=== PERFORMING DOMAIN LOOKUP ===")

        # Get the lookup class name by removing DomainLookup_ from the class name
        lookup_weaviate_class_name = weaviate_class_name[13:]

        print(f"lookup_weaviate_class_name: {lookup_weaviate_class_name}")

        # Perform a lookup against the lookup class
        lookup_weaviate_class_properties = ['col_Domain','col_Description']

        lookup_weaviate_query_data = query_weaviate(
            search_text,
            lookup_weaviate_class_name,
            lookup_weaviate_class_properties,
            1,
            weaviate_client,
            0.0)
        
        # If there are no results return an error
        if len(lookup_weaviate_query_data) == 0:
            raise BadRequestError(f"Invalid Domain: No results found in lookup domain {lookup_weaviate_class_name}")
        
        weaviate_class_name = lookup_weaviate_query_data[0]["col_Domain"]

        print(f"new weaviate_class_name: {weaviate_class_name}")

    # Lookup the weaviate class name in the schema
    weaviate_schema = weaviate_client.schema.get()
    
    # Pull the class names from the array of classes
    weaviate_class = None
    for weaviate_class_from_schema in weaviate_schema["classes"]:
        if weaviate_class_name == weaviate_class_from_schema["class"]:
            weaviate_class = weaviate_class_from_schema
    
    # Verify the provided domain is in the list of classes
    if weaviate_class is None:
        logger.info(weaviate_schema)
        raise BadRequestError("Invalid Domain")
    
    # Convert the following object to an array of names {"class":"name", "properties": [{"name":"name","dataType":["text"}],"description":"description"}]}
    # Collect all the column names that are searchable to skip cross-reference columns
    if "properties" not in weaviate_class:
        print(f"weaviate_class: {json.dumps(weaviate_class, indent=2)}")
        raise BadRequestError("Invalid Schema")
    weaviate_class_properties = [c["name"] for c in weaviate_class['properties']]
    
    # weaviate_class_properties = []
    # for property in weaviate_class["properties"]:
    #     weaviate_class_properties.append(property["name"])

    weaviate_query_data = query_weaviate(
        search_text,
        weaviate_class_name,
        weaviate_class_properties,
        max_results,
        weaviate_client,
        similarity_threshold)
        
    response = ""
    urls = []

    # Check to see if there are weaviate responses
    if len(weaviate_query_data) > 0:
        # Check to see if the first item has text and a url
        if "text" in weaviate_query_data[0] and "url" in weaviate_query_data[0]:
            for item in weaviate_query_data:
                response += f'\n**Confidence: {int(float(item["_additional"]["certainty"])*100)}%**\n{item["text"]}\n'
                if "url" in item:
                    urls.append(item["url"])
        else:
            for item in weaviate_query_data:
                response += f'\nResult:\n'
                for item_key in item:
                    item_key_processed = item_key
                    item_value = item[item_key]
                    # If item_key_processed starts with col_ remove it
                    if item_key_processed == "_additional":
                        item_key_processed = "similarity"
                        item_value = item[item_key]["certainty"]
                    elif item_key_processed.startswith("col_"):
                        item_key_processed = item_key_processed[4:]
                    response += f' {item_key_processed}: {item_value}\n'

        response += '\n'
    
    deduped_urls = [item for item in urls if item not in deduped_urls]
    
    urls_string = ""

    if len(deduped_urls) > 0:
        urls_string = '---\n***Context used:***\n'
    
        for url in deduped_urls:
            urls_string += f'* *[{url.split("/")[-1]}]({url})*\n'
        
        urls_string += "---"

    print("=== RESPONSE ===")
    print(response)

    return json.dumps({"result":response,"urls":[urls_string]})

@app.get("/weaviate/<domain>/")
@tracer.capture_method
def weaviate(domain, search_text: Annotated[str, Query()], max_results: Annotated[int, Query()], similarity_threshold: Annotated[float, Query()]):
    # adding custom metrics
    # See: https://awslabs.github.io/aws-lambda-powertools-python/latest/core/metrics/
    metrics.add_metric(name="HelloWorldInvocations", unit=MetricUnit.Count, value=1)

    if domain is None or domain == "domain":
        raise BadRequestError("Missing Domain")
    
    if search_text is None or search_text == "":
        raise BadRequestError("Missing search_text")
    
    if max_results is None or max_results == 0:
        raise BadRequestError("Missing max_results")
        
    if similarity_threshold is None or similarity_threshold == 0.0:
        raise BadRequestError("Missing similarity_threshold")
    
    weaviate_client = app.context.get("weaviate_client")
    bedrock_client = app.context.get("bedrock_client")
    
    try:
        prompt_return = query(
            weaviate_client=weaviate_client,
            domain=domain,
            search_text=search_text,
            max_results=max_results,
            similarity_threshold=similarity_threshold,
            bedrock_client=bedrock_client)

        return {
            'statusCode':200,
            'body': prompt_return,
            'headers':
                {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                }
        }
    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()
        
        return {
            'statusCode':500,
            'body': json.dumps({'error':f'ERROR: {e}'}),
            'headers':
                {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                }
        }

# Enrich logging with contextual information from Lambda
@logger.inject_lambda_context(correlation_id_path=correlation_paths.API_GATEWAY_REST)
# Adding tracer
# See: https://awslabs.github.io/aws-lambda-powertools-python/latest/core/tracer/
@tracer.capture_lambda_handler
# ensures metrics are flushed upon request completion/failure and capturing ColdStart metric
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    WEAVIATE_ENDPOINT_URL = parameters.get_secret(os.environ["WEAVIATE_ENDPOINT_URL_ARN"])
    WEAVIATE_AUTH_API_KEY = parameters.get_secret(os.environ["WEAVIATE_AUTH_API_KEY_ARN"])
    WEAVIATE_INFERENCE_ENGINE_API_KEY = parameters.get_secret(os.environ["WEAVIATE_INFERENCE_ENGINE_API_KEY_ARN"])

    # Initialize the Weaviate client defined in the global scope
    weaviate_client = weaviate.Client(
        url = WEAVIATE_ENDPOINT_URL,
        auth_client_secret=weaviate.AuthApiKey(WEAVIATE_AUTH_API_KEY),
        additional_headers = {
            "X-OpenAI-Api-Key": WEAVIATE_INFERENCE_ENGINE_API_KEY
        }
    )
    app.append_context(weaviate_client=weaviate_client)
    
    bedrock_client = boto3.client(service_name = 'bedrock-runtime')
    app.append_context(bedrock_client=bedrock_client)
    
    return app.resolve(event, context)
