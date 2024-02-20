import json

# Common code for the Bedrock API

def generate_embeddings(content, bedrock_client):
    # Get embeddings for the semantic_search from Bedrock
    
    embedding_response = bedrock_client.invoke_model(
            body=json.dumps({"inputText": content}),
            modelId='amazon.titan-embed-text-v1',
            accept='application/json',
            contentType='application/json'
    )

    if embedding_response["ResponseMetadata"]["HTTPStatusCode"] != 200:
        raise f"Failed to execute Bedrock model: {embedding_response['ResponseMetadata']['HTTPStatusCode']} {embedding_response['body'].read()}"

    embedding_response_body = json.loads(embedding_response['body'].read())

    return embedding_response_body["embedding"]