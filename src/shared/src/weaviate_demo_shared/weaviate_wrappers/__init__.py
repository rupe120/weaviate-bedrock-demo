import json


OBJECT_COLUMN_TYPES = [
    {
        "description": f"Property text",
        "name": "text",
        "dataType": ["text"]
    },
    {
        "description": f"Property skill list for the object from the LLM",
        "name": "skills",
        "dataType": ["text[]"]
    },
    {
        "description": "Concordant skills from the global skills list",
        "name": "concordantSkills",
        "dataType": ["text[]"]
    },
    {
        "description": f"Property S3Bucket",
        "name": "s3Bucket",
        "dataType": ["text"]
    },
    {
        "description": f"Property S3Key",
        "name": "s3Key",
        "dataType": ["text"]
    },
    {
        "description": f"Property for cross-reference of concordant globalSkills",
        "name": "concordantGlobalSkills",
        "dataType": ["Skill"]
    }
]

def get_query_result_count(     
    weaviate_class_name, 
    weaviate_client, 
    where_filter=None,):
    query = (weaviate_client.query
        .aggregate(weaviate_class_name)
    )

    if where_filter:
        query = query.with_where(where_filter)
    
    count_query_result = (query
            .with_meta_count()
            .do())
    if ("data" not in count_query_result or 
        "Aggregate" not in count_query_result["data"] or 
        weaviate_class_name not in count_query_result["data"]["Aggregate"] or
        len(count_query_result["data"]["Aggregate"][weaviate_class_name]) == 0 or
        "meta" not in count_query_result["data"]["Aggregate"][weaviate_class_name][0] or
        "count" not in count_query_result["data"]["Aggregate"][weaviate_class_name][0]["meta"]):
        print(f"Invalid Weaviate Response: {json.dumps(count_query_result, indent=2)}")
        raise Exception(f"Could not get result count for {weaviate_class_name}")
    
    target_result_count = count_query_result["data"]["Aggregate"][weaviate_class_name][0]["meta"]["count"]
    return target_result_count

def weaviate_query(
    weaviate_class_name, 
    weaviate_class_properties, 
    weaviate_client, 
    bedrock_client, 
    query_limit=None, 
    additional_columns=None,
    where_filter=None,
    offset=0,
    target_result_count=None,
    debug=False):
    global diagnostics
    
    query = (weaviate_client.query
        .get(weaviate_class_name, weaviate_class_properties)
        .with_offset(offset)
    )

    if additional_columns:
        query = query.with_additional(additional_columns)
    
    if where_filter:
        query = query.with_where(where_filter)
    
    # Determine the target result count if the limit is not set.
    # This is used to determine if a partial result set is returned. If so we will query again.
    if not target_result_count:
        if query_limit:
            print(f"Setting target_result_count to query limit of {query_limit}")
            target_result_count = query_limit
        else:
            if debug:
                print(f"Setting target_result_count to full result count")
            
            target_result_count = get_query_result_count(weaviate_class_name, weaviate_client, where_filter=where_filter)
            
            print(f"Setting target_result_count to full result count of {target_result_count}")
    
    if query_limit:
        query = query.with_limit(query_limit)
        
    weaviate_query_response = query.do()
    
    if "data" not in weaviate_query_response or "Get" not in weaviate_query_response["data"] or weaviate_class_name not in weaviate_query_response["data"]["Get"]:
        return []
    
    result_count = len(weaviate_query_response["data"]["Get"][weaviate_class_name])
    
    result_list = weaviate_query_response["data"]["Get"][weaviate_class_name]
    offset=offset + result_count
    if result_count > 0 and offset < target_result_count:
        print(f"Retrieving the next set of results. Retrieved {offset} of {target_result_count}.")
        next_result_list = weaviate_query(
            weaviate_class_name, 
            weaviate_class_properties,
            weaviate_client, 
            bedrock_client, 
            query_limit,
            additional_columns=additional_columns,
            where_filter=where_filter,
            offset=offset,
            target_result_count=target_result_count,
            debug=debug
        )
        
        result_list.extend(next_result_list)
        
    return result_list
    
def weaviate_semantic_query(
    semantic_search, 
    weaviate_class_name, 
    weaviate_class_properties, 
    weaviate_client, 
    similarity_threshold, 
    bedrock_client, 
    query_limit=None, 
    additional_columns=None,
    where_filter=None,
    offset=0,
    target_result_count=None,
    debug=False):
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
    
    query = (weaviate_client.query        
        .get(weaviate_class_name, weaviate_class_properties)
        .with_near_vector({"vector": embedding, "certainty": similarity_threshold})
        .with_additional(['certainty'])
        .with_offset(offset)
    )
    
    if additional_columns:
        query = query.with_additional(additional_columns)
    
    if where_filter:
        query = query.with_where(where_filter)
    
    # Determine the target result count if the limit is not set.
    # This is used to determine if a partial result set is returned. If so we will query again.
    if not target_result_count and query_limit:
        print(f"Setting target_result_count to query limit of {query_limit}")
        target_result_count = query_limit
    if query_limit:
        query = query.with_limit(query_limit)
    
    weaviate_query_response = query.do()
    
    if "data" not in weaviate_query_response or "Get" not in weaviate_query_response["data"] or weaviate_class_name not in weaviate_query_response["data"]["Get"]:
        return []
    
    result_count = len(weaviate_query_response["data"]["Get"][weaviate_class_name])
    
    result_list = weaviate_query_response["data"]["Get"][weaviate_class_name]
    offset=offset + result_count,
    if result_count > 0 and target_result_count and offset < target_result_count:
        print(f"Retrieving the next set of results. Retrieved {offset} of {target_result_count}.")
        next_result_list = weaviate_semantic_query(
            semantic_search, 
            weaviate_class_name, 
            weaviate_class_properties, 
            weaviate_client, 
            similarity_threshold, 
            bedrock_client, 
            query_limit, 
            additional_columns=additional_columns,
            where_filter=where_filter,
            offset=offset,
            target_result_count=target_result_count,
            debug=debug
        )
        
        result_list.extend(next_result_list)
        
    return result_list

