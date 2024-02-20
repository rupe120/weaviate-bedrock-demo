from http.client import INTERNAL_SERVER_ERROR
from aws_lambda_powertools.utilities.typing import LambdaContext
from aws_lambda_powertools.logging import correlation_paths
from aws_lambda_powertools import Logger
from aws_lambda_powertools import Tracer
from aws_lambda_powertools import Metrics
from aws_lambda_powertools.metrics import MetricUnit
from aws_lambda_powertools.utilities import parameters
import weaviate
import json
import os
import boto3
import pandas as pd
import urllib
import urllib3
import re
from weaviate.util import generate_uuid5  # Generate a deterministic ID
import anthropic
from langchain.text_splitter import TokenTextSplitter
from pypdf import PdfReader
import weaviate_demo_shared.bedrock_wrappers as bedrock_api_wrappers
import weaviate_demo_shared.weaviate_wrappers as weaviate_wrappers
import weaviate_demo_shared.skill_processing as skill_processing

http = urllib3.PoolManager()
tracer = Tracer()
logger = Logger()
metrics = Metrics(namespace="Powertools")

s3_client = None
bedrock_client = None
weaviate_client = None

global_skill_collection = "Skill"
global_skills = {} # I.e. {"name": {"id": <guid>, "status": "<current status>", "isActive": True/False}}
global_inactive_skills = {}

skill_column_names = [col["name"] for col in weaviate_wrappers.SKILL_COLUMN_TYPES]

# Columns from the collected data elements to save into the object for the data of Weaviate collection records.
# Excluding the globalSkills cross-reference because it is added outside of the record data object.
object_column_names = [col["name"] for col in weaviate_wrappers.OBJECT_COLUMN_TYPES if col["name"] not in weaviate_wrappers.CROSS_REFERENCE_COLUMN_NAMES]
object_column_names_all = [col["name"] for col in weaviate_wrappers.OBJECT_COLUMN_TYPES]


WEAVIATE_ENDPOINT_URL = parameters.get_secret(os.environ["WEAVIATE_ENDPOINT_URL_ARN"])
WEAVIATE_AUTH_API_KEY = parameters.get_secret(os.environ["WEAVIATE_AUTH_API_KEY_ARN"])
WEAVIATE_INFERENCE_ENGINE_API_KEY = parameters.get_secret(os.environ["WEAVIATE_INFERENCE_ENGINE_API_KEY_ARN"])

INFERENCE_ENGINE_API_PROVIDER = os.environ["INFERENCE_ENGINE_API_PROVIDER"]

CERTAINTY_THRESHOLD_FOR_SKILL_MATCHING = float(os.environ.get("CERTAINTY_THRESHOLD_FOR_SKILL_MATCHING", "0.9"))
CERTAINTY_THRESHOLD_FOR_SKILL_ADDING = float(os.environ.get("CERTAINTY_THRESHOLD_FOR_SKILL_ADDING", "0.9"))

DEBUG_PROCESSING = os.environ.get("DEBUG_PROCESSING", "false").lower() == "true"
DEBUG_INDEXING_ROW_LIMIT = int(os.environ.get("DEBUG_INDEXING_ROW_LIMIT", "5"))

SUPPORTED_TYPES = ["parquet", "txt", "vtt", "pdf","csv"]

# If environment variables say we're using Bedrock then initialize the bedrock client
# The initialization of the bedrock client is used as an indicator that we're using Bedrock
if INFERENCE_ENGINE_API_PROVIDER == "bedrock":
    logger.info("Using Bedrock")
    bedrock_client = boto3.client(service_name = 'bedrock-runtime')
    logger.info("Bedrock client initialized")

if WEAVIATE_ENDPOINT_URL and WEAVIATE_AUTH_API_KEY and WEAVIATE_INFERENCE_ENGINE_API_KEY:
        
    # If we're not using bedrock we need to provide embedding engine API keys
    if bedrock_client is None:
        if INFERENCE_ENGINE_API_PROVIDER == "openai":
            weaviate_headers = {
                "X-OpenAI-Api-Key": WEAVIATE_INFERENCE_ENGINE_API_KEY
            }
        else:
            weaviate_headers = {
                "X-HuggingFace-Api-Key": WEAVIATE_INFERENCE_ENGINE_API_KEY
            }
    else:
        weaviate_headers = {}
    logger.info("Weaviate headers set")
    
    # Initialize the Weaviate client defined in the global scope
    weaviate_client = weaviate.Client(
        url = WEAVIATE_ENDPOINT_URL,
        auth_client_secret=weaviate.AuthApiKey(WEAVIATE_AUTH_API_KEY),
        additional_headers = weaviate_headers
    )
    logger.info("Weaviate client initialized")
else:
    raise Exception(f"WEAVIATE_ENDPOINT_URL, WEAVIATE_AUTH_API_KEY, and WEAVIATE_INFERENCE_ENGINE_API_KEY must all be set. WEAVIATE_ENDPOINT_URL: {WEAVIATE_ENDPOINT_URL}, WEAVIATE_AUTH_API_KEY: {WEAVIATE_AUTH_API_KEY}, WEAVIATE_INFERENCE_ENGINE_API_KEY: {WEAVIATE_INFERENCE_ENGINE_API_KEY}")
    
def ensure_s3_client():
    global s3_client
    if s3_client is None:
        s3_client = boto3.client("s3")

#####
# Manage Global Skills

def ensure_weaviate_schema_class_for_skills():
    if not check_weaviate_schema_class_exists(global_skill_collection):
        create_weaviate_schema_class(global_skill_collection, weaviate_wrappers.SKILL_COLUMN_TYPES)


def get_global_skills():
    global global_skills
    global global_inactive_skills
    (global_skills, global_inactive_skills) = skill_processing.get_global_skills(global_skill_collection, skill_column_names, weaviate_client, bedrock_client, logger, DEBUG_PROCESSING = False)
    
# def get_global_skills():
#     logger.info("Refreshing global skills")
#     global global_skills
#     global_skills = {}
#     try:
#         ensure_weaviate_schema_class_for_skills()
#         result = weaviate_wrappers.weaviate_query(
#             global_skill_collection, 
#             skill_column_names, 
#             weaviate_client, 
#             bedrock_client,
#             additional_columns = ["id"],
#             debug = DEBUG_PROCESSING)
        
#         for skill in result:
#             if DEBUG_PROCESSING:
#                 logger.debug(f"skill: {skill}")
#             skill_name = skill["name"]
#             global_skill = {
#                 "status": skill["status"],
#                 "isActive": skill["isActive"],
#                 "id": skill["_additional"]["id"]
#             }
#             if global_skill["isActive"]:
#                 global_skills[skill_name] = global_skill
#             else:
#                 global_inactive_skills[skill_name] = global_skill
            
#         logger.info(f"global_skills count: {len(global_skills)}")
        
#         if DEBUG_PROCESSING:
#             logger.debug(f"global_skills: {global_skills}")
        
#         return global_skills
#     except Exception as e:
#         logger.exception(f"Failed to refresh global skills from Weaviate: {e}")
#         raise e

def collect_missing_global_skills(concordant_skills_map):
    logger.info("Collecting skills that have no concordant skills")
    skills_to_add = {}
    for skill in concordant_skills_map:
        if skill in global_inactive_skills:
            continue
        max_certainty = 0
        if len(concordant_skills_map[skill]) == 0:
            if skill in global_skills:
                raise Exception(f"Skill {skill} is already in global_skills but has no concordant skills. The skills embeddings may need to be updated.")
        else:
            max_certainty = max([r['certainty'] for r in concordant_skills_map[skill]])
        
        if DEBUG_PROCESSING:
            logger.debug(f"max_certainty for {skill}: {max_certainty}")
        
        if max_certainty < CERTAINTY_THRESHOLD_FOR_SKILL_ADDING:
            logger.info(f"Adding {skill} as new skill. The maximum certainty for concordant skills is {max_certainty}, which is less than the threshold of {CERTAINTY_THRESHOLD_FOR_SKILL_ADDING}.")
            skills_to_add[skill] = {
                "status": weaviate_wrappers.SkillStatuses.system_added_not_reviewed,
                "isActive": True
            }
            
    return skills_to_add

def insert_missing_global_skills(skills_to_add):
    logger.info("Inserting missing concordant skills")
    with weaviate_client.batch as batch:
        batch.batch_size = 100
        for skill in skills_to_add:
            logger.info(f"Inserting missing concordant skill {skill}")
            data_obj = {
                "name": skill,
                "status": skills_to_add[skill]["status"],
                "isActive": skills_to_add[skill]["isActive"]
            }
            if DEBUG_PROCESSING:
                logger.debug(f"Adding data object: {json.dumps(data_obj)}")
            if bedrock_client is not None:
                try:
                    # Create text to vectorize by combining all of the data that will be indexed and searched on
                    input_text = weaviate_wrappers.build_skill_embedding_string(skill)
                    if DEBUG_PROCESSING:    
                        logger.debug("input_text for embedding")
                        logger.debug(input_text)
                    
                    embedding = bedrock_api_wrappers.generate_embeddings(input_text, bedrock_client)

                    logger.info("Saving embedding to Weaviate")

                    batch.add_data_object(
                        data_object = data_obj,
                        class_name = global_skill_collection,
                        vector = embedding)
                except Exception as e:
                    # If there is an error with Bedrock then just write to the logs and continue
                    logger.exception(f"ERROR: Could not index object because of error {e}\n{object}")
                    raise e
            else:
                batch.add_data_object(
                    data_object = data_obj,
                    class_name = global_skill_collection)
                
def populate_object_skills_from_LLM(header_names, data_types, rows):
    header_names.append("skills")
    data_types.append("text[]")
    
    all_skills = set()
    count = len(rows)
    logger.info(f"Retrieving skills for {count} rows")
    
    text_index = header_names.index("text")
    
    for index, row in enumerate(rows):
        text_data = row[text_index]        
        
        logger.info(f"Retrieving skills for row {index + 1}/{count}")
        
        genai_prompt = f"""{anthropic.HUMAN_PROMPT}
        Generate a list of technical skills referenced in the following text:
        
        {text_data}
        
        Return just the list of skills as a JSON object in a property called skills, with human targeted statements outside of the json.
        
        Just return the JSON string.
        
        {anthropic.AI_PROMPT}
        """
        # logger.debug(f"genai_prompt: {genai_prompt}")
        bedrock_model_response = bedrock_client.invoke_model(
            body=json.dumps({"prompt": genai_prompt, 
                             "temperature": 0.0, 
                             "max_tokens_to_sample":1000}),
            modelId='anthropic.claude-v2',
            accept='application/json',
            contentType='application/json'
        )        
        
        if bedrock_model_response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise INTERNAL_SERVER_ERROR(f"Failed to execute Bedrock model: {bedrock_model_response['ResponseMetadata']['HTTPStatusCode']} {bedrock_model_response['body'].read()}")

        bedrock_model_response_body = json.loads(bedrock_model_response['body'].read())
                
        completion_str = bedrock_model_response_body["completion"]
        
        start_text = '{'
        start_index = completion_str.find(start_text)
        end_text = '}'
        end_index = completion_str.find(end_text, start_index) + len(end_text)
        cleaned = completion_str[start_index:end_index].strip()
        
        if DEBUG_PROCESSING:
            logger.debug("cleaned")
            logger.debug(cleaned)
        
        skills_list = []        
        if len(cleaned) > 0:
            skills_list_dic = json.loads(cleaned)
            if "skills" in skills_list_dic:
                skills_list = skills_list_dic["skills"]
        
        skills = set(skills_list)
        
        logger.info(f"Skills for row {index + 1}/{count}: {skills}")
        
        rows[index].append(list(skills))
    
        all_skills.update(skills)
        
    return all_skills

def get_concordant_skills(skills):
    logger.info("Getting concordant skills")
    concordant_skills = {}
    
    #filter = weaviate_client.query.Filter.by_property("isActive").equal(False)
    
    where = {
        "path": ["isActive"],
        "operator": "Equal",
        "valueBoolean": True
    }
    
    for skill in skills:
        logger.info(f"Getting concordant skills for {skill}")
        skill_query_text = weaviate_wrappers.build_skill_embedding_string(skill)
        response = weaviate_wrappers.weaviate_semantic_query(
            skill_query_text, 
            global_skill_collection, 
            ["name"], 
            weaviate_client, 
            CERTAINTY_THRESHOLD_FOR_SKILL_MATCHING, 
            bedrock_client,
            additional_columns = ["certainty"],
            where_filter = where,
            debug = DEBUG_PROCESSING)
        if DEBUG_PROCESSING:
            logger.debug(f"Concordant skills for {skill}: {response}")
        concordant_skills[skill] = [{'name': r['name'], 'certainty': r['_additional']['certainty'] }for r in response]
        
    return concordant_skills

def remove_weaviate_objects_for_s3_key(weaviate_class_name: str, s3bucket: str, s3key: str):
    logger.info(f"Removing Weaviate objects for s3key {s3bucket}{s3key}")
    successful = -1
    while successful != 0:
        response = weaviate_client.batch.delete_objects(
            class_name = weaviate_class_name,
            where = {
                    "operator": "And",
                    "operands": [
                        {
                            "path": ["s3Bucket"],
                            "operator": "Equal",
                            "valueString": s3bucket
                        },
                        {
                            "path": ["s3Key"],
                            "operator": "Equal",
                            "valueString": s3key
                        }
                    ]
                }
        )    
        results = response["results"]
        matches = results["matches"]
        if DEBUG_PROCESSING:
            logger.debug(f"matches: {matches}")
        successful = results["successful"]
        if successful != matches:
            raise Exception(f"Failed to remove {matches - successful} objects from Weaviate")
        elif matches > 0:
            logger.info(f"Removed {matches} objects from Weaviate")

#
#####


def add_file_metadata_column_map(file_metadata_column_map, header_names, rows):
    for column_name in file_metadata_column_map:
        header_names.append(column_name)       
        for row in rows:    
            row.append(file_metadata_column_map[column_name])            
    return header_names, rows

def index_local_file(local_file_path: str, weaviate_class_name: str, file_metadata_column_map) -> str:
    
    # Get the s3key file extension
    file_type = local_file_path.split(".")[-1].lower()
    
    # Validate the object is a supported type
    supported_types = SUPPORTED_TYPES
    if not file_type in supported_types:
        raise Exception(f"Object {local_file_path} is not a supported file type {supported_types}")
    
    # Call function that takes a path to the local temp parquet file and processes it into three collections:
    # 1. The header names in an array
    # 2. The data types in an array
    # 3. The data in an array of arrays
    if file_type in ("txt", "vtt"):
        header_names, data_types, rows = process_text_file(local_file_path)
    elif file_type == "parquet":
        header_names, data_types, rows = process_parquet_file(local_file_path)
    elif file_type == "pdf":
        header_names, data_types, rows = process_pdf_file(local_file_path)
    elif file_type == "csv":
        header_names, data_types, rows = process_csv_file(local_file_path)
    else:
        raise Exception(f"Unsupported file type {file_type}")
    logger.info(f'Processed {local_file_path}')
    
    if DEBUG_PROCESSING:
        rows = rows[:DEBUG_INDEXING_ROW_LIMIT]
    
    all_current_skills = populate_object_skills_from_LLM(header_names, data_types, rows)

    add_file_metadata_column_map(file_metadata_column_map, header_names, rows)

    concordant_skills_map = get_concordant_skills(all_current_skills)
    
    missing_global_skills = collect_missing_global_skills(concordant_skills_map)
    
    insert_missing_global_skills(missing_global_skills)

    # refresh global_skills
    get_global_skills()

    # Call function to see if schema class already exists (named after the s3key_folder)
    if not check_weaviate_schema_class_exists(weaviate_class_name):        
        # Log weaviate_class_properties
        logger.info(f"OBJECT_COLUMN_TYPES: {weaviate_wrappers.OBJECT_COLUMN_TYPES}")

        create_weaviate_schema_class(weaviate_class_name, weaviate_wrappers.OBJECT_COLUMN_TYPES)
        logger.info(f"Created Weaviate schema class named {weaviate_class_name}")
    else:
        logger.info(f"Weaviate schema class named {weaviate_class_name} already exists")

    remove_weaviate_objects_for_s3_key(weaviate_class_name, file_metadata_column_map["s3Bucket"], file_metadata_column_map["s3Key"])
    data_obj_map = index_weaviate_data(weaviate_class_name, rows, header_names, concordant_skills_map)
    weaviate_wrappers.add_object_to_global_skills_reference(weaviate_class_name, data_obj_map, global_skill_collection, global_skills, weaviate_client)
    
    logger.info(f"Indexed {len(rows)} objects into Weaviate class {weaviate_class_name}")    
    
    return f"Successfully indexed {len(rows)} objects into Weaviate class {weaviate_class_name}"

def index_s3_file(s3bucket: str, s3key: str) -> str:

    # Get the top level folder name from the s3key
    s3key_folder = s3key.split("/")[0]

    # Validate the s3key_folder is actually a folder and not an object
    if "." in s3key_folder:
        raise Exception(f"Object {s3key} is not a folder")

    # Get the s3key file extension
    file_type = s3key.split(".")[-1].lower()

    # Validate the object is a supported type
    supported_types = SUPPORTED_TYPES
    if not file_type in supported_types:
        raise Exception(f"Object {s3key} is not a supported file type {supported_types}")
    
    weaviate_class_name = s3key_folder
    # Remove any characters from the weaviate_class_name that don't conform to the following regex: /^[A-Z][_0-9A-Za-z]*$/
    weaviate_class_name = re.sub(r"[^A-Z0-9_]", "", weaviate_class_name, flags=re.IGNORECASE)
    # Capitalize the first letter of the weaviate_class_name to conform to Weaviate naming conventions
    weaviate_class_name = weaviate_class_name[0].upper() + weaviate_class_name[1:]
    
    # Download the parquet file to /tmp
    # Log the object being downloaded
    tmpfile = f"/tmp/{file_type}_file.{file_type}"
    logger.info(f"Downloading {s3key} from bucket {s3bucket} to {tmpfile}")
    ensure_s3_client()
    s3_client.download_file(s3bucket, s3key, tmpfile)
    logger.info(f"Downloaded {s3key} to {tmpfile}")
    
    return index_local_file(tmpfile, weaviate_class_name, {"s3Bucket": s3bucket, "s3Key": s3key})

# Function that checks to see if a Weaviate schema class exists
def check_weaviate_schema_class_exists(weaviate_class_name: str):
    weaviate_schema = weaviate_client.schema.get()
    
    # Pull the class names from the array of classes
    weaviate_class = None
    for weaviate_class_from_schema in weaviate_schema["classes"]:
        if weaviate_class_name == weaviate_class_from_schema["class"]:
            weaviate_class = weaviate_class_from_schema
    
    # Verify the provided domain is in the list of classes
    if weaviate_class is None:
        return False
    
    return True

# Function that creates a Weaviate schema class
def create_weaviate_schema_class(class_name: str, weaviate_class_schema: list):
    # Create weaviate class in weaviate schema
    if os.environ["INFERENCE_ENGINE_API_PROVIDER"] == "openai":
        class_obj = {
            "class": class_name,
            "vectorizer": "text2vec-openai",
            "properties": weaviate_class_schema
        }
    else:
        # https://weaviate.io/developers/weaviate/modules/retriever-vectorizer-modules/text2vec-huggingface#how-to-configure
        class_obj = {
            "class": class_name,
            "vectorizer": "text2vec-huggingface",
            "moduleConfig": {
                "text2vec-huggingface": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "options": {
                        "waitForModel": True,
                        "useGPU": False,
                        "useCache": True
                    }
                }
            },
            "properties": weaviate_class_schema
        }
    weaviate_client.schema.create_class(
        class_obj
    )

def check_batch_result(batch_result):
    logger.info("Checking batch result")
    logger.info(batch_result)

# Function that takes the data and indexes it into Weaviate
def index_weaviate_data(schema_class_name: str, data: list, header_names: list, concordant_skills_map: dict):
    logger.info(f"Indexing {len(data)} objects into Weaviate class {schema_class_name} with headers {header_names}")
    
    obj_map = {}
    
    with weaviate_client.batch as batch:
        batch.batch_size = 100
        batch.callback = check_batch_result
        
        if DEBUG_PROCESSING:
            logger.debug(header_names)
        header_names.append("id")
        header_names.append("concordantSkills")

        for index, row in enumerate(data):
            
            logger.info(f"Indexing row {index + 1}/{len(data)}")
            if DEBUG_PROCESSING:
                logger.debug(row)
            
            # Create a data object that maps the header names to the data
            data_obj = {}
            data_obj["concordantSkills"] = []
            skills_list = []                        
            
            for index, column_data in enumerate(row):
                if column_data is not None and column_data == column_data: # The column == column check is checking for NaN while also working for non-floats
                    # if column is a date convert it to a string
                    if  "date" in str(type(column_data)).lower() or \
                        "time" in str(type(column_data)).lower():
                        column_data = str(column_data)
                    column_name = header_names[index]
                    if column_name in object_column_names:                        
                        data_obj[column_name] = column_data
                    if column_name == "skills":
                        skills_list = column_data
            
            # Ensure that all of the object_column_names are present in the data object
            missing_columns = list(set(object_column_names) - set(data_obj.keys()))
            if len(missing_columns) > 0:
                raise Exception(f"Missing columns {missing_columns} for data object {data_obj}")

            # Log data object being indexed
            if DEBUG_PROCESSING:
                logger.debug(f"Indexing data object {index}: {data_obj}")
            
            if bedrock_client is not None:                
                try:
                    if len(skills_list) > 0:
                        concordant_skills_list = set()
                        for skill in skills_list:
                            concordant_skills_list.update([skill['name'] for skill in concordant_skills_map[skill]])
                        data_obj["concordantSkills"] = list(concordant_skills_list)
                        if DEBUG_PROCESSING:
                            logger.debug(f"Concordant skills for {skills_list}: {data_obj['concordantSkills']}")
                    
                    # Create text to vectorize by combining all of the data that will be indexed and searched on
                    embedding = bedrock_api_wrappers.generate_embeddings(weaviate_wrappers.get_content_chunk_embedding_string(data_obj), bedrock_client)
                    
                    # logger.debug(embedding)

                    rowId = generate_uuid5(data_obj)
                    batch.add_data_object(
                        data_object = data_obj,
                        class_name = schema_class_name,
                        vector = embedding,
                        uuid = rowId)
                    
                except Exception as e:
                    # If there is an error with Bedrock then just write to the logs and continue
                    logger.exception(f"ERROR: Could not index object because of error {e}\n{object}")
                    raise e
            else:
                rowId = generate_uuid5(data_obj)
                batch.add_data_object(
                    data_object = data_obj,
                    class_name = schema_class_name,
                    uuid = rowId)
            
            row.append(rowId)
            row.append(data_obj["concordantSkills"])
            obj_map[rowId] = data_obj
            
    return obj_map

def process_pandas_dataframe(dataframe):
    # Get the header names
    header_names = list(dataframe.columns)

    # Prepend the header names with "col_" to conform to Weaviate naming conventions
    header_names = list(map(lambda x: "col_" + x, header_names))

    # Get the data types
    data_types = list(dataframe.dtypes)

    # Get the data
    data = dataframe.values.tolist()

    # Return the results
    return header_names, data_types, data

def process_parquet_file(parquet_file_path):
    # Read the parquet file
    parquet_data = pd.read_parquet(parquet_file_path)

    return process_pandas_dataframe(parquet_data)

def process_csv_file(csv_file_path):
    # Read the csv file
    csv_data = pd.read_csv(csv_file_path)

    return process_pandas_dataframe(csv_data)

def process_text_file(text_file_path):
    # Read the text file as a single string
    with open(text_file_path, "r") as file:
        text_data = file.read()    

    return process_text_data(text_data)
    
def process_pdf_file(pdf_file_path):
    reader = PdfReader(pdf_file_path)

    return_header_names, return_data_types, return_data = [], [], []

    # Loop over the pages independently as it's a good semantic boundary
    page_counter = 0
    for page in reader.pages:
        page_counter += 1
        logger.info(f"Processing page {page_counter} of {len(reader.pages)} from file {pdf_file_path}")

        text_data = page.extract_text()
        header_names, data_types, rows = process_text_data(text_data)

        # Yes, we're reassigning these variables on every iteration of the loop. It's ok as it keep the code simpler and
        # incurs negligable overhead.
        return_header_names = header_names
        return_data_types = data_types
        return_data.extend(rows)
    
    logger.info(f'Split pdf into {len(return_data)} chunks')

    return return_header_names, return_data_types, return_data

# Function that takes a string and splits it into chunks
# The url is an example of adding metadata to the data
def process_text_data(text_data, url=None):
    # Get the header names
    header_names = ["text"]
    if url:
        header_names.append("url")

    # Get the data types
    data_types = ["string"]
    if url:
        data_types.append("string")

    # Get the data
    text_splitter = TokenTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    
    split_data = text_splitter.split_text(text_data)

    rows = []

    for _, split_text in enumerate(split_data):
        row = [split_text]
        if url:
            row.append(url)
        rows.append(row)

    logger.info(f"Split text into {len(rows)} chunks")

    # Return the results
    return header_names, data_types, rows
