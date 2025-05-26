import traceback
import json
import time
import boto3

from langchain_aws import AmazonKnowledgeBasesRetriever
from urllib import parse
from langchain.docstore.document import Document
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

import logging
import sys
logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("knowledge_base")

config, environment = utils.load_config()

# variables
projectName = config["projectName"] if "projectName" in config else "langgraph-nova"

vectorIndexName = projectName
knowledge_base_name = projectName
bedrock_region = config["region"] if "region" in config else "us-west-2"
region = config["region"] if "region" in config else "us-west-2"
logger.info(f"region: {region}")
s3_bucket = config["s3_bucket"] if "s3_bucket" in config else None
if s3_bucket is None:
    raise Exception ("No storage!")

parsingModelArn = f"arn:aws:bedrock:{region}::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0"
embeddingModelArn = f"arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v2:0"

collectionArn = config["collectionArn"] if "collectionArn" in config else None
if collectionArn is None:
    raise Exception ("No collectionArn")

knowledge_base_role = config["knowledge_base_role"] if "knowledge_base_role" in config else None
if knowledge_base_role is None:
    raise Exception ("No Knowledge Base Role")

s3_arn = config["s3_arn"] if "s3_arn" in config else None
if s3_arn is None:
    raise Exception ("No S3 ARN")

path = config["sharing_url"] if "sharing_url" in config else None
if path is None:
    raise Exception ("No Sharing URL")

opensearch_url = config["opensearch_url"] if "opensearch_url" in config else None
if opensearch_url is None:
    raise Exception ("No OpenSearch URL")

credentials = boto3.Session().get_credentials()
service = "aoss" 
awsauth = AWSV4SignerAuth(credentials, region, service)

def print_doc(i, doc):
    if len(doc.page_content)>=100:
        text = doc.page_content[:100]
    else:
        text = doc.page_content
            
    logger.info(f"{i}: {text}, metadata:{doc.metadata}")

s3_prefix = 'docs'
doc_prefix = s3_prefix+'/'

# Knowledge Base
knowledge_base_id = ""
data_source_id = ""

os_client = OpenSearch(
    hosts = [{
        'host': opensearch_url.replace("https://", ""), 
        'port': 443
    }],
    http_auth=awsauth,
    use_ssl = True,
    verify_certs = True,
    connection_class=RequestsHttpConnection,
)

def is_not_exist(index_name):    
    logger.info(f"index_name: {index_name}")
        
    if os_client.indices.exists(index_name):
        logger.info(f"use exist index: {index_name}")
        return False
    else:
        logger.info(f"no index: {index_name}")
        return True
    
def retrieve_documents_from_knowledge_base(query, top_k):
    relevant_docs = []
    if knowledge_base_id:    
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=knowledge_base_id, 
            retrieval_config={"vectorSearchConfiguration": {
                "numberOfResults": top_k,
                "overrideSearchType": "HYBRID"   # SEMANTIC
            }},
            region_name=bedrock_region
        )
        
        try: 
            documents = retriever.invoke(query)
            # print('documents: ', documents)
            logger.info(f"--> docs from knowledge base")
            for i, doc in enumerate(documents):
                print_doc(i, doc)
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")    
            raise Exception ("Not able to request to LLM: "+err_msg)
        
        relevant_docs = []
        for doc in documents:
            content = ""
            if doc.page_content:
                content = doc.page_content
            
            score = doc.metadata["score"]
            
            link = ""
            if "s3Location" in doc.metadata["location"]:
                link = doc.metadata["location"]["s3Location"]["uri"] if doc.metadata["location"]["s3Location"]["uri"] is not None else ""
                
                # print('link:', link)    
                pos = link.find(f"/{doc_prefix}")
                name = link[pos+len(doc_prefix)+1:]
                encoded_name = parse.quote(name)
                # print('name:', name)
                link = f"{path}/{doc_prefix}{encoded_name}"
                
            elif "webLocation" in doc.metadata["location"]:
                link = doc.metadata["location"]["webLocation"]["url"] if doc.metadata["location"]["webLocation"]["url"] is not None else ""
                name = "WEB"

            url = link
            logger.info(f"url: {url}")
            
            relevant_docs.append(
                Document(
                    page_content=content,
                    metadata={
                        'name': name,
                        'score': score,
                        'url': url,
                        'from': 'RAG'
                    },
                )
            )    
    return relevant_docs

def sync_data_source():
    if knowledge_base_id and data_source_id:
        try:
            client = boto3.client(
                service_name='bedrock-agent',
                region_name=bedrock_region                
            )
            response = client.start_ingestion_job(
                knowledgeBaseId=knowledge_base_id,
                dataSourceId=data_source_id
            )
            logger.info(f"(start_ingestion_job) response: {response}")
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")


    
