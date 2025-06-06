import logging
import sys
import json

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("mcp-cost")

try:
    config = json.load(open('application/config.json'))
    tavily_api_key = config.get("TAVILY_API_KEY", "")
    if not tavily_api_key:
        logger.warning("TAVILY_API_KEY is not set in config.json")
except (FileNotFoundError, json.JSONDecodeError) as e:
    logger.warning(f"Error reading config.json file: {e}")
    config = {}
    tavily_api_key = ""

mcp_user_config = {}    
def load_config(mcp_type):
    if mcp_type == "default":
        return {
            "mcpServers": {
                "search": {
                    "command": "python",
                    "args": [
                        "application/mcp_server_basic.py"
                    ]
                }
            }
        }
    
    elif mcp_type == "code_interpreter":
        return {
            "mcpServers": {
                "aws_storage": {
                    "command": "python",
                    "args": [
                        "application/mcp_server_coder.py"
                    ]
                }
            }
        }    

    elif mcp_type == "aws_documentation":
        return {
            "mcpServers": {
                "awslabs.aws-documentation-mcp-server": {
                    "command": "uvx",
                    "args": ["awslabs.aws-documentation-mcp-server@latest"],
                    "env": {
                        "FASTMCP_LOG_LEVEL": "ERROR"
                    }
                }
            }
        }
    
    elif mcp_type == "aws_cli":
        return {
            "mcpServers": {
                "aws-cli": {
                    "command": "python",
                    "args": [
                        "application/mcp_server_aws_cli.py"
                    ]
                }
            }
        }    
    
    elif mcp_type == "aws_cloudwatch":
        return {
            "mcpServers": {
                "aws_cloudwatch_log": {
                    "command": "python",
                    "args": [
                        "application/mcp_server_aws_log.py"
                    ]
                }
            }
        }    
    
    elif mcp_type == "aws_storage":
        return {
            "mcpServers": {
                "aws_storage": {
                    "command": "python",
                    "args": [
                        "application/mcp_server_aws_storage.py"
                    ]
                }
            }
        }    
    
    elif mcp_type == "aws_diagram":
        return {
            "mcpServers": {
                "awslabs.aws-diagram-mcp-server": {
                    "command": "uvx",
                    "args": ["awslabs.aws-diagram-mcp-server"],
                    "env": {
                        "FASTMCP_LOG_LEVEL": "ERROR"
                    },
                }
            }
        }

    elif mcp_type == "tavily":
        if not tavily_api_key:
            logger.warning("Tavily server is disabled due to missing API key")
            return {}
        return {
            "mcpServers": {
                "tavily-mcp": {
                    "command": "npx",
                    "args": ["-y", "tavily-mcp@0.1.4"],
                    "env": {
                        "TAVILY_API_KEY": tavily_api_key
                    },
                }
            }
        }
    
    elif mcp_type == "arxiv":
        return {
            "mcpServers": {
                "arxiv-mcp-server": {
                    "command": "npx",
                    "args": [
                        "-y",
                        "@smithery/cli@latest",
                        "run",
                        "arxiv-mcp-server",
                        "--config",
                        "{\"storagePath\":\"/Users/ksdyb/Downloads/ArXiv\"}"
                    ]
                }
            }
        }
    
    elif mcp_type == "wikipedia":
        return {
            "mcpServers": {
                "wikipedia": {
                    "command": "python",
                    "args": [
                        "application/mcp_server_wikipedia.py"
                    ]
                }
            }
        }      
        
    elif mcp_type == "filesystem":
        return {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": [
                        "@modelcontextprotocol/server-filesystem",
                        "~/"
                    ]
                }
            }
        }
    
    elif mcp_type == "use_aws":
        return {
            "mcpServers": {
                "use_aws": {
                    "command": "python",
                    "args": [
                        "application/mcp_server_use_aws.py"
                    ]
                }
            }
        }
    
    elif mcp_type == "playwright":
        return {
            "mcpServers": {
                "playwright": {
                    "command": "npx",
                    "args": [
                        "@playwright/mcp@latest"
                    ]
                }
            }
        }

    elif mcp_type == "user_config":
        return mcp_user_config

def load_selected_config(mcp_selections: dict[str, bool]):
    #logger.info(f"mcp_selections: {mcp_selections}")
    loaded_config = {}

    # Convert keys with True values to a list
    selected_servers = [server for server, is_selected in mcp_selections.items() if is_selected]
    logger.info(f"selected_servers: {selected_servers}")

    for server in selected_servers:
        logger.info(f"server: {server}")

        if server == "image generation":
            config = load_config('image_generation')
        elif server == "aws diagram":
            config = load_config('aws_diagram')
        elif server == "aws document":
            config = load_config('aws_documentation')
        elif server == "aws cost":
            config = load_config('aws_cost')
        elif server == "ArXiv":
            config = load_config('arxiv')
        elif server == "aws cloudwatch":
            config = load_config('aws_cloudwatch')
        elif server == "aws storage":
            config = load_config('aws_storage')
        elif server == "code interpreter":
            config = load_config('code_interpreter')
        elif server == "aws cli":
            config = load_config('aws_cli')
        elif server == "text editor":
            config = load_config('text_editor')
        else:
            config = load_config(server)
        logger.info(f"config: {config}")
        
        if config:
            loaded_config.update(config["mcpServers"])

    logger.info(f"loaded_config: {loaded_config}")
        
    return {
        "mcpServers": loaded_config
    }
