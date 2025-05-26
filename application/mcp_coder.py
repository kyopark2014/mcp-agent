import re
import base64
import logging
import traceback
import chat 
import sys
import uuid
import os
from PIL import Image

from urllib import parse
from langchain_experimental.tools import PythonAstREPLTool
from io import BytesIO

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("mcp-cost")

repl = PythonAstREPLTool()

def repl_coder(code):
    """
    Use this to execute python code and do math. 
    If you want to see the output of a value, you should print it out with `print(...)`. This is visible to the user.
    code: the Python code was written in English
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    if result is None:
        result = "It didn't return anything."

    return result

def generate_short_uuid(length=8):
    full_uuid = uuid.uuid4().hex
    return full_uuid[:length]

def repl_drawer(code):
    """
    Execute a Python script for draw a graph.
    Since Python runtime cannot use external APIs, necessary data must be included in the code.
    The graph should use English exclusively for all textual elements.
    Do not save pictures locally bacause the runtime does not have filesystem.
    When a comparison is made, all arrays must be of the same length.
    code: the Python code was written in English
    return: the url of graph
    """ 
        
    code = re.sub(r"seaborn", "classic", code)
    code = re.sub(r"plt.savefig", "#plt.savefig", code)
    code = re.sub(r"plt.show", "#plt.show", code)

    post = """\n
import io
import base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
image_base64 = base64.b64encode(buffer.getvalue()).decode()

print(image_base64)
"""
    code = code + post    
    logger.info(f"code: {code}")
    
    image_url = ""
    try:     
        resp = repl.run(code)
        logger.info(f"repl.run response: {resp}")

        base64Img = resp
        
        if base64Img:
            byteImage = BytesIO(base64.b64decode(base64Img))

            image_name = generate_short_uuid()+'.png'
            
            # Create artifacts directory
            artifacts_dir = "artifacts"
            if not os.path.exists(artifacts_dir):
                os.makedirs(artifacts_dir)
            
            # Save image
            im = Image.open(BytesIO(base64.b64decode(base64Img)))
            image_path = os.path.join(artifacts_dir, image_name)
            im.save(image_path, 'PNG')
            
            # Set image_url to local path
            image_url = image_path
            logger.info(f"Generated image_url: {image_url}")

    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")

    result = {"path": image_url}
    logger.info(f"Returning result: {result}")
    return result
