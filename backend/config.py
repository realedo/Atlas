import os
from dotenv import load_dotenv

#----env varaibles/secrets----#
load_dotenv()
MODEL_DIR = os.environ.get("dir_model")
default_temp = 0.7
default_max_tok_out = 1024
#----#

#---- model settings ----#
MAX_CONTEXT = 4096 #trial and error for my gpu
#----#

#----web search settings ----#
MX_RES = 3 #max web searches allowed
WEB_TOGGLE = True #toggles web research
append_links = False
append_images = False
max_url_scrape_len = 4000
query_transform_temperature = 0.1
query_transform_max_tokens = 15
#----#