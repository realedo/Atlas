import os
from dotenv import load_dotenv

#----env varaibles/secrets----#
load_dotenv()
MODEL_DIR = os.environ.get("dir_model")
#----#

#---- model settings ----#
MAX_CONTEXT = 6000 #trial and error for my gpu
default_temp = 0.6
default_max_tok_out = 2000
min_free_tokens_for_inf = 400


query_transform_temperature = 0.1
query_transform_max_tokens = 20
#----#

#----web search settings ----#
MX_RES = 3 #max web searches allowed (DDGS)
mx_trafilatura_per_prompt = 2
WEB_TOGGLE = True #toggles web research
append_links = False
append_images = False
max_url_scrape_len = 4000
#----#