import os
from dotenv import load_dotenv

#----env varaibles/secrets----#
load_dotenv()
MODEL_DIR = os.environ.get("dir_model")
#----#

#---- model settings ----#
MAX_CONTEXT = 8960#trial and error for my gpu !has to be X of 256
default_temp = 0.8
default_max_tok_out = 5120
min_free_tokens_for_inf = 400
max_memory_turns =4


query_transform_temperature = 0.1
query_transform_max_tokens = 300
#----#

#----web search settings ----#
MX_RES = 3 #max web searches allowed (DDGS)
mx_trafilatura_per_prompt = 2
WEB_TOGGLE = True #toggles web research
append_links = False
append_images = False
max_url_scrape_len = 1500
#----#