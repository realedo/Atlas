import os
from dotenv import load_dotenv

#----env varaibles/secrets----#
load_dotenv()
MODEL_DIR = os.environ.get("dir_model")
#----#

#---- model settings ----#
MAX_CONTEXT =64000#trial and error for my gpu !has to be X of 256   --- 24576 --32768 --51200
chache_quantization = 4
default_temp = 0.6
default_max_tok_out = 8192
min_free_tokens_for_inf = 4096
max_memory_turns =4
sliding_window_trocate_buffer = 400

query_transform_temperature = 0.1
query_transform_max_tokens = 300
#----#

#----web search settings ----#
MX_RES = 3 #max web searches allowed (DDGS)
mx_trafilatura_per_prompt = 2
WEB_TOGGLE = False #toggles web research
append_links = False
append_images = False
max_url_scrape_len = 1500
#----#