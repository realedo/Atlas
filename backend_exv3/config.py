import os
from dotenv import load_dotenv

#----env varaibles/secrets----#
load_dotenv()
MODEL_DIR = os.environ.get("dir_model")
#----#

#---- model settings ----#
MAX_CONTEXT =64000 #trial and error!has to be X of 256   --- 24576 --32768 --51200
chache_quantization = 4 # weights quantization (keep at 4 for balance)
kv_cache_bits = 4 # down from 6 to save ~225MB VRAM for 45K context

# generation settings - optimized for speed
default_temp = 0.3 # lowered from 0.6 for faster sampling
default_max_tok_out = 4352 # reduced from 8192 to limit prefill time
min_tokens_think = 1024
min_free_tokens_for_inf = 2048 # reduced from 4096 to allow more context usage
max_memory_turns =10
sliding_window_trocate_buffer = 8192 # enabled - keeps only recent 8K tokens for faster prefill
max_retries_on_stream = 3
gen_timeout_seconds = 600
query_transform_temperature = 0.1
query_transform_max_tokens = 300
#----#

#----web search settings ----#
WEB_TOGGLE = False  #toggles web research
MX_RES = 5 #max web searches allowed (DDGS)
mx_trafilatura_per_prompt = 4
append_links = False
append_images = False
max_url_scrape_len = 4500
#----#