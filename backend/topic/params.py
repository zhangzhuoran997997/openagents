PROBABILITY_THRESHOLD = 0.2
PROBABILITY_MEDIA_THRESHOLD = 0.9
MEDIA_COUNTRY_NUM = 5
ENTITY_NUM = 3
PEOPLE_NUM = 10
STOP_PEOPLE_LIST = ['Civilian', 'United Sates', 'Tai Wan', 'Saint Kitts And Nevis', 'Falun Gong', 'Ho Chi Minh City',
                    'S. Korea', 'Port Au Prince','Santo Domingo']
RANDOM_SEED = 42
DOC_LENGTH = 800
NEWS_LENGTH_THRESHOLD = 2000
CHN_LIST = ['CHN', 'TWN', 'HKG', 'MAC']
USA_LIST = ['United States', 'U.S.', ' US']
feedback_dict = {"政治磋商": '04', "公开声明": '01', "外交合作": '05', '会见谈判': '036', '战争对抗': '19',
                 "军事展示": '15'}
PERSON_TOPIC_NUM = 5


NEWS_DIR = '/data/llmagents/data/llm_agent/tai_news_0526/'
# RES_DIR = '/home/dingfei/taihai/result/document'  # document/event-level
sent_embed_model_dir = '/data/llmagents/plm/all-MiniLM-L6-v2-copy'
sent_embed_save_dir = '/data/llmagents/data/llm_agent/tai_news_0526/sent_embed/'
TABLE_DATA_DIR = '/data/llmagents/data/llm_agent/tai_news_0526/table_data/'
#save_path = '/data/llmagents/code/OpenAgents/backend/data/DefaultUser/'
save_path = '/data/zhuoran/code/openagents/backend/data/DefaultUser/'
model_id = '/data/llmagents/plm/llama-2-13B-chat-hf-copy'
# model_id = '/data/llmagents/plm/llama-2-7B-chat-hf'

# ABSA_MODEL_ID = '/data/dingfei/models/APC'