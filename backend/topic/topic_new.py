import json
import pandas as pd
import os
import re
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import gc
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech, TextGeneration
from bertopic import BERTopic
import collections
import torch
from torch.utils.data import Dataset
import transformers
import pickle
from backend.topic.params import TABLE_DATA_DIR, NEWS_DIR, sent_embed_model_dir, sent_embed_save_dir, model_id, \
     DOC_LENGTH, RANDOM_SEED, save_path

from backend.topic.utils import write_data, get_data, get_json, dump_json

# from params import TABLE_DATA_DIR, NEWS_DIR, sent_embed_model_dir, sent_embed_save_dir, model_id, \
#      DOC_LENGTH, RANDOM_SEED

# from utils import write_data, get_data, get_json, dump_json


torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

from collections import defaultdict

# spacy.load("en_core_web_sm")
# nltk.download('punkt')


def clean_file_read(path):
    '''
    read clean data
    :param path: the path to raw news
    :return: cleannews
    '''
    # newpath = os.path.join(NEWS_DIR, 'tw_news' + str(2023) + '_r3k.json')

    news = get_data(path)
    statistics = {}
    maxtime, mintime = "", "3000-01-01"
    urls = []
    lengths = []

    # ['title', 'content', 'date', 'source', 'url', 'stanford']
    cleannews = []
    for line in news:
        tmp = line['title'] + '\n' + line['content']
        # re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', "", tmp)
        tmp = re.sub(r'http\S+', '', tmp)
        cleannews.append(tmp)
        lengths.append(len(tmp.split(' ')))
        if 'https://' in line['url']:
            url = line['url'].split('https://')[-1].split('/')[0]
        elif 'http://' in line['url']:
            url = line['url'].split('http://')[-1].split('/')[0]
        else:
            url=" "
        if url not in urls:
            urls.append(url)
        if line['date'] > maxtime:
            maxtime = line['date']
        if line['date'] < mintime:
            mintime = line['date']
    
    statistics["count"] = len(cleannews)
    statistics["url_count"] = len(urls)
    statistics["maxtime"] = maxtime
    statistics["mintime"] = mintime
    statistics["lengths"] = sum(lengths) / len(cleannews)
    statistics["path"] = path

    print('clean news:', len(cleannews))
    # print(cleannews[0])
    return statistics, cleannews



def news_topic_classify(path):
    '''
    runtime: ~10min
    classify topic based on news, save bertopic models to document_full_2023 (safetensors)
    save topic table and news table to topic_table_2023.csv, document_table_2023.csv
    :param path:
    :return: topic_model
    '''
    # read news contents
    stats, contents = clean_file_read(path)

    print("Statics: ", stats)
    index = path.split('/')[-1].split('.')[0]
    
    # save sentence embedding
    sent_embed_save_file = os.path.join(sent_embed_save_dir, f'embeddings_{index}.npy')
    embedding_model = SentenceTransformer(sent_embed_model_dir)
    if not os.path.exists(sent_embed_save_file):
        print(f'produce news embedding of {index}.')
        # sentence-transformers/all-MiniLM-L6-v2
        embeddings = embedding_model.encode(contents, show_progress_bar=True)  # 66125*384
        # save sentence embeddings
        os.makedirs(sent_embed_save_dir, exist_ok=True)
        np.save(sent_embed_save_file, embeddings)
    embeddings = np.load(sent_embed_save_file)
    # save topic model
    topic_model_save_file = os.path.join(sent_embed_save_dir, f'document_{index}.pkl')
    if not os.path.exists(topic_model_save_file):
        print(f'produce topic model of {index}.')
        # default
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        # default: =min_topic_size=min_cluster_size=10, big: large cluster
        hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom',
                                prediction_data=True)
        # default: stop_words=None, min_df=1, ngram_range=(1,1)
        vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
        # KeyBERT
        keybert_model = KeyBERTInspired()
        # Part-of-Speech
        pos_model = PartOfSpeech("en_core_web_sm")
        # MMR
        mmr_model = MaximalMarginalRelevance(diversity=0.3)
        # All representation models
        representation_model = {
            "KeyBERT": keybert_model,
            "MMR": mmr_model,
            "POS": pos_model
        }
        topic_model = BERTopic(
            # Pipeline models
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            # Hyperparameters
            top_n_words=10,  # default
            verbose=True  # Set to True if you want to track the stages of the model.
        )
        topics, probs = topic_model.fit_transform(contents, embeddings)
        topic_model.save(topic_model_save_file, serialization="pickle", save_ctfidf=True,
                         save_embedding_model=embedding_model)
    topic_model = BERTopic.load(topic_model_save_file, embedding_model=embedding_model)
    # save table topic
    topic_info = topic_model.get_topic_info()
    # json dump list
    for col in ['Representation', 'KeyBERT', 'MMR', 'POS', 'Representative_Docs']:
        topic_info[col] = topic_info[col].apply(lambda x: json.dumps(x))
    topic_info = topic_info.drop(columns=['Representative_Docs'])
    topic_info.to_csv(os.path.join(TABLE_DATA_DIR, f'topic_table_{index}.csv'), index=False)
    # table: topic (Topic,Count,Name,Representation,KeyBERT,MMR,POS)
    # save table news
    document_info = topic_model.get_document_info(contents)
    document_info = document_info.drop(
        columns=['Representation', 'KeyBERT', 'MMR', 'POS', 'Representative_Docs', 'Top_n_words'])
    document_info = document_info.sort_values(by=['Topic', 'Probability'], ascending=[True, False])
    document_info = document_info.drop(columns=['Document'])
    document_info.to_csv(os.path.join(TABLE_DATA_DIR, f'document_table_{index}.csv'), index=True)
    print("Topic cluster finish!")
    # return stats, topic_model
    del topic_model, embedding_model
    gc.collect()
    torch.cuda.empty_cache()
    return stats

    # table: news (index, Topic, Name, Probability, Representative_document)



def generate_topic_name_description(path):
    '''
    runtime: ~20min; memory: 13116MB
    generate topic name and description with llama-2-13b-chat 4bit quantization
    save bertopic models to document_full_index.pkl
    save topic table and news table to topic_table_full_index.csv
    :param path:
    :return: topic_model
    '''

    index = path.split('/')[-1].split('.')[0]
    # topic name begin
    embedding_model = SentenceTransformer(sent_embed_model_dir)
    topic_model_save_file = os.path.join(sent_embed_save_dir, f'document_{index}.pkl')
    topic_model = BERTopic.load(topic_model_save_file, embedding_model=embedding_model)
    stats, contents = clean_file_read(path)
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    save_dir = os.path.join(sent_embed_save_dir, f'document_full_{index}')
    if not os.path.exists(save_dir):
        print(f'produce full topic model of {index}.')
        os.makedirs(save_dir, exist_ok=True)
        # set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,  # 4-bit quantization
            bnb_4bit_quant_type='nf4',  # Normalized float 4
            bnb_4bit_use_double_quant=True,  # Second quantization after the first
            bnb_4bit_compute_dtype=torch.bfloat16  # Computation type
        )
        # Llama 2 Tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

        # Llama 2 Model
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map={"":0},
            # device_map='auto',
            # load_in_8bit=True
        )
        model.eval()
        # topic name generate
        generator_name = transformers.pipeline(
            model=model, tokenizer=tokenizer,
            task='text-generation',
            temperature=0.1,  # smaller to sharper token distribution
            max_new_tokens=50,  # label max tokens=50
            repetition_penalty=1.1
        )
        # System prompt describes information given to all conversations
        system_prompt = """<s>[INST] <<SYS>>
        You are a helpful, respectful and honest assistant for labeling topics.
        <</SYS>>
        """
        # Example prompt demonstrating the output we are looking for
        example_prompt = """
        I have a topic that contains the following documents:
        - China threatened retaliation on Wednesday if U.S. House Speaker Kevin McCarthy meets with Taiwan's president during her upcoming trip through Los Angeles. President Tsai Ing-wen left Taiwan Wednesday afternoon on a tour of the island's diplomatic allies in the Americas, which she framed as a chance to demonstrate Taiwan’s commitment to democratic values on the world stage.
        - Following a visit by then-House Speaker Nancy Pelosi to Taiwan in 2022, Beijing launched missiles over the area, deployed warships across the median line of the Taiwan Strait and carried out military exercises in a simulated blockade of the island. Beijing also suspended climate talks with the U.S. and restricted military-to-military communication with the Pentagon.
        - Risking China’s anger, House Speaker Kevin McCarthy hosted Taiwan President Tsai Ing-wen on Wednesday as a “great friend of America” in a fraught show of U.S. support at a rare high-level, bipartisan meeting on U.S. soil. Speaking carefully to avoid unnecessarily escalating tensions with Beijing, Tsai and McCarthy steered clear of calls from hard-liners in the U.S. for a more confrontational stance toward China in defense of self-ruled Taiwan.

        The topic is described by the following keywords: 'tsai, taiwan, mccarthy, speaker, china, meeting, house, beijing, house speaker, visit'.

        Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.

        [/INST] US House speaker and Taiwan president meet as China protests
        """

        # Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
        main_prompt = """
        [INST]
        I have a topic that contains the following documents:
        [DOCUMENTS]

        The topic is described by the following keywords: '[KEYWORDS]'.

        Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
        [/INST]
        """
        prompt_name = system_prompt + example_prompt + main_prompt
        # Text generation with Llama 2
        llama2_name = TextGeneration(generator_name, prompt=prompt_name, nr_docs=3, diversity=0.1, doc_length=200,
                                     tokenizer='whitespace')
        # default: nr_docs=4, diversity=None, doc_length/tokenizer=None
        # 3 representative diverse documents, every limited to word length=200

        # topic description generate
        generator_description = transformers.pipeline(
            model=model, tokenizer=tokenizer,
            task='text-generation',
            temperature=0.1,  # smaller to sharper token distribution
            max_new_tokens=200,  # label max tokens=50
            repetition_penalty=1.1
        )
        # System prompt describes information given to all conversations
        system_prompt = """<s>[INST] <<SYS>>
        You are a helpful, respectful and honest assistant for describing topics.
        <</SYS>>
        """
        # Example prompt demonstrating the output we are looking for
        example_prompt = """
        I have a topic that is described by the following keywords: 'tsai, taiwan, mccarthy, speaker, china, meeting, house, beijing, house speaker, visit'.

        In this topic, the following documents are a small but representative subset of all documents in the topic:
        - China threatened retaliation on Wednesday if U.S. House Speaker Kevin McCarthy meets with Taiwan's president during her upcoming trip through Los Angeles. President Tsai Ing-wen left Taiwan Wednesday afternoon on a tour of the island's diplomatic allies in the Americas, which she framed as a chance to demonstrate Taiwan’s commitment to democratic values on the world stage.
        - Following a visit by then-House Speaker Nancy Pelosi to Taiwan in 2022, Beijing launched missiles over the area, deployed warships across the median line of the Taiwan Strait and carried out military exercises in a simulated blockade of the island. Beijing also suspended climate talks with the U.S. and restricted military-to-military communication with the Pentagon.
        - Risking China’s anger, House Speaker Kevin McCarthy hosted Taiwan President Tsai Ing-wen on Wednesday as a “great friend of America” in a fraught show of U.S. support at a rare high-level, bipartisan meeting on U.S. soil. Speaking carefully to avoid unnecessarily escalating tensions with Beijing, Tsai and McCarthy steered clear of calls from hard-liners in the U.S. for a more confrontational stance toward China in defense of self-ruled Taiwan.

        Based on the information above, please give a description of this topic. Make sure you to only return the topic description and nothing more.

        [/INST] Despite China's warning, US House Speaker Kevin McCarthy received Taiwanese President Tsai Ing-wen as a "great friend of the United States" and held a high-level bipartisan meeting in the United States. This is another meeting between Taiwan and the United States after then-House Speaker Nancy Pelosi visited Taiwan and China conducted military exercises simulating a blockade of Taiwan in 2022. Tsai Ing-wen and McCarthy are cautious to avoid unnecessary escalation of tensions with China, eschewing calls from U.S. hardliners to take a more confrontational stance toward China.
        """

        # Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
        main_prompt = """
        [INST]
        I have a topic that is described by the following keywords: '[KEYWORDS]'.

        In this topic, the following documents are a small but representative subset of all documents in the topic:
        [DOCUMENTS]

        Based on the information above, please give a description of this topic. Make sure you to only return the topic description and nothing more.
        [/INST]
        """

        prompt_description = system_prompt + example_prompt + main_prompt
        # Text generation with Llama 2
        llama2_description = TextGeneration(generator_description, prompt=prompt_description, nr_docs=3, diversity=0.1,
                                            doc_length=200,
                                            tokenizer='whitespace')

        # All representation models
        representation_model = {
            "llama2-name": llama2_name,
            "llama2-description": llama2_description
        }
        topic_model.update_topics(contents, representation_model=representation_model)
        # save full topic model
        topic_model.save(save_dir, serialization="safetensors", save_ctfidf=True,
                         save_embedding_model=embedding_model)
        with open(os.path.join(save_dir, 'rep_docs.pickle'), 'wb') as handle:
            pickle.dump(topic_model.representative_docs_, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # save table topics
    print('loading topic model')
    topic_model = BERTopic.load(save_dir, embedding_model=embedding_model)
    topic_info = topic_model.get_topic_info()
    # extract name and description from list
    topic_labels = topic_model.get_topics(full=True)
    topic_info['llama2-name'] = [label[0][0].split("\n")[0].strip() for label in topic_labels["llama2-name"].values()]
    topic_info['llama2-description'] = [label[0][0].strip() for label in topic_labels["llama2-description"].values()]
    for col in ['Representation', 'KeyBERT', 'MMR', 'POS']:
        topic_info[col] = topic_info[col].apply(lambda x: json.dumps(x))
    '''
    topic_keywords = []
    for ind in topic_info.index:
        tmp_set = set()
        for col in ['Representation', 'KeyBERT', 'MMR', 'POS']:
            tmp = topic_info[col][ind]
            tmp_set = tmp_set | set(tmp)
        topic_keywords.append(json.dumps(list(tmp_set)))
    topic_info['keywords'] = topic_keywords
    '''
    # rename columns
    topic_info.rename(columns={"Count": "count", "llama2-name": "name", "llama2-description": "summary"}, inplace=True)
    # drop columns
    # columns_to_remove = ['Representation', 'KeyBERT', 'MMR', 'POS', 'Representative_Docs']
    columns_to_remove = ['Representative_Docs']
    topic_info = topic_info.drop(columns=columns_to_remove)
    topic_info.to_csv(os.path.join(TABLE_DATA_DIR, f'topic_table_full_{index}.csv'), index=False)
    print("Generate finish!")
    print(f"save path: {os.path.join(TABLE_DATA_DIR, f'topic_table_full_{index}.csv')}")
    # table: topic (Topic, Count, Name, keywords, name, summary)
    # save table news (same)
    '''
    document_info = topic_model.get_document_info(contents)
    document_info = document_info.drop(
        columns=columns_to_remove + ['llama2-name', 'llama2-description', 'Top_n_words'])
    document_info = document_info.sort_values(by=['Topic', 'Probability'], ascending=[True, False])
    document_info = document_info.drop(columns=['Document'])
    document_info.to_csv(os.path.join(RES_DIR, GRAIN, f'document_table_full_{year}.csv'), index=True)
    # table: news (index, Topic,Name,Probability,Representative_document)
    '''
    # return topic_model

    # del topic_model, model, embedding_model
    del topic_model, embedding_model
    gc.collect()
    torch.cuda.empty_cache()


def extract_topic(topk, path):
    '''
    extract topic and document number from topic table topic_table_full_2023.csv
    :param topk topics
    :return: topic_dict
    '''
    topic_dict = defaultdict(list)
    index = path.split('/')[-1].split('.')[0]
    
    # path = os.path.join(TABLE_DATA_DIR, f'topic_table_test.csv')
    npath = os.path.join(TABLE_DATA_DIR, f'topic_table_full_{index}.csv')
    # path = os.path.join(TABLE_DATA_DIR, f'topic_table_full_{year}.csv'))

    datas = pd.read_csv(npath)
    
    
    keys = ['name', 'count', 'summary']
    rec_keys = ['Topic', 'Count', 'Summary']

    
    for key, newkey in zip(keys, rec_keys):
        for i in range(topk):
            topic_dict[newkey].append(datas[key][i])

    tmp = pd.DataFrame(columns=rec_keys)
    for ind in rec_keys:
        tmp[ind] = topic_dict[ind]
    topic_save_path = os.path.join(save_path, f'topic_table_test.csv')
    tmp.to_csv(topic_save_path)
    # 将topic_dict内容存储到user目录下面

    topic_dict['path'] = topic_save_path

    return topic_dict


def topic_analysis(path):

    stats = news_topic_classify(path)

    generate_topic_name_description(path)
    # stats, _ = clean_file_read(year)
    print(stats)

    topic_dict = extract_topic(topk=10, path=path)
    print(topic_dict)

    return stats, topic_dict
    
# if __name__== "__main__" :
#     year = 2023
#     topic_analysis(year)
    # stats = news_topic_classify(year)
    # print(stats)

#     generate_topic_name_description(year)
#     # stats, _ = clean_file_read(year)
    

#     topic_dict = extract_topic(topk=10, year=year)
#     print(topic_dict)
