from langchain_community.document_loaders import JSONLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
import copy
from tqdm import tqdm


# 文件元数据修改
def metadata_func(record: dict, metadata: dict) -> dict:

    metadata["title"] = record.get("title")
    metadata["date"] = record.get("date")
    # 可以更好地修改metadata
    if "source" in metadata:
        source = metadata["source"].split("/")
        source = source[source.index("tw_news"):]
        metadata["source"] = "/".join(source)

    return metadata

def get_JSON_data(_file_path: str,
                  _jq_schema: str = '.',
                  _content_key: str = 'content',
                  _json_lines: bool = True,
                  _metadata_func: callable = metadata_func,
                  **kwargs):
    # 加载文件
    '/home/zhuoran/Workspace/Task/RAG/TW/simple_test.jsonl'
    # loader = JSONLoader(
    #     file_path=_file_path,
    #     jq_schema=_jq_schema,
    #     content_key=_content_key,
    #     json_lines=_json_lines,
    #     metadata_func=_metadata_func)
    loader = JSONLoader(
        file_path=_file_path,
        jq_schema=_jq_schema,
        content_key=_content_key,
        json_lines=_json_lines,
        metadata_func=_metadata_func)
    data = loader.load()
    return data

def load_data(file_path: str,**kwargs):
    # 判断文件类型
    if file_path.endswith('.jsonl') or file_path.endswith('.json'):
        data = get_JSON_data(file_path,**kwargs)
    # print(kwargs)
    # print(kwargs.get('split',False))
    if kwargs.get('split',False):
        data = split_data(data,embedding=kwargs.get('embedding'),threshold_type = kwargs.get('threshold_type'))
    return data

# def split_data(data):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=0)

#     all_data = []
#     for item in data:
#         content = item.page_content
#         docs = text_splitter.create_documents([content])
#         character_num = 0
#         for doc in docs:
#             temp = copy.deepcopy(item.metadata)
#             temp['begin_character'] = character_num
#             doc.metadata = temp
#             character_num += len(doc.page_content)
#         all_data += docs
#     return all_data

def split_data(data,embedding,threshold_type = "standard_deviation"):
    print("----Split data!")
    text_splitter = SemanticChunker(embedding,breakpoint_threshold_type=threshold_type)

    all_data = []
    for item in tqdm(data, desc="Splitting data"):
        content = item.page_content
        docs = text_splitter.create_documents([content])
        character_num = 0
        for doc in docs:
            temp = copy.deepcopy(item.metadata)
            temp['begin_character'] = character_num
            doc.metadata = temp
            character_num += len(doc.page_content)
        all_data += docs
    return all_data