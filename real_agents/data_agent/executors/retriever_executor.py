from typing import Any, Dict, List, Literal, Optional, Union

from langchain.base_language import BaseLanguageModel
from openai import embeddings

from real_agents.adapters.data_model import DatabaseDataModel, TableDataModel, ImageDataModel
from real_agents.adapters.memory import ReadOnlySharedStringMemory
from real_agents.adapters.data_model.doc_loader import load_data
# from real_agents.adapters.schema import SQLDatabase
# from real_agents.data_agent.python.base import PythonChain
# from real_agents.data_agent.sql.base import SQLDatabaseChain
# from real_agents.adapters.models import ChatOpenAI, ChatAnthropic, AzureChatOpenAI

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever

# Prompt
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain.prompts.prompt import PromptTemplate
# Vectorstore
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import utils as chromautils
import os
import time

# Embedding
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
import time
from functools import wraps
from operator import itemgetter

# Chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from real_agents.adapters.agent_helpers.stuff_combine_documents import  create_stuff_documents_chain
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f"func:{f.__name__} took: {te-ts} sec")
        return result
    return wrap
class KnowledgeRetriever:
    """Code Generation Executor.

    Example:
        .. code-block:: python

                from real_agents.adapters.executors import CodeGenerationExecutor
                executor = CodeGenerationExecutor(programming_language="sql")
                executor.run(
                    user_intent="What is the name of the first employee?",
                    grounding_source=SQLDatabase.from_uri(...)
                )

    """

    def __init__(
        self,
        Knowledge_Base: Any,
        retrieval_type: Literal["ensemble", "self_query", "bm25", "multi_query"],
        usage: Union[None, str] = None,
    ) -> None:
        """Initialize the executor.

        Args:
            programming_language: Programming language to generate.
            example_selector: Example selector to select few-shot in-context exemplars.
        """
        self._retrieval_type = retrieval_type
        self._knowledgebase = Knowledge_Base
        self._usage = usage

    @property
    def retrieval_type(self) -> str:
        """Get programming language."""
        return self._retrieval_type

    @timing
    def get_retriever(self,retriever_type:str, **kwargs):
        def get_self_retriever(llm, vectorstore, document_content_description, metadata_field_info):
            # SelfQueryRetriever
            metadata_field_info = [
                AttributeInfo(
                    name="title",
                    description="The title of the news",
                    type="string",
                ),
                AttributeInfo(
                    name="author",
                    description="Name of the news' author",
                    type="string",
                ),
                AttributeInfo(
                    name="date",
                    description="The date when the news written",
                    type="string",
                ),
            ]
            document_content_description = "News about somebody"
            self_retriever = SelfQueryRetriever.from_llm(
                llm,
                vectorstore,
                document_content_description,
                metadata_field_info,
            )
            return self_retriever

        def get_bm25_retriever(docs,bm25_k):
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = bm25_k
            return bm25_retriever

        def get_multiquery_retriever(vectordb,llm = None, **kwargs):
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0,api_key="sk-fQ7kzlsGQ8J4jjyj1MsvMmUkAkXflD5TEwgcG4KlJGrkg5Tn",
                        base_url="https://api.chatanywhere.tech/v1")
            return MultiQueryRetriever.from_llm(
            retriever = vectordb.as_retriever(search_type="similarity_score_threshold",search_kwargs={"score_threshold": 0.2}),llm=llm,include_original=True
        )

        if  retriever_type == "self":
            vectorstore = kwargs.get('vectorstore')
            return get_self_retriever(vectorstore, **kwargs)
        if  retriever_type == "multi_query":
            vectorstore = kwargs.get('vectorstore')
            #llm = kwargs.get('llm')
            return get_multiquery_retriever(vectorstore,**kwargs)
        if  retriever_type == "bm25":
            docs = kwargs.get('docs')
            bm25_k = kwargs.get('search_kwargs').get('bm25_k')
            return get_bm25_retriever(docs,bm25_k)
        if  retriever_type == "similarity":
            vectorstore = kwargs.get('vectorstore')
            retriever_k = kwargs.get('search_kwargs').get('similarity_k')
            return vectorstore.as_retriever(search_type="similarity_score_threshold",search_kwargs={"k": retriever_k, "score_threshold": 0.2})
    
    def get_ensemble_retriever(self,retrievers_name,weights,**kwargs):
            Retrievers = []
            Weights = weights
            for retriever_name in retrievers_name:
                retriever = self.get_retriever(retriever_type=retriever_name,**kwargs)
                Retrievers.append(retriever)
            return EnsembleRetriever(retrievers = Retrievers,weights=Weights)
    def get_prompt_template(self):
            RETRIEVAL_QA_PROMPT = """Based on a series of documents as follow, give detailed answers with appropriate references from background knowledge documents. If the user asks a question that is not directly found in the context below, try to summarise the context and answer the question.
            GDo not output irrelevant information and redundant answer.

            <context>
            {context}
            </context>

            <User question>
            {input}
            """
            retrieval_qa_prompt = ChatPromptTemplate.from_template(RETRIEVAL_QA_PROMPT)
            return retrieval_qa_prompt

    @timing
    def run(
        self,
        user_intent: str,
        llm: BaseLanguageModel = None,
        grounding_source: Optional[Union[List[TableDataModel], DatabaseDataModel, ImageDataModel]] = None,
        user_id: str = None,
        chat_id: str = None,
        return_intermediate_steps: bool = True,
        return_direct: bool = True,
        verbose: bool = True,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Run the executor.

        Args:
            user_intent: User intent to execute.
            grounding_source: Grounding source to execute the program on. should be {file_name: data}
            llm: Language model to use.
            return_intermediate_steps: Whether to return the intermediate steps, e.g., the program.
            return_direct: Whether to return the result of program execution directly.
            verbose: Whether to print the logging.

        Returns:
            Result dictionary of code generation
        """
        if self.retrieval_type == "ensemble":
            if self._usage is None:
                my_vectorstore = self._knowledgebase._vectorstore
                data = self._knowledgebase._data
                retriever = self.get_ensemble_retriever(retrievers_name=['similarity','multi_query'],weights=[0.5,0.5],search_kwargs={'similarity_k':1000},vectorstore=my_vectorstore,docs=data)
                prompt = self.get_prompt_template()
                doc_combine = create_stuff_documents_chain(llm = llm,prompt=prompt,document_key=['context'])
                #_input = {"input": user_intent}
                retrieved_documents = {
                    'input': itemgetter("input"),
                    'context':itemgetter('input')|retriever,
                    'inter_context':itemgetter("input")
                }
                final_inputs = {
                    "context": doc_combine | itemgetter('context'),
                    "input": itemgetter("input"),
                }
                answer = {
                    "answer": final_inputs | prompt | llm | StrOutputParser(),
                    "extra_docs": itemgetter("context"),
                }
                chain = (RunnablePassthrough() | retrieved_documents | answer)
                result = chain.invoke({'input':user_intent})
            else:
                raise ValueError(f"Usage {self._usage} not supported yet.")
        else:
            raise ValueError(f"Retrieval type {self._retrieval_type} not supported.")
        return result

class KnowledgeBase:
    @timing
    def __init__(
        self,
        store_type: Literal["chroma"],
        embedding_model: str = None,
        data_path: str = None,
        Persist_directory: str = None
    ) -> None:
        """Initialize the executor.
        """
        self._embedding = self.get_embedding_model(embedding_model)
        self._data = None
        if not os.path.isdir(Persist_directory):
            self._data = load_data(data_path)
        self._store_type = store_type
        self._persist_directory= Persist_directory
        self._vectorstore = self.get_vectorstore(use_embedding=self._embedding,data=self._data,store_type="chroma",persist_directory=self._persist_directory)
        
    @timing
    def get_embedding_model(self,Modal_Name,**kwargs):
        #TODO 这里可以把model_kwargs写全并设置好Default，后续可以在config中配置
        model_kwargs = {'device': 'cuda:0'}
        #local_embeddings = HuggingFaceEmbeddings(model_name=Modal_Name,model_kwargs=model_kwargs)
        local_embeddings = HuggingFaceEmbeddings(model_name=Modal_Name)
        return local_embeddings
    @timing
    def get_vectorstore(self,use_embedding, data=None, store_type="chroma",**kwargs):
        if store_type == "chroma":
            return self.get_chroma_vectorstore(use_embedding,data,**kwargs)
    @timing
    def get_chroma_vectorstore(self,use_embedding,data = None,**kwargs):
        Persist_directory = kwargs.get("persist_directory")
        if not os.path.isdir(Persist_directory):
            print('----Generate Chroma Vectorstore')
            start_time = time.time()
            docs = chromautils.filter_complex_metadata(data)
            Persist_directory = kwargs.get("persist_directory")
            vectorstore = Chroma.from_documents(
                docs, embedding=use_embedding,persist_directory=Persist_directory
            )
            end_time = time.time()
            print(f"运行时间: {end_time - start_time} 秒")
        else:
            print("----Load Chroma Vectorstore")
            vectorstore = Chroma(persist_directory=Persist_directory, embedding_function=use_embedding)

        # example_db.get(where={"source": "some_other_source"})
        return vectorstore