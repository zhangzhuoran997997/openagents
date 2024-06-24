"""Chain that combines documents by stuffing into context."""
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import BasePromptTemplate, format_document,PromptTemplate
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.runnables import (
    Runnable, 
    RunnablePassthrough,
    RunnableBinding,
    RunnableBranch,
    RunnableWithFallbacks,)
from langchain_core.runnables.configurable import DynamicRunnable
from langchain_core.language_models import BaseLanguageModel
from langchain.chains.combine_documents.base import (
    DEFAULT_DOCUMENT_PROMPT,
    DEFAULT_DOCUMENT_SEPARATOR,
    DOCUMENTS_KEY,
    BaseCombineDocumentsChain,
    _validate_prompt,
)
TIME_DOCUMENT_PROMPT = PromptTemplate.from_template(
     '''
     Document:
     Date:{date}
     Content:{page_content}'''
     )

def create_stuff_documents_chain(
    llm: LanguageModelLike = None,
    prompt: BasePromptTemplate = None,
    *,
    output_parser: Optional[BaseOutputParser] = None,
    document_prompt: Optional[BasePromptTemplate] = TIME_DOCUMENT_PROMPT,
    document_key: List[str] = [DOCUMENTS_KEY],
    document_separator: str = DEFAULT_DOCUMENT_SEPARATOR,
) -> Runnable[Dict[str, Any], Any]:
    """Create a chain for passing a list of Documents to a model.

    Args:
        llm: Language model.
        prompt: Prompt template. Must contain input variable "context", which will be
            used for passing in the formatted documents.
        output_parser: Output parser. Defaults to StrOutputParser.
        document_prompt: Prompt used for formatting each document into a string. Input
            variables can be "page_content" or any metadata keys that are in all
            documents. "page_content" will automatically retrieve the
            `Document.page_content`, and all other inputs variables will be
            automatically retrieved from the `Document.metadata` dictionary. Default to
            a prompt that only contains `Document.page_content`.
        document_separator: String separator to use between formatted document strings.

    Returns:
        An LCEL Runnable. The input is a dictionary that must have a "context" key that
        maps to a List[Document], and any other input variables expected in the prompt.
        The Runnable return type depends on output_parser used.

    Example:
        .. code-block:: python

            # pip install -U langchain langchain-community

            from langchain_community.chat_models import ChatOpenAI
            from langchain_core.documents import Document
            from langchain_core.prompts import ChatPromptTemplate
            from langchain.chains.combine_documents import create_stuff_documents_chain

            prompt = ChatPromptTemplate.from_messages(
                [("system", "What are everyone's favorite colors:\\n\\n{context}")]
            )
            llm = ChatOpenAI(model_name="gpt-3.5-turbo")
            chain = create_stuff_documents_chain(llm, prompt)

            docs = [
                Document(page_content="Jesse loves red but not yellow"),
                Document(page_content = "Jamal loves green but not as much as he loves orange")
            ]

            chain.invoke({"context": docs})
    """  # noqa: E501

    _validate_prompt(prompt)
    _document_prompt = document_prompt or DEFAULT_DOCUMENT_PROMPT
    _output_parser = output_parser or StrOutputParser()

    # merge all docs
    
    #TODO 这个的地方可以将新闻的发布时间等 metadata 解析出来加入到上下文中去
    #! 需要使用_document_prompt来限制文本的格式
    #? 这里相当于是将input中的Document[page_content,metadata]巴拉巴拉的东西给变成string
    old_document_key = DOCUMENTS_KEY
    def format_docs(inputs: dict) -> str:
        all_docs = []
        if len(inputs['context']) > 0:
            print('1111111111111111')
            inputs['inter_context'] = []
            #inputs['inter_context'][0].page_content = ' '
            #inputs['inter_context'][0].metadata']['date'] = ' '
        for single_document_key in document_key :
            all_docs += inputs[single_document_key]
        all_docs = _reduce_tokens_below_limit(llm,all_docs,max_tokens_limit=3000)
        return document_separator.join(
            format_document(doc, _document_prompt) for doc in all_docs
        )

    #? 这里通过RunnablePassthrough.assign来将format_docs的结果赋值给context当作代替
    #? 如果换个document_key,assign就会多加字典的一个对
    return (
        RunnablePassthrough.assign(**{old_document_key: format_docs}).with_config(
            run_name="format_inputs"
        )
    ).with_config(run_name="stuff_documents_chain")

def _get_num_tokens(llm:Runnable,text: str) -> int:
        return _get_language_model(llm).get_num_tokens(text)

def _reduce_tokens_below_limit(llm ,docs: List[Document],max_tokens_limit=3500) -> List[Document]:
        num_docs = len(docs)

        if max_tokens_limit:
            tokens = [
                _get_num_tokens(llm,doc.page_content)
                for doc in docs
            ]
            token_count = sum(tokens[:num_docs])
            while token_count > max_tokens_limit:
                num_docs -= 1
                token_count -= tokens[num_docs]

        return docs[:num_docs]

def _get_language_model(llm_like: Runnable) -> BaseLanguageModel:
    if isinstance(llm_like, BaseLanguageModel):
        return llm_like
    elif isinstance(llm_like, RunnableBinding):
        return _get_language_model(llm_like.bound)
    elif isinstance(llm_like, RunnableWithFallbacks):
        return _get_language_model(llm_like.runnable)
    elif isinstance(llm_like, (RunnableBranch, DynamicRunnable)):
        return _get_language_model(llm_like.default)
    else:
        raise ValueError(
            f"Unable to extract BaseLanguageModel from llm_like object of type "
            f"{type(llm_like)}"
        )
    #TODO 这里的代码可以用于做参考，加入summary等功能
    # return (
    #     RunnablePassthrough.assign(**{document_key: format_docs}).with_config(
    #         run_name="format_inputs"
    #     )
    #     | prompt
    #     | llm
    #     | _output_parser
    # ).with_config(run_name="stuff_documents_chain")
