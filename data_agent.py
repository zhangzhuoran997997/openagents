# flake8: noqa
from langchain import PromptTemplate

# Text-to-sql prompt
_DEFAULT_TEMPLATE = """Here are chat histories you may refer to, maybe empty.
{chat_history}

Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, remember to wrap the table names in double quotes.
Use the following format:
Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"
Only use the tables listed below.
{table_info}
Question: {question}"""
PROMPT = PromptTemplate(
    input_variables=["chat_history", "question", "table_info", "dialect"],
    template=_DEFAULT_TEMPLATE,
)


# Few-shot text-to-sql prompt
FEW_SHOT_PREFIX = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Use the following format:
Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"
Here are some examples you can follow:"""
EXAMPLE_PROMPT_TEMPLATE = """{table_info}\nQuestion: {question}\nSQLQuery: {query}"""
EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["table_info", "question", "query"],
    template=EXAMPLE_PROMPT_TEMPLATE,
)
FEW_SHOT_SUFFIX = """
User the tables listed below.
{table_info}
Question: {question}"""
FEW_SHOT_INPUT_VARIABLES = ["question", "table_info", "dialect", "top_k"]
"""Chain for interacting with SQL Database."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from langchain_core.pydantic_v1 import BaseModel, Extra, Field
from loguru import logger

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain import BasePromptTemplate, FewShotPromptTemplate

from real_agents.data_agent.evaluation.sql_evaluator import SQLEvaluator
from real_agents.adapters.schema import SQLDatabase
from real_agents.adapters.memory import ReadOnlySharedStringMemory
from real_agents.data_agent.sql.prompt import (
    EXAMPLE_PROMPT,
    FEW_SHOT_INPUT_VARIABLES,
    FEW_SHOT_PREFIX,
    FEW_SHOT_SUFFIX,
    PROMPT,
)
from real_agents.adapters.llm import LLMChain
from real_agents.adapters.data_model import MessageDataModel


class SQLDatabaseChain(Chain, BaseModel):
    """Chain for interacting with SQL Database"""

    llm: BaseLanguageModel
    """LLM wrapper to use."""
    database: SQLDatabase = Field(exclude=True)
    """SQL Database to connect to."""
    example_selector: Any = None
    """Example selector to select few-shot in-context exemplars."""
    memory: Optional[ReadOnlySharedStringMemory] = None
    """Shared memory."""
    prompt: BasePromptTemplate = PROMPT
    """Prompt to use to translate natural language to SQL."""
    input_key: str = "user_intent"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    return_intermediate_steps: bool = False
    """Whether or not to return the intermediate steps along with the final answer."""
    return_direct: bool = False
    """Whether or not to return the result of querying the SQL table directly."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.
        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.
        :meta private:
        """
        if not self.return_intermediate_steps:
            return [self.output_key]
        else:
            # return [self.output_key, "intermediate_steps", "binder_steps"]
            return [self.output_key, "intermediate_steps"]

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        logger.bind(msg_head="SQLChain inputs").trace(inputs)

        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        if self.example_selector is not None:
            self.prompt = FewShotPromptTemplate(
                example_selector=self.example_selector,
                example_prompt=EXAMPLE_PROMPT,
                prefix=FEW_SHOT_PREFIX,
                suffix=FEW_SHOT_SUFFIX,
                input_variables=FEW_SHOT_INPUT_VARIABLES,
            )
        llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
        input_text = f"{inputs[self.input_key]} \nSQLQuery:"
        _run_manager.on_text(input_text, verbose=self.verbose)

        # If not present, then defaults to None which is all tables.
        table_names_to_use = inputs.get("table_names_to_use")
        table_info = self.database.get_table_info(table_names=table_names_to_use)
        llm_inputs = {
            "question": input_text,
            "dialect": self.database.dialect,
            "table_info": table_info,
            "chat_history": "",
            "stop": ["\nSQLResult:"],
        }

        # Load memory into chat history
        if self.memory is not None:
            llm_inputs["chat_history"] = self.memory.load_memory_variables({})["chat_history"]
            llm_inputs["chat_history"] = MessageDataModel.extract_code_for_sql_tool(llm_inputs["chat_history"])
        sql_cmd = llm_chain.predict(**llm_inputs)
        # TODO: Move this post-processing to a post-process function
        sql_cmd = sql_cmd.replace("\n", " ")
        if sql_cmd.endswith('"') and sql_cmd.startswith('"'):
            sql_cmd = sql_cmd.strip('"')
        if sql_cmd.endswith("'") and sql_cmd.startswith("'"):
            sql_cmd = sql_cmd.strip("'")

        logger.bind(msg_head="SQLChain generate program").trace(sql_cmd)

        # Call SQL/binder evaluator to execute the SQL command
        sql_evaluator = SQLEvaluator()
        result = sql_evaluator.run(sql_cmd, self.database)

        logger.bind(msg_head="SQLChain execution result").trace(result)

        # If return direct, we just set the final result equal to the sql query
        if self.return_direct:
            final_result = result
        else:
            input_text += f"{sql_cmd}\nSQLResult: {result}\nAnswer:"
            llm_inputs["input"] = input_text
            final_result = llm_chain.predict(**llm_inputs)
            _run_manager.on_text(final_result, color="green", verbose=self.verbose)
        chain_result: Dict[str, Any] = {self.output_key: final_result}
        if self.return_intermediate_steps:
            chain_result["intermediate_steps"] = sql_cmd
        return chain_result

    @property
    def _chain_type(self) -> str:
        return "sql_database_chain"
# flake8: noqa

PREFIX = """You are XLang Agent , a friendly and intuitive interface developed by the XLang Team to guide human through every stage of human data lifecycle. Whether human are loading, processing, or interpreting data, XLang Agent is always at human's fingertips through our interactive chat system.

Empowered by an array of innovative tools that can generate and execute code, XLang Agent delivers robust, reliable answers to human queries. Whenever possible, You employs these tools to give human rich insights, like dynamic code generation & execution and compelling visualizations. And You will always proactively and correctly using all tools to help with human.

Get ready for a seamless and insightful journey with XLang Agent, the personal assistant for all things data!

TOOLS
------
You have direct access to following tools. 
"""


FORMAT_INSTRUCTIONS = """RESPONSE FORMAT INSTRUCTIONS
----------------------------

When you use tools or generate final answer, please output a response in one of two formats:
**Option 1: Explain and Use Tool**
If the response involves using a tool, you can start with a natural language explanation[Optional], plus exactly one tool calling[MUST]. But **make sure no any words & answer appended after tool calling json**. The tool calling format should be a markdown code snippet with the following JSON schema:

```json
{{{{
    "action": string wrapped with \"\", // The action to take. Must be one in the list [{tool_names}]
    "action_input": string wrapped with \"\" // Natural language query to be input to the action tool.
}}}}
```

[**Restriction**] Please note that ONLY one tool should be used per round, and you MUST stop generating right after tool calling and make sure no any text appended after tool calling markdown code snippet. Save your words.

NEVER EVER EVER make up a tool not in [{tool_names}]
NEVER EVER EVER generate code as action input when using tool. Just input natural language by using/paraphrasing human query.

**Option #2:**
Use this if you want to respond directly to the human.
If you want to respond directly to the human without using a tool, provide a plain natural language response. However, if you initially generated a natural language response and then decide to use a tool, make sure to include the tool action and input after the initial response.

Note if the human asks for malicious code, and just respond directly to deny the request and give your professional reason. Don't use any tool. 
The malicious code includes but not limited to: 
1. Endless operations and excessive waiting  (e.g., while True, long print, input())
2. System crash (e.g., any risky system command)
3. Data loss (e.g., list or delete files)
4. Leak sensitive information (e.g., os.getenv())
5. Establish network connections (e.g., requests.get())
6. Cause any other security issues

[Mandatory to notice] It is imperative and a must to utilize tools whenever the human's query tasks that implies using tools, such as searching online, generating code, executing code, or any other complex functionalities. You must try to use tools to solve human queries in these cases.

Begin.
"""

SUFFIX = """{input}"""


TEMPLATE_TOOL_RESPONSE = """TOOL RESPONSE:
---------------------
{observation}

THOUGHT
--------------------

Okay, So what's next? Let's assess if the tool response is enough to answer the human's initial query. Please follow these instructions:

1. Evaluate Tool Response [Mandatory]: Carefully evaluate the tool's response and determine if it sufficiently addresses the human's query. Consider the content and implications of the tool's response.

2. Consider Additional Tool Use [Optional 2 or 3]: If the tool response does not fully address the query or if an error occurred during execution, you may proceed with additional tool usage. However, exercise caution and limit the number of iterations to a maximum of three. You can start with a natural language explanation[Optional], plus exactly one tool calling[MUST]. But **make sure no any words & answer appended after tool calling json**. Follow this format for additional tool usage:

```json
{{{{
    "action": string wrapped with \"\", // The action to take. Must be one of [{tool_names}]
    "action_input": string wrapped with \"\" // Natural language query to be input to the action tool
}}}}
```
[**Restriction**] Please note that only one tool should be used per round, and you MUST stop generating right after tool calling and make sure no any text appended after tool calling markdown code snippet.


3. Deliver Comprehensive Answer [Optional 2 or 3]: If the tool response sufficiently addresses the query, deliver a comprehensive answer to the human. Focus solely on the content and implications of the tool's response. MUST NOT include explanations of the tool's functions.

3.1. Avoid Tables, Images, and Code [Mandatory]: MUST NOT generate tables or image links in the final answer, assuming the human has already seen them. Avoid generating code in the final answer as well. Instead, paraphrase the code into a human query if you need to explain it.

Note. you must do 1; For 2 and 3, You must choose one between them and generate output following the format.

Begin.
"""

# models like anthropic claude-v1 or claude-2 can only return valid completion with human message as the last message, so we append the fake AI message at the end.
fake_continue_prompt = {
    "claude-2": "you can start to think and respond to me using the above formats. No Apology. Just respond with format in Option 2(use tool) or Option 3(direct text response), no other words.\n\nBegin.",
    "claude-v1": "you can start to think and respond to me using the above formats. No Apology. Just respond with format in Option 2(use tool) or Option 3(direct text response), no other words.\n\nBegin.",
}
import os
from typing import Any, List, Optional, Tuple, Dict
from langchain_core.pydantic_v1 import BaseModel
import requests
import time
import ast

import pandas as pd
from io import StringIO
import redis
from loguru import logger

from IPython.core.interactiveshell import InteractiveShell
from IPython.core.getipython import get_ipython
from IPython.utils.capture import capture_output


# subscribed channels
SUBMIT_EVENT = "job_submitted"
RUNNING_EVENT = "job_started"
COMPLETE_EVENT = "job_completed"
# Error render prefix
ERROR_PREFIX = "[ERROR]: "


def check_danger_code(code):
    code_line = []
    for line in code.split("\n"):
        if not line.startswith("%"):
            code_line.append(line)
    code = "\n".join(code_line)

    def check_imports(code):
        ast_failed = False
        try:
            tree = ast.parse(code)
        except Exception as e:
            ast_failed = str(e)
            return ast_failed
        return ast_failed

    ast_failed = check_imports(code)
    return True, ast_failed, []


class DisplayData(BaseModel):
    """Both display_data and execute_result messages use this format."""

    data: Optional[dict] = None
    metadata: Optional[dict] = None

    @classmethod
    def from_tuple(cls, formatted: Tuple[dict, dict]):
        return cls(data=formatted[0], metadata=formatted[1])

    def to_dict(self) -> Dict:
        return {
            "data": self.data,
            "metadata": self.metadata,
        }


class PythonEvaluator:
    """
    Util class for Python code evaluation.
    """

    name = "Python Evaluator"
    base_url = "http://{0}:8100".format(os.getenv("CODE_INTER_SERVER"))
    r: redis.Redis = redis.Redis(host=os.getenv("REDIS_SERVER"), port=6379, decode_responses=True)

    def __init__(self, code_execution_mode: str = "local", jupyter_kernel_pool: Optional[Any] = None):
        self.code_execution_mode = code_execution_mode
        self.jupyter_kernel_pool = jupyter_kernel_pool

    @staticmethod
    def parse_command(program: str) -> List[str]:
        """patchify the code"""
        program_lines = program.strip().split("\n")
        return program_lines

    def run_program_local(self, program: str, user_id: Optional[str] = "u" * 24):
        """Run python program on the local machine using Ipython shell."""
        is_safe, ast_failed, danger_pcks = check_danger_code(program)
        if ast_failed != False:
            return {
                "success": False,
                "error_message": f"{ERROR_PREFIX}Error Code Parsing, please check code grammar!\n{ast_failed}",
            }
        try:
            # Run code using local ipython shell
            # Change working dir to data directory to load from pretty path
            #   Note! This is not thread safe, only for local use
            os.chdir(os.path.join("backend/data/", user_id))

            shell = InteractiveShell.instance()
            shell.enable_gui = lambda x: False
            with capture_output() as captured:
                ip = get_ipython()
                code = "%matplotlib inline\n" + program  # magic command to display matplotlib plots
                result = ip.run_cell(code)

            # Change working dir to project root
            os.chdir("../../../")

            if result.success:
                return {
                    "success": True,
                    "result": result.result,
                    "stdout": str(captured.stdout),
                    "stderr": str(captured.stderr),
                    "outputs": captured.outputs,
                }
            elif result.error_in_exec is not None:
                return {
                    "success": False,
                    "error_message": f"{ERROR_PREFIX}{str(result.error_in_exec)}",
                    "outputs": captured.outputs,
                }
            else:
                # error_before_exec
                return {
                    "success": False,
                    "error_message": f"{ERROR_PREFIX}{str(result.error_before_exec)}",
                    "outputs": captured.outputs,
                }
        except Exception as e:
            logger.bind(user_id=user_id, msg_head="Python evaluator running error").trace(e)
            import traceback

            traceback.print_exc()
            return {
                "success": False,
                "error_message": f"{ERROR_PREFIX}{str(e)}",
            }

    def _apply_for_kernel(self, kernel_id: Optional[str], user_id: str, chat_id: str):
        """Apply for a kernel in docker to run program."""
        if kernel_id is not None:
            # If kernel id is provided, use it directly
            cur_kid = kernel_id
        else:
            # If kernel id is not provided, apply for a new kernel
            kernel_info = self.jupyter_kernel_pool.get_pool_info_with_id(user_id, chat_id, None)
            cur_kid = kernel_info["kid"] if kernel_info is not None else None
            user_exists = requests.get(f"{self.base_url}/user/status/{user_id}").json()["exists"]

            logger.bind(user_id=user_id, chat_id=chat_id, msg_head="user exists").trace(user_exists)

            if not user_exists:
                response = requests.post(f"{self.base_url}/user/create", json={"username": user_id}).json()

                logger.bind(user_id=user_id, chat_id=chat_id, msg_head="user create").trace(response)

            response = requests.get(f"{self.base_url}/kernel/list/{user_id}").json()
            existing_kernel_list = response["list"]

            logger.bind(user_id=user_id, chat_id=chat_id, msg_head="kernel list").trace(response)

            if cur_kid not in existing_kernel_list:
                response = requests.post(f"{self.base_url}/kernel/create", json={"username": user_id}).json()
                if response["code"] != 0 and response["msg"] == "Too many kernels":
                    # kill oldest kernel
                    oldest_kernel_id = existing_kernel_list[0]
                    response = requests.post(
                        f"{self.base_url}/kernel/stop", json={"username": user_id, "kid": oldest_kernel_id}
                    ).json()

                    logger.bind(user_id=user_id, chat_id=chat_id, msg_head="kill oldest kernel").trace(response)

                    response = requests.post(f"{self.base_url}/kernel/create", json={"username": user_id}).json()
                cur_kid = response["id"]

                logger.bind(user_id=user_id, chat_id=chat_id, msg_head="create kernel id").trace(cur_kid)

                self.jupyter_kernel_pool.set_pool_info_with_id(
                    user_id, chat_id, {"kid": cur_kid, "ktime": time.time()}
                )

        logger.bind(user_id=user_id, chat_id=chat_id, msg_head="current kernel id").trace(cur_kid)

        return cur_kid

    def run_program_docker(
        self,
        program: str,
        kernel_id: Optional[str] = None,
        user_id: Optional[str] = "u" * 24,
        chat_id: Optional[str] = "c" * 24,
    ):
        """Run python program on the docker container(jupyter client)."""
        is_safe, ast_failed, danger_pcks = check_danger_code(program)
        if not is_safe:
            return {
                "success": False,
                "error_message": f"{ERROR_PREFIX}Unsafe Code Detected {str(danger_pcks)}, Execution Denied!",
            }
        elif ast_failed != False:
            return {
                "success": False,
                "error_message": f"{ERROR_PREFIX}Error Code Parsing, please check code grammar!\n{ast_failed}",
            }

        try:
            # Run code using remote docker jupyter kernel
            # Get the running id(i.e., a permission to run program) from redis running queue
            #  otherwise wait until a running_id is available
            p = self.r.pubsub()
            p.subscribe(RUNNING_EVENT)
            self.r.publish(SUBMIT_EVENT, chat_id)
            for message in p.listen():
                # the initial message for each channel is a message with an integer
                if isinstance(message["data"], int):
                    continue
                running_id = message["data"]
                if running_id == chat_id:
                    break
                time.sleep(1)
            # Get kernel id(i.e., the real jupyter kernel to run the program) to execute program
            cur_kid = self._apply_for_kernel(kernel_id, user_id, chat_id)
            # Execute program
            response = requests.post(
                f"{self.base_url}/kernel/exec", json={"username": user_id, "code": program, "kid": cur_kid}
            ).json()
            # Notify Redis that a job has been completed
            self.r.publish(COMPLETE_EVENT, chat_id)

            # Parse jupyter kernel output
            result, stdout, stderr, outputs, displays, error_message = None, "", "", None, [], None
            if response["status"] == "ok":
                output = response.get("output", None)
                if output is not None:
                    for output_dict in output:
                        if output_dict["type"] == "stream":
                            content = output_dict.get("content", None)
                            if content is not None:
                                if content["name"] == "stdout":
                                    stdout = content["text"]
                                if content["name"] == "stderr":
                                    stderr = content["text"]
                        elif output_dict["type"] == "execute_result":
                            content = output_dict.get("content", None)
                            if content is not None:
                                data = content.get("data", None)
                                if data is not None:
                                    if "text/plain" in data and "text/html" in data:
                                        try:
                                            # Try to recover a dataframe
                                            df = pd.read_csv(StringIO(data["text/plain"]))
                                            result = df
                                        except Exception as e:
                                            pass
                                    elif "text/plain" in data:
                                        result = data["text/plain"]
                                    else:
                                        # TODO: If not match any of the above, return the first value
                                        result = list(data.values())[0]
                        elif output_dict["type"] == "display_data":
                            content = output_dict.get("content", None)
                            if content is not None:
                                data = content.get("data", None)
                                metadata = content.get("metadata", None)
                                if data is not None and metadata is not None:
                                    if "image/png" in data:
                                        displays.append(DisplayData.from_tuple((data, metadata)))
                            if displays:
                                outputs = displays
            else:
                return {
                    "success": False,
                    "error_message": f"{ERROR_PREFIX}Error status returned by kernel",
                }
            # Check success status and return
            shell_msg = response["shell"]
            if shell_msg["status"] == "ok":
                return {
                    "success": True,
                    "result": result,
                    "stdout": stdout,
                    "stderr": stderr,
                    "outputs": outputs,
                }
            elif shell_msg["status"] == "error":
                error_message = f"{shell_msg['ename']}: {shell_msg['evalue']}"
                return {"success": False, "error_message": f"{ERROR_PREFIX}{error_message}", "outputs": outputs}
        except Exception as e:
            logger.bind(user_id=user_id, chat_id=chat_id, msg_head="Python evaluator running error").trace(e)
            import traceback

            traceback.print_exc()

            try:
                # Notify Redis that a job has been completed
                self.r.publish(COMPLETE_EVENT, chat_id)
            except:
                pass

            return {
                "success": False,
                "error_message": f"{ERROR_PREFIX}{str(e)}",
            }

    def run(
        self,
        program: str,
        environment: Optional[Any] = None,
        kernel_id: Optional[str] = None,
        user_id: Optional[str] = "u" * 24,
        chat_id: Optional[str] = "c" * 24,
    ) -> Any:
        """run generated code in certain environment"""

        lines_code = self.parse_command(program)
        program = "\n".join(lines_code)
        program = "%matplotlib inline\n" + program  # magic command to display matplotlib plots

        logger.bind(user_id=user_id, chat_id=chat_id, msg_head="Code execution mode").trace(self.code_execution_mode)

        if self.code_execution_mode == "local":
            return self.run_program_local(program, user_id)
        elif self.code_execution_mode == "docker":
            return self.run_program_docker(program, kernel_id, user_id, chat_id)
        else:
            raise ValueError("Invalid code execution mode")
import traceback
from typing import Any, Dict, List

from langchain_core.pydantic_v1 import root_validator

from real_agents.adapters.schema import SQLDatabase


class SQLEvaluator:
    """
    Util class for SQL code evaluation.
    """

    name = "SQL Evaluator"
    ERROR_PREFIX = "[ERROR]: "

    @root_validator(pre=True)
    def validate(cls, values: Dict) -> Any:
        """validate requirements for evaluation"""
        try:
            import sqlite3  # noqa F401 E402

            import sqlalchemy  # noqa F401 E402
        except ImportError:
            raise ValueError("This tool relies on sqlite3 and sqlalchemy, use `pip` to install these packages")
        return values

    @staticmethod
    def parse_command(program: str) -> List[str]:
        """patchify the code"""
        program_lines = program.strip().split("\n")
        return program_lines

    def run(self, program: str, environment: SQLDatabase) -> Any:
        """run generated code in certain environment"""
        try:
            output = environment.run(program)
            return {
                "success": True,
                "result": output,
            }
        except Exception as e:
            traceback.print_exc()
            error_message = str(e)
            return {"success": False, "error_message": f"{self.ERROR_PREFIX}{error_message}"}
import json
import os
import re
import shutil
import uuid
from typing import Any, Dict, List, Tuple
import requests
from bs4 import BeautifulSoup
from loguru import logger
from kaggle.api.kaggle_api_extended import KaggleApi

from langchain.base_language import BaseLanguageModel
from langchain import PromptTemplate

from real_agents.adapters.llm import LLMChain


class KaggleDataLoadingExecutor:
    KAGGLE_TEMPLATE = """

Determine whether the user input aims to (1) connect to a specific kaggle dataset that the user mentions its kaggle path
(2) search for relevant kaggle datasets given the information the user provides.

You need to output the action wrapped in <action></action>, the action space is ['connect', 'search']. You also need
to output the keywords wrapped in <keywords></keywords>. For 'search', the keywords MUST be ONE search term/word to be
searched by kaggle api. Note keywords CAN'T be too specific or contain trivial word(e.g., dataset), make sure there are various search results. For
'connect', the keywords are the kaggle dataset path.

Input: {input}

Begin."
"""

    def run(
        self,
        user_intent: str,
        llm: BaseLanguageModel,
        search_top_k: int = 4,
    ) -> Dict[str, Any]:
        logger.bind(msg_head="KaggleDataLoader inputs").trace(user_intent)
        kaggle_template = PromptTemplate(
            input_variables=["input"],
            template=self.KAGGLE_TEMPLATE,
        )
        method = LLMChain(llm=llm, prompt=kaggle_template)
        result = method.run({"input": user_intent})
        logger.bind(msg_head="LLM result").trace(result)
        kaggle_action, keywords = self._parse_output(result)
        logger.bind(msg_head="Kaggle action").trace(kaggle_action)
        logger.bind(msg_head="Kaggle keywords").trace(keywords)
        """Use export to manage the Kaggle API key for now."""
        api = KaggleApi()
        api.authenticate()
        if kaggle_action == "connect":
            kaggle_output_info = keywords
        elif kaggle_action == "search":
            kaggle_output_info = self._search_kaggle(api, keywords, search_top_k)
        else:
            # Regard the rest as "search" action now
            kaggle_action = "search"
            kaggle_output_info = self._search_kaggle(api, keywords, search_top_k)
        return {"kaggle_action": kaggle_action, "kaggle_output_info": kaggle_output_info}

    def _search_kaggle(self, api: KaggleApi, keywords: str, search_top_k: int) -> List[Dict]:
        """Search kaggle datasets given the keywords."""
        # Search for datasets
        datasets = []
        for page in range(1, 10):
            try:
                searched_datasets = api.dataset_list(search=keywords, page=page, max_size=20000, file_type="csv")

                logger.bind(msg_head="Kaggle search result").trace(searched_datasets)

                datasets.extend(searched_datasets)
                if len(datasets) >= search_top_k:
                    datasets = datasets[:search_top_k]
                    break
                if len(searched_datasets) < 20:
                    # Default page_size is 20, less than 20 means no more datasets can be searched
                    break
            except Exception:
                break

        # Get url, cover image and some meta data for each dataset
        if len(datasets) == 0:
            # No datasets found
            datasets = api.dataset_list(max_size=20000, page=1, file_type="csv")[:search_top_k]

        output_info = self._get_dataset_meta_info(api, datasets)
        return output_info

    def _get_dataset_meta_info(self, api: KaggleApi, datasets: List) -> List[Dict]:
        """Get dataset key meta-data to be shown to the user."""
        output_info = []
        for dataset in datasets:
            dataset_hash_id = str(uuid.uuid4())
            dataset_tmp_dir = os.path.join(".kaggle_meta/", dataset_hash_id)
            os.makedirs(dataset_tmp_dir, exist_ok=True)
            api.dataset_metadata(dataset.ref, path=dataset_tmp_dir)
            with open(os.path.join(dataset_tmp_dir, "dataset-metadata.json")) as f:
                dataset_metadata = json.load(f)
            shutil.rmtree(os.path.join(".kaggle_meta/", dataset_hash_id))
            dataset_url = "https://www.kaggle.com/datasets/" + dataset.ref
            # Crawling the dataset page to get the dataset image
            dataset_cover_image_url = self._crawl_dataset_cover_image(dataset_url)

            logger.bind(msg_head="Dataset cover image url").trace(dataset_cover_image_url)

            output_metadata = {
                "id": dataset_metadata["id"],
                "id_no": dataset_metadata["id_no"],
                "title": dataset_metadata["title"],
                "subtitle": dataset_metadata["subtitle"],
                "total_views": dataset_metadata["totalViews"],
                "total_votes": dataset_metadata["totalVotes"],
                "total_downloads": dataset_metadata["totalDownloads"],
                "url": dataset_url,
                "cover_image_url": dataset_cover_image_url,
            }
            output_info.append(output_metadata)
        return output_info

    def _crawl_dataset_cover_image(
        self, url: str, default_image_path="https://images.datacamp.com/image/upload/v1647430873/kaggle_logo_icon_168474_4eb653edb6.png"
    ) -> str:
        """Crawl the kaggle dataset cover image from the dataset url."""
        # Get the HTML content of the webpage
        response = requests.get(url)

        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Find the image element
        try:
            kaggle_component_element = soup.find("script", {"class": "kaggle-component"})
            match = re.search(r'"coverImageUrl":\s*"([^"]*)"', kaggle_component_element.string)
            image_url = match.group(1)
        except Exception:
            import traceback

            traceback.print_exc()
            image_url = default_image_path

        return image_url

    def _parse_output(self, content: str) -> Tuple[str, str]:
        """Parse the output of the LLM to get the kaggle action and keywords."""
        from bs4 import BeautifulSoup

        # Using 'html.parser' to parse the content
        soup = BeautifulSoup(content, "html.parser")
        # Parsing the tag and summary contents
        try:
            action = soup.find("action").text
        except Exception:
            action = ""

        try:
            keywords = soup.find("keywords").text
        except Exception:
            keywords = ""

        return action, keywords
from typing import Any, Dict, List, Literal, Optional, Union

from langchain.base_language import BaseLanguageModel

from real_agents.adapters.data_model import DatabaseDataModel, TableDataModel, ImageDataModel
from real_agents.adapters.memory import ReadOnlySharedStringMemory
from real_agents.adapters.schema import SQLDatabase
from real_agents.data_agent.python.base import PythonChain
from real_agents.data_agent.sql.base import SQLDatabaseChain


class CodeGenerationExecutor:
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
        programming_language: Literal["sql", "python"],
        usage: Union[None, str] = None,
        example_selector: Any = None,
        memory: Optional[ReadOnlySharedStringMemory] = None,
    ) -> None:
        """Initialize the executor.

        Args:
            programming_language: Programming language to generate.
            example_selector: Example selector to select few-shot in-context exemplars.
        """
        self._programming_language = programming_language
        self._usage = usage
        self._example_selector = example_selector
        self._memory = memory

    @property
    def programming_language(self) -> str:
        """Get programming language."""
        return self._programming_language

    def run(
        self,
        user_intent: str,
        llm: BaseLanguageModel,
        grounding_source: Optional[Union[List[TableDataModel], DatabaseDataModel, ImageDataModel]] = None,
        user_id: str = None,
        chat_id: str = None,
        code_execution_mode: str = "local",
        jupyter_kernel_pool: Any = None,
        return_intermediate_steps: bool = True,
        return_direct: bool = True,
        verbose: bool = True,
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

        def _concat_grounding_source() -> str:
            assert isinstance(grounding_source, list)
            table_schema = ""
            for gs in grounding_source:
                table_schema += f"{gs.get_llm_side_data()}\n"
            return table_schema

        if self._programming_language == "sql":
            db = grounding_source.raw_data
            assert isinstance(db, SQLDatabase)
            method = SQLDatabaseChain(
                llm=llm,
                database=db,
                example_selector=self._example_selector,
                memory=self._memory,
                return_direct=return_direct,
                return_intermediate_steps=return_intermediate_steps,
                verbose=verbose,
            )
            _input = {"user_intent": user_intent}
            result = method(_input)
        elif self._programming_language == "python":
            if self._usage is None:
                # General python code generation for data analysis
                method = PythonChain.from_python_prompt(
                    llm,
                    return_intermediate_steps=return_intermediate_steps,
                    verbose=True,
                    memory=self._memory,
                    user_id=user_id,
                    chat_id=chat_id,
                    code_execution_mode=code_execution_mode,
                    jupyter_kernel_pool=jupyter_kernel_pool,
                )
                # Get each source_item (table, db, files...) from the grounding_source
                _input = {"question": user_intent, "data_info": _concat_grounding_source()}
                result = method(_input)
            elif self._usage == "echarts":
                # Python code generation for echarts interactive chart
                method = PythonChain.from_echarts_prompt(
                    llm,
                    return_intermediate_steps=return_intermediate_steps,
                    verbose=True,
                    memory=self._memory,
                    user_id=user_id,
                    chat_id=chat_id,
                    code_execution_mode=code_execution_mode,
                    jupyter_kernel_pool=jupyter_kernel_pool,
                )
                _input = {"question": user_intent, "data_info": _concat_grounding_source()}
                result = method(_input)
            else:
                raise ValueError(f"Usage {self._usage} not supported yet.")
        else:
            raise ValueError(f"Programming language {self._programming_language} not supported.")
        return result
from typing import Any, Dict, Tuple, Union
from abc import ABC

from langchain.base_language import BaseLanguageModel
from langchain import PromptTemplate

from real_agents.adapters.callbacks.executor_streaming import ExecutorStreamingChainHandler
from real_agents.adapters.data_model import DatabaseDataModel, TableDataModel, ImageDataModel
from real_agents.adapters.llm import LLMChain


class DataSummaryExecutor(ABC):
    tool_name = "DataProfiling"

    def _intelligent_summary(self, grounding_source: ImageDataModel, num_insights: int, llm: BaseLanguageModel) -> str:
        """Use LLM to generate data summary."""
        pass


class TableSummaryExecutor(DataSummaryExecutor):
    SUMMARY_PROMPT_TEMPLATE = """
{table_info}

Provide a succinct yet meaningful summary of the table with less than 20 words, encapsulating its essence beyond just enumerating the columns. Please ensure your summary is a complete sentence and include it within <summary></summary> tags."
Note the table actually far more rows than shown above, so you MUST NOT make any rash conclusions based on the shown table rows or cells."
Then provide {num_insights} insightful and interesting suggestions in natural language that users can directly say to analyze the table. The suggestions should be able to be solved by python/sql."
The final results should be markdown '+' bullet point list, e.g., + The first suggestion.

Begin."
"""
    stream_handler = ExecutorStreamingChainHandler()

    def run(
        self,
        grounding_source: Union[TableDataModel, DatabaseDataModel],
        llm: BaseLanguageModel,
        use_intelligent_summary: bool = True,
        num_insights: int = 3,
    ) -> Dict[str, Any]:
        summary = ""
        if isinstance(grounding_source, TableDataModel):
            df = grounding_source.raw_data
            df_name = grounding_source.raw_data_name
            # Basic summary
            summary += f"Your table {df_name} contains {df.shape[0]} rows and {df.shape[1]} columns. "

            null_count = df.isnull().sum().sum()  # Get total number of null values
            unique_values_avg = df.nunique().mean()  # Get average number of unique values

            summary += f"On average, each column has about {unique_values_avg:.0f} unique values. "
            if null_count > 0:
                summary += f"Watch out, there are {null_count} missing values in your data. "
            else:
                summary += "Good news, no missing values in your data. "

            # Intelligent summary
            if use_intelligent_summary:
                intelligent_summary = self._intelligent_summary(
                    grounding_source,
                    num_insights=num_insights,
                    llm=llm,
                )
                table_summary, suggestions = self._parse_output(intelligent_summary)
                summary += table_summary
                summary += "\n" + "Here are some additional insights to enhance your understanding of the table."
                summary += "\n" + suggestions

            for stream_token in summary.split(" "):
                self.stream_handler.on_llm_new_token(stream_token)

        elif isinstance(grounding_source, DatabaseDataModel):
            # TODO: Convert to df or use SQL query for basic summary
            raise NotImplementedError("DatabaseDataModel is not supported yet.")
        else:
            raise ValueError(f"Unsupported grounding source type: {type(grounding_source)}")
        return summary

    def _intelligent_summary(
        self, grounding_source: Union[TableDataModel, DatabaseDataModel], num_insights: int, llm: BaseLanguageModel
    ) -> str:
        """Use LLM to generate data summary."""
        summary_prompt_template = PromptTemplate(
            input_variables=["table_info", "num_insights"],
            template=self.SUMMARY_PROMPT_TEMPLATE,
        )
        method = LLMChain(llm=llm, prompt=summary_prompt_template)
        result = method.run({"table_info": grounding_source.get_llm_side_data(), "num_insights": num_insights})
        return result

    def _parse_output(self, content: str) -> Tuple[str, str]:
        """Parse the output of the LLM to get the data summary."""
        from bs4 import BeautifulSoup

        # Using 'html.parser' to parse the content
        soup = BeautifulSoup(content, "html.parser")
        # Parsing the tag and summary contents
        try:
            table_summary = soup.find("summary").text
        except Exception:
            import traceback

            traceback.print_exc()
            table_summary = ""

        lines = content.split("\n")
        # Initialize an empty list to hold the parsed bullet points
        bullet_points = []
        # Loop through each line
        bullet_point_id = 1
        for line in lines:
            # If the line starts with '+', it is a bullet point
            if line.startswith("+"):
                # Remove the '+ ' from the start of the line and add it to the list
                bullet_points.append(f"{bullet_point_id}. " + line[1:].strip().strip('"'))
                bullet_point_id += 1
        return table_summary, "\n".join(bullet_points)


class ImageSummaryExecutor(DataSummaryExecutor):
    SUMMARY_PROMPT_TEMPLATE = """
{img_info}

Provide a succinct summary of the uploaded file with less than 20 words. Please ensure your summary is a complete sentence and include it within <summary></summary> tags. For image, just show its name is basically enough."
Then provide {num_insights} very simple and basic suggestions in natural language about further processing with the data. The suggestions should be able to be solved by python(e.g., grayscale, rescale, rotation, etc). The final results should be markdown '+' bullet point list, e.g., + The first suggestion."

Begin.
"""
    stream_handler = ExecutorStreamingChainHandler()

    def run(
        self,
        grounding_source: ImageDataModel,
        llm: BaseLanguageModel,
        use_intelligent_summary: bool = True,
        num_insights: int = 3,
    ) -> Dict[str, Any]:
        summary = ""
        if isinstance(grounding_source, ImageDataModel):
            # Basic summary
            raw_data = grounding_source.raw_data
            img_size, img_mode, img_format = raw_data["size"], raw_data["mode"], raw_data["format"]
            summary += f"Your image **{grounding_source.simple_filename}** is a {img_size[0]}x{img_size[1]} {img_mode} image in {img_format} format.\n"

            # Intelligent summary
            if use_intelligent_summary:
                intelligent_summary = self._intelligent_summary(
                    grounding_source,
                    num_insights=num_insights,
                    llm=llm,
                )
                _, suggestions = self._parse_output(intelligent_summary)
                summary += "\n" + "Here are some additional insights to enhance your understanding of the image"
                summary += "\n" + suggestions

            for stream_token in summary.split(" "):
                self.stream_handler.on_llm_new_token(stream_token)
        else:
            raise ValueError(f"Unsupported data summary for grounding source type: {type(grounding_source)}")
        return summary

    def _intelligent_summary(self, grounding_source: ImageDataModel, num_insights: int, llm: BaseLanguageModel) -> str:
        """Use LLM to generate data summary."""
        summary_prompt_template = PromptTemplate(
            input_variables=["img_info", "num_insights"],
            template=self.SUMMARY_PROMPT_TEMPLATE,
        )
        method = LLMChain(llm=llm, prompt=summary_prompt_template)
        result = method.run({"img_info": grounding_source.get_llm_side_data(), "num_insights": num_insights})
        return result

    def _parse_output(self, content: str) -> Tuple[str, str]:
        """Parse the output of the LLM to get the data summary."""
        from bs4 import BeautifulSoup

        # Using 'html.parser' to parse the content
        soup = BeautifulSoup(content, "html.parser")
        # Parsing the tag and summary contents
        try:
            table_summary = soup.find("summary").text
        except Exception:
            import traceback

            traceback.print_exc()
            table_summary = ""

        lines = content.split("\n")
        # Initialize an empty list to hold the parsed bullet points
        bullet_points = []
        # Loop through each line
        bullet_point_id = 1
        for line in lines:
            # If the line starts with '+', it is a bullet point
            if line.startswith("+"):
                # Remove the '+ ' from the start of the line and add it to the list
                bullet_points.append(f"{bullet_point_id}. " + line[1:].strip().strip('"'))
                bullet_point_id += 1
        return table_summary, "\n".join(bullet_points)
"""An agent designed to hold a conversation in addition to using tools."""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union
from typing_extensions import override
from langchain_core.pydantic_v1 import Field

from langchain.agents.agent import AgentOutputParser
from langchain.agents.utils import validate_tools_single_input
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun, Callbacks
from langchain.chains import LLMChain
from langchain.schema import AgentAction, AgentFinish, HumanMessage, AIMessage, BaseMessage, BaseOutputParser
from langchain.tools.base import BaseTool

from real_agents.adapters.agent_helpers.agent import Agent
from real_agents.adapters.agent_helpers.output_parser import ConversationOutputParser
from real_agents.data_agent.copilot_prompt import PREFIX, SUFFIX, TEMPLATE_TOOL_RESPONSE, fake_continue_prompt
from real_agents.adapters.data_model import DataModel, MessageDataModel
from langchain.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)


class ExceptionTool(BaseTool):
    name = "_Exception"
    description = "Exception tool"

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return query

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return query


class ConversationalChatAgent(Agent):
    """An agent designed to hold a conversation in addition to using data tools."""

    output_parser: ConversationOutputParser = Field(default_factory=ConversationOutputParser())
    template_tool_response: str = TEMPLATE_TOOL_RESPONSE
    continue_model: Optional[str] = None

    @classmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> ConversationOutputParser:
        return ConversationOutputParser()

    @property
    def _agent_type(self) -> str:
        raise NotImplementedError

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Thought:"

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        super()._validate_tools(tools)
        validate_tools_single_input(cls.__name__, tools)

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        system_message: str = PREFIX,
        human_message: str = SUFFIX,
        input_variables: Optional[List[str]] = None,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> BasePromptTemplate:
        # tools
        tool_strings = "\n".join([f"> {tool.name}: {tool.description}" for tool in tools])
        tool_names = ", ".join([tool.name for tool in tools])
        _output_parser = output_parser or cls._get_default_output_parser()

        # format instructions for system message
        format_instructions = _output_parser.get_format_instructions()
        format_instructions = format_instructions.format(tool_names=tool_names)

        # system message
        system_message = system_message + f"{tool_strings}\n\n{format_instructions}"

        # human input
        final_prompt = human_message
        if input_variables is None:
            input_variables = ["input", "chat_history", "agent_scratchpad"]
        messages = [
            SystemMessagePromptTemplate.from_template(system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(final_prompt),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)

    @override
    def _construct_scratchpad(self, intermediate_steps: List[Tuple[AgentAction, str]]) -> List[BaseMessage]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts: List[BaseMessage] = []

        # Try to only use AI message for scratchpad
        content = []
        for idx, (action, full_observation) in enumerate(intermediate_steps):
            content.append(MessageDataModel.extract_action_for_llm(action.log))

            observation = full_observation
            if isinstance(full_observation, DataModel):
                llm_raw_observation = full_observation.get_llm_side_data()
                observation = MessageDataModel.extract_tool_response_for_llm(llm_raw_observation)
                tool_response = self.template_tool_response.format(
                    observation=str(observation), tool_names=self.allowed_tools
                )
                if idx == len(intermediate_steps) - 1:
                    content.append(tool_response)
                else:
                    content.append(observation)
        content_str = "\n".join(content)
        thoughts.append(AIMessage(content=content_str))
        if self.continue_model is not None and len(intermediate_steps) != 0:
            thoughts.append(HumanMessage(content=fake_continue_prompt[self.continue_model]))
        return thoughts

    @override
    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        system_prompt = self.llm_chain.prompt.messages[0].format().content
        system_prompt_tokens = MessageDataModel._count_tokens(system_prompt)
        max_tokens = 8000
        max_gen_tokens = 1000
        # FIXME: need more accurate token limit calculation
        full_inputs = MessageDataModel.truncate_chat_history(
            full_inputs, max_token=max_tokens - system_prompt_tokens - max_gen_tokens
        )
        full_output = self.llm_chain.predict(callbacks=callbacks, **full_inputs)

        return self.output_parser.parse(full_output)

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callbacks: Callbacks = None,
        output_parser: Optional[AgentOutputParser] = None,
        system_message: str = PREFIX,
        human_message: str = SUFFIX,
        input_variables: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)

        _output_parser = output_parser or cls._get_default_output_parser()
        prompt = cls.create_prompt(
            tools,
            system_message=system_message,
            human_message=human_message,
            input_variables=input_variables,
            output_parser=_output_parser,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
        )
        tool_names = [tool.name for tool in tools]
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            **kwargs,
        )
FUNCTION_ROLE_PLAY = """def generate_continuous_elegant_python_code(history_dict: Dict[str, str], reference_code: str = "") -> str:
    \"\"\"
    This function generates elegant, coherent Python code based on a history of previously executed code and its corresponding results. The code is generated in response to human questions and is intended to continue from the last provided code snippet.

    The function takes two inputs: a `history_dict` and an optional `reference_code` string.

    The `history_dict` is a dictionary with the following keys:
    - 'history code': Contains the chat history of previously executed code snippets. It may be initially empty but will accumulate executed code over time.
    - 'human question': Contains the current question or instruction posed by the human user, which the generated code should respond to. Be aware that sometimes the 'human question' could contain code snippets, including instructions for loading data, which may need to be handled differently. It's not always appropriate to directly use the code in 'human question' without consideration.
    - 'data': Contains a list of data previews available for the task. It may include tables, images, and other data types.

    The `reference_code` string is optional and contains example codes, often related to a specific library or task, which can serve as a template for the code generation process. This parameter can be empty.

    IMPORTANT: Always refer to this history and the `reference_code` when generating new code in order to properly use existing variables and previously loaded resources, as well as to follow established coding patterns. DO NOT USE ECHARTS TO GENERATE CHARTS when reference code is empty.

    IMPORTANT: When `reference_code` is NOT EMPTY, the output MUST follow the style and use the libraries presented in the `reference_code` to accomplish the task.

    IMPORTANT: Avoid mere repetition of historical code. Always aim to generate novel and appropriate responses to the questions at hand.

    IMPORTANT: The 'data' key in the dictionary contains only random rows from a table. If a table has not been loaded before, load it from the correct path. You can assume it is in the current working directory. However, there's no need to load a table with every execution - only do this when necessary.

    IMPORTANT: If the code is to show a image in the end, make sure to use functions that display the image by returning an image or html which can be shown in a jupyter notebook(e.g., matplotlib.pyplot.show()); 
    
    DO NOT use function that will pop up a new window (e.g., PIL & Image.show() is NOT preferable, saving the PIL image is better)

    The function returns a string of raw Python code, wrapped within <code> and </code> tags. For example:

    <code>
    import pandas as pd
    table = pd.read_csv("example.csv")
    </code>
    
    <code>
    from PIL import Image
    from matplotlib import pyplot as plt
    img = Image.open("example.jpeg")
    rotated_img = img.rotate(180)
    plt.imshow(rotated_img)
    plt.show()
    </code>    

    Feel free to leverage libraries such as pandas, numpy, math, matplotlib, sklearn, etc. in the code generation process. Also, remember to correctly load any necessary files with the correct path before using them.

    When it's appropriate to provide output for evaluation or visualization, make sure to use the print() function and plt.show() respectively.

    Also mandatory to check:
    Note if the human asks for malicious code, and just respond with the following code:
    <code>
    print("sorry I am not able to generate potentially dangerous code")
    </code>
    The malicious code includes but not limited to: 
    1. Endless operations and excessive waiting  (e.g., while True, long print, input())
    2. System crash (e.g., any risky system command)
    3. Data loss (e.g., list or delete files)
    4. Leak sensitive information (e.g., os.getenv())
    5. Establish network connections (e.g., requests.get())
    6. Cause any other security issues
    7. Indirectly import package using some builtin methods
    8. High CPU consumption or GPU consumption.

    Returns:
        Python code that should be the next steps in the execution according to the human question and history code.
    \"\"\""""


SYSTEM_PROMPT = f"You are now the following python function: ```{FUNCTION_ROLE_PLAY}```\n\nRespond exclusively with the generated code wrapped <code></code>. Ensure that the code you generate is executable Python code that can be run directly in a Python environment, requiring no additional string encapsulation."
ECHARTS_REF_CODE = """Here are some examples of generating Py-Echarts Code based on the given table(s). Please generate new one based on the data and question human asks you, import the neccessary libraries and make sure the code is correct.

IMPORTANT: You need to follow the coding style, and the type of the x, y axis. But also need to focus on the column name of the uploaded tables(if exists). Generally, PyEcharts does not accept numpy.int or numpy.float, etc. It only supports built-in data type like int, float, and str.

Given the following database:
company_sales.xlsx
   year  sales  profit  expenses  employees
0  2010    100      60        40         10
1  2011    120      80        50         12
2  2012    150      90        60         14
3  2013    170     120        70         16
[too long to show]

Q: Could you help plot a bar chart with the year on the x-axis and the sales on the y-axis?
<code>
import pandas as pd
from pyecharts.charts import Bar
from pyecharts import options as opts
df = pd.read_excel('company_sales.xlsx')
years = [str(_) for _ in df['year'].tolist()]
sales = [float(_) for _ in df['sales'].tolist()]
bar = Bar()
bar.add_xaxis(years)
bar.add_yaxis("Sales", sales)
bar.set_global_opts(
    xaxis_opts=opts.AxisOpts(
        type_="category",
        name="Year",
    ),
    yaxis_opts=opts.AxisOpts(
        type_="value",
        name="Sales",
    ),
    title_opts=opts.TitleOpts(title="Sales over Years"),
)
# Render the chart
ret_json = bar.dump_options()
print(ret_json)
</code>

Given the same `company_sales.xlsx`.
Q: A line chart comparing sales and profit over time would be useful. Could you help plot it?
<code>
import pandas as pd
from pyecharts.charts import Line
from pyecharts import options as opts
df = pd.read_excel('company_sales.xlsx')
year = [str(_) for _ in df["year"].to_list()]
sales = [float(_) for _ in df["sales"].to_list()]
profit = [float(_) for _ in df["profit"].to_list()]
line = Line()
# Add x-axis and y-axis data
line.add_xaxis(year)
line.add_yaxis("Sales", sales)
line.add_yaxis("Profit", profit)
line.set_global_opts(
    xaxis_opts=opts.AxisOpts(
        type_="category", # better use category rather than value
        name="year",
        min_=min(year),
        max_=max(year),
    ),
    yaxis_opts=opts.AxisOpts(
        type_="value",
        name="price",
    ),
    title_opts=opts.TitleOpts(title="Sales and Profit over Time"),
)
ret_json = line.dump_options()
print(ret_json)
</code>


Given the same `company_sales.xlsx`.
Q: A `stacked` line chart comparing sales and profit over time would be useful. Could you help plot it?
Note: stacked line chart is more fancy in display, while the former is more neat.
<code>
import pandas as pd
from pyecharts.charts import Line
from pyecharts import options as opts
df = pd.read_excel('company_sales.xlsx')
year = [str(_) for _ in df["year"].to_list()] # better use category rather than value
sales = [float(_) for _ in df["sales"].to_list()]
profit = [float(_) for _ in df["year"].to_list()]
line = Line()
# Add x-axis and y-axis data
line.add_xaxis(year)
line.add_yaxis("Sales", df["sales"].tolist(), stack="")
line.add_yaxis("Profit", df["profit"].tolist(), stack="")
line.set_global_opts(
    xaxis_opts=opts.AxisOpts(
        type_="category",
        name="year",
        min_=min(year),
        max_=max(year),
    ),
    yaxis_opts=opts.AxisOpts(
        type_="value",
        name="price",
        axistick_opts=opts.AxisTickOpts(is_show=True),
        splitline_opts=opts.SplitLineOpts(is_show=True),
    ),
    title_opts=opts.TitleOpts(title="Sales and Profit over Time"),
)
line.set_series_opts(
    areastyle_opts=opts.AreaStyleOpts(opacity=0.5),
)
ret_json = line.dump_options()
print(ret_json)
</code>


Given the following database:
shop_sales.tsv
   shop_id  total_sales  espresso_sales  latte_sales  cappuccino_sales  city_population
0        1         5000            1500         2000              1500           500000
1        2         5500            1800         2200              1500           800000
2        3         6000            2000         2500              1500          1200000
3        4         4500            1300         1800              1400           300000
4        5         6200            2200         2700              1300           600000
Q: I would like a pie chart showing the sales proportion of espresso, latte, and cappuccino for Shop 1.
<code>
import pandas as pd
from pyecharts.charts import Pie
from pyecharts import options as opts
df = pd.read_csv('shop_sales.tsv', sep='\\t')
shop1 = df.loc[df['shop_id'] == 1]
data_pair = [
    ('Espresso', float(shop1['espresso_sales'].item())), # pair must be (str, int/float)
    ('Latte', float(shop1['latte_sales'].item())),       # pair must be (str, int/float)
    ('Cappuccino', int(shop1['cappuccino_sales'].item())), # pair must be (str, int/float)
]
pie = Pie()
pie.add(
    series_name="Sales Breakdown",
    data_pair=data_pair,
    radius=["30%", "75%"],
)
pie.set_global_opts(
    title_opts=opts.TitleOpts(
        title="Coffee Sales Breakdown for Shop 1",
    ),
)
ret_json = pie.dump_options()
print(ret_json)
</code>

Q: Generate a scatter plot.
<code>
import random
from pyecharts import options as opts
from pyecharts.charts import Scatter
from pyecharts.faker import Faker

# Create some random data
data = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(10)]
x = [i[0] for i in data]
y = [i[1] for i in data]
print(data)
scatter = Scatter()
scatter.add_xaxis(x)
scatter.add_yaxis("size", y)
scatter.set_global_opts(
    xaxis_opts=opts.AxisOpts(type_="value"), # scatter x axis must be numeric
    yaxis_opts=opts.AxisOpts(type_="value"), # scatter y axis must be numeric
    title_opts=opts.TitleOpts(title="Scatter Plot Example"),
    visualmap_opts=opts.VisualMapOpts(type_="size", max_=max(y), min_=min(y)),
)
ret_json = scatter.dump_options()
print(ret_json)
</code>
"""

FUNCTION_ROLE_PLAY = """def generate_continuous_elegant_python_echarts_code(reference_code: str, history_dict: Dict[str, str]) -> str:
    \"\"\"
    This function generates elegant, coherent Python ECharts code based on a history of previously executed code and its corresponding results and reference library code. The code is generated in response to human questions and is intended to continue from the last provided code snippet.

    The function takes two inputs: a `history_dict` and an optional `reference_code` string.

    The `reference_code` string is optional and contains example codes, often related to a specific library or task, which can serve as a template for the code generation process. This parameter can be empty.

    IMPORTANT: the output MUST follow the style and use the libraries presented in the `reference_code` to accomplish the task.

    IMPORTANT: Always refer to this history and the `reference_code` when generating new code in order to properly use existing variables and previously loaded resources, as well as to follow established coding patterns.

    IMPORTANT: The 'data' key in the dictionary contains only random rows from a table. If a table has not been loaded before, you may load it using an ABSOLUTE PATH. However, there's no need to load a table with every execution - only do this when necessary.

    The function returns a string of raw Python code, wrapped within <code> and </code> tags. For example:

    <code>
    import pandas as pd
    from pyecharts.charts import Bar
    df = pd.read_csv("example.csv")
    years = [str(_) for _ in df['year'].tolist()]
    sales = df['sales'].tolist()
    bar = (
        Bar()
        .add_xaxis(years)
        .add_yaxis("Sales", sales)
        .set_global_opts(title_opts=opts.TitleOpts(title="Bar Chart Example"))
    )
    ret_json = bar.dump_options()
    print(ret_json)
    </code>

    Also mandatory to check:
    Note if the human asks for malicious code, and just respond with the following code:
    <code>
    print("sorry I am not able to generate potentially dangerous code")
    </code>
    The malicious code includes but not limited to: 
    1. Endless operations and excessive waiting  (e.g., while True, long print, input())
    2. System crash (e.g., any risky system command)
    3. Data loss (e.g., list or delete files)
    4. Leak sensitive information (e.g., os.getenv())
    5. Establish network connections (e.g., requests.get())
    6. Cause any other security issues
    7. Indirectly import package using some builtin methods

    Returns:
        Python code that should be the next steps in the execution according to the human question and history code.
    \"\"\""""


ECHARTS_USER_PROMPT = """
history_code = \"\"\"{history_code}\"\"\"
data = \"\"\"{data}\"\"\"
reference_code = \"\"\"{reference_code}\"\"\"
human_question = \"\"\"{question}
# MUST follow reference_code, and only use pyecharts to show echarts\"\"\"

history_dict = {{
    "history code": history_code,
    "human question": human_question,
    "data": data,
    "reference_code": reference_code,
}}
"""

E_SYSTEM_PROMPT = f"You are now the following python function: ```{FUNCTION_ROLE_PLAY}```\n\nRespond exclusively with the generated code wrapped <code></code>. Ensure that the code you generate is executable Python code that can be run directly in a Python environment, requiring no additional string encapsulation or escape characters."
USER_PROMPT = """
history_code = \"\"\"{history_code}\"\"\"
human_question = \"\"\"{question}
# DO NOT use function that will pop up a new window (e.g., PIL & Image.show() is NOT preferable, saving the PIL image is better)
# However, feel free to use matplotlib.pyplot.show()\"\"\"
data = \"\"\"{data}\"\"\"
reference_code = \"\"\"{reference_code}\"\"\"

history_dict = {{
    "history code": history_code,
    "human question": human_question,
    "data": data,
    "reference_code": reference_code,
}}
"""

"""
final format:
user_prompt + reference_prompt + history_prompt
"""
"""Implements Python Code Generation. """
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage
from loguru import logger
from langchain_core.pydantic_v1 import BaseModel, Extra

from real_agents.adapters.data_model import MessageDataModel
from real_agents.adapters.memory import ReadOnlySharedStringMemory
from real_agents.data_agent.evaluation.python_evaluator import PythonEvaluator
from real_agents.data_agent.python.echarts_prompt import E_SYSTEM_PROMPT, ECHARTS_REF_CODE, ECHARTS_USER_PROMPT
from real_agents.data_agent.python.system_prompt import SYSTEM_PROMPT
from real_agents.data_agent.python.python_prompt import USER_PROMPT
from real_agents.adapters.llm import LLMChain


class PythonChain(Chain, BaseModel):
    """Chain for Generating Python Code"""

    llm_chain: LLMChain

    memory: Optional[ReadOnlySharedStringMemory] = None
    stop: str = "\n\n"
    get_answer_expr: str = ""
    python_globals: Optional[Dict[str, Any]] = None
    python_locals: Optional[Dict[str, Any]] = None
    output_key: str = "result"  #: :meta private:
    return_intermediate_steps: bool = False
    code_execution_mode: str = "local"
    jupyter_kernel_pool: Optional[Any] = None
    reference_code: str = ""

    chat_id: Optional[str] = None
    user_id: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.ignore
        arbitrary_types_allowed = True

    def _validate_inputs(self, inputs: Dict[str, str]) -> None:
        """Check that all inputs are present."""
        missing_keys = set(self.input_keys).difference(inputs)
        if "chat_history" in missing_keys:
            missing_keys.remove("chat_history")
        if missing_keys:
            raise ValueError(f"Missing some input keys: {missing_keys}")

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return ["data_info", "question", "chat_history"]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        if not self.return_intermediate_steps:
            return [self.output_key]
        else:
            return [self.output_key, "intermediate_steps"]

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        logger.bind(msg_head="PythonChain inputs").trace(inputs)

        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        _run_manager.on_text(inputs[self.input_keys[0]])
        inputs["chat_history"] = ""
        if self.memory is not None:
            inputs["chat_history"] = self.memory.load_memory_variables({})["chat_history"]
            inputs["chat_history"] = MessageDataModel.extract_code_for_python_tool(inputs["chat_history"])

        history = {
            "history_code": inputs["chat_history"],
            "question": inputs["question"],
            "data": inputs["data_info"],
            "reference_code": self.reference_code,
        }

        # we apply llm as a magic function, which serves as python code generation func.
        raw_output = self.llm_chain.run(**history)

        def _extract_code(_raw_output: str) -> str:
            # Using 'html.parser' to parse the content
            soup = BeautifulSoup(_raw_output, "html.parser")
            try:
                _raw_output = soup.find("code").text
            except:
                pass
            if "```python:" in _raw_output:
                pattern = r"```python\n{(.*?)}\n```"
                match = re.search(pattern, _raw_output, re.DOTALL)
                if match:
                    return match.group(1)
                else:
                    return _raw_output
            else:
                return _raw_output

        code = _extract_code(raw_output).replace("\\n", "\n")

        logger.bind(msg_head="PythonChain generated program").trace(code)

        repl = PythonEvaluator(
            code_execution_mode=self.code_execution_mode,
            jupyter_kernel_pool=self.jupyter_kernel_pool,
        )

        """
        Since there will be error if we try to launch matplotlib GUI in the server,
        I add this line to avoid backend execution of matplotlib for now.
        """
        result = repl.run(code + f"\n{self.get_answer_expr}", user_id=self.user_id, chat_id=self.chat_id)

        logger.bind(msg_head="PythonChain execution result").trace(result)

        output = {self.output_key: result}
        if self.return_intermediate_steps:
            output["intermediate_steps"] = code
        return output

    @classmethod
    def create_python_prompt(cls, system_prompt: str, reference_code_prompt: str) -> BasePromptTemplate:
        input_variables = ["history_code", "question", "data", "reference_code"]
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessagePromptTemplate.from_template(template=USER_PROMPT),
        ]

        return ChatPromptTemplate(input_variables=input_variables, messages=messages)

    @classmethod
    def create_echarts_prompt(cls, system_prompt: str, reference_code_prompt: str) -> BasePromptTemplate:
        input_variables = ["history_code", "question", "data", "reference_code"]
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessagePromptTemplate.from_template(template=ECHARTS_USER_PROMPT),
        ]

        return ChatPromptTemplate(input_variables=input_variables, messages=messages)

    @classmethod
    def from_python_prompt(cls, llm: BaseLanguageModel, **kwargs: Any) -> PythonChain:
        """Load from Echarts prompt."""
        llm_chain = LLMChain(llm=llm, prompt=cls.create_python_prompt(SYSTEM_PROMPT, ""))
        return cls(
            llm_chain=llm_chain,
            get_answer_expr="",
            reference_code="",
            **kwargs,
        )

    @classmethod
    def from_echarts_prompt(cls, llm: BaseLanguageModel, **kwargs: Any) -> PythonChain:
        """Load from Echarts prompt."""
        llm_chain = LLMChain(llm=llm, prompt=cls.create_echarts_prompt(E_SYSTEM_PROMPT, ""))
        return cls(
            llm_chain=llm_chain,
            get_answer_expr="",
            reference_code=ECHARTS_REF_CODE,
            **kwargs,
        )

    @property
    def _chain_type(self) -> str:
        return "program_chain"
from real_agents.data_agent.copilot import ConversationalChatAgent
from real_agents.data_agent.evaluation.python_evaluator import PythonEvaluator
from real_agents.data_agent.evaluation.sql_evaluator import SQLEvaluator
from real_agents.data_agent.executors.code_generation_executor import CodeGenerationExecutor
from real_agents.data_agent.executors.data_summary_executor import (
    DataSummaryExecutor,
    TableSummaryExecutor,
    ImageSummaryExecutor,
)
from real_agents.data_agent.executors.kaggle_data_loading_executor import KaggleDataLoadingExecutor
from real_agents.data_agent.python.base import PythonChain
from real_agents.data_agent.sql.base import SQLDatabaseChain
from real_agents.adapters.schema import SQLDatabase
