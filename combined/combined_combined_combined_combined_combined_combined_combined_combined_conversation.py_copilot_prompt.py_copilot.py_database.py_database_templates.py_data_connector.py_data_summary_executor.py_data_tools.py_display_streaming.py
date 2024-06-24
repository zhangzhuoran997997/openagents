import struct
import json
import datetime
from typing import Any, Generator
from bson.objectid import ObjectId
from flask import jsonify, request, Response

from backend.app import app
from backend.utils.user_conversation_storage import get_user_conversation_storage
from backend.main import threading_pool, logger
from backend.schemas import DEFAULT_USER_ID
from backend.schemas import INTERNAL, UNFOUND


@app.route("/api/conversations/get_conversation_list", methods=["POST"])
def get_conversation_list() -> Response:
    """Gets the history conversations."""
    print('执行get_conversation_list')
    request_json = request.get_json()
    user_id = request_json.pop("user_id", DEFAULT_USER_ID)
    conversations = []
    try:
        # Login with API Key, then retrieve the user history based
        # on the hashed API key.
        db = get_user_conversation_storage()
        conversation_list = db.conversation.find({"user_id": user_id})
        for conversation in conversation_list:
            conversations.append(
                {
                    "id": str(conversation["_id"]),
                    "name": conversation["name"],
                    "folderId": conversation["folder_id"],
                }
            )
    except Exception as e:
        return Response(response=None,
                        status=f'{INTERNAL} error fetch conversation list')
    return jsonify(conversations)


@app.route("/api/conversations/get_folder_list", methods=["POST"])
def get_folder_list() -> Response:
    """Gets the folder list."""
    print('执行get_folder_list')
    user_id = DEFAULT_USER_ID
    folders = []
    try:
        db = get_user_conversation_storage()
        folder_list = db.folder.find({"user_id": user_id})
        for folder in folder_list:
            folders.append(
                {
                    "id": str(folder["_id"]),
                    "name": folder["name"],
                    "type": "chat",
                }
            )
        return jsonify({"success": True, "data": folders})
    except Exception as e:
        return Response(response=None, status=f'{INTERNAL} error fetch folder list')


def process_rich_content_item(data: dict, message_id: str) -> dict:
    """Processes the rich content from db format into frontend renderable format."""
    processed_items: dict = {"intermediateSteps": [], "finalAnswer": []}
    if "intermediate_steps" in data:
        for item in data["intermediate_steps"]:
            processed_items["intermediateSteps"].append(
                {"message_id": message_id, "content": item["text"],
                 "type": item["type"]}
            )
    if "final_answer" in data:
        for item in data["final_answer"]:
            processed_items["finalAnswer"].append(
                {"message_id": message_id, "content": item["text"],
                 "type": item["type"]}
            )
    return processed_items


@app.route("/api/conversation", methods=["POST"])
def get_conversation_content() -> Response:
    """Gets the conversation content for one assigned conversation."""
    print('执行conversation')
    request_json = request.get_json()
    conversation_id = request_json.get("chat_id", None)
    if conversation_id is not None:
        try:
            db = get_user_conversation_storage()
            conversation = db.conversation.find_one({"_id": ObjectId(conversation_id)})
            message_list = db.message.find({"conversation_id": conversation_id}).sort(
                "_id", -1)
            messages = [
                {
                    "id": message["message_id"],
                    "parent_message_id": message["parent_message_id"],
                    "role": message["role"],
                    "content": message["data_for_human"] if message[
                                                                "role"] == "user" else None,
                    "type": "rich_message" if isinstance(message["data_for_human"],
                                                         dict) else "",
                    "richContent": process_rich_content_item(message["data_for_human"],
                                                             message["message_id"])
                    if isinstance(message["data_for_human"], dict)
                    else None,
                }
                for message in message_list
            ]

            def _get_activated_conversation_branch(messages: list) -> list:
                # By default, the latest message is the end point, e.g., the current branch of messages.
                activated_messages: list = []
                end_point = messages[0]["id"]
                while len(messages) > 0 and end_point != -1:
                    flag = False
                    for msg in messages:
                        if msg["id"] == end_point:
                            if end_point == msg["parent_message_id"]:
                                flag = False
                                break
                            activated_messages = [msg] + activated_messages
                            end_point = msg["parent_message_id"]
                            flag = True
                            break
                    if not flag:
                        break
                return activated_messages

            # Find the current activated branch of messages as frontend only shows one branch

            if messages:
                messages = _get_activated_conversation_branch(messages)

            logger.bind(msg_head=f"get_activated_message_list").debug(messages)

            conversation = {
                "id": conversation_id,
                "name": conversation["name"],
                "messages": messages,
                "agent": conversation["agent"],
                "prompt": conversation["prompt"],
                "temperature": conversation["temperature"],
                "folderId": conversation["folder_id"],
                "bookmarkedMessagesIds": conversation["bookmarked_message_ids"],
                "selectedCodeInterpreterPlugins": conversation[
                    "selected_code_interpreter_plugins"],
                "selectedPlugins": conversation["selected_plugins"],

            }
            return jsonify(conversation)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response(response=None,
                            status=f'{INTERNAL} error fetch conversation')
    else:
        return Response(response=None, status=f'{INTERNAL} error fetch conversation')


@app.route("/api/conversations/update_conversation", methods=["POST"])
def update_conversation() -> Response:
    """Updates a conversation name."""
    try:
        request_json = request.get_json()
        conversations = request_json["conversations"]
        db = get_user_conversation_storage()
        messages = []
        success = True
        update_key_dict = {"name": "name", "folder_id": "folderId"}
        for conversation_to_update in conversations:
            conversation_id = conversation_to_update["id"]
            name = conversation_to_update["name"]
            updates = {}
            for key in update_key_dict.keys():
                if update_key_dict[key] in conversation_to_update:
                    updates[key] = conversation_to_update[update_key_dict[key]]
            if conversation_id is not None:
                try:
                    db.conversation.update_one({"_id": ObjectId(conversation_id)},
                                               {"$set": updates})
                    messages.append("Conversation name updated to {}.".format(name))
                except Exception as e:
                    messages.append(str(e))
                    success = False
            else:
                success = False
                messages.append("Missing conversation id or title.")
        return jsonify({"success": success, "message": " ".join(messages)})
    except Exception as e:
        return Response(response=None, status=f"{INTERNAL} error fetch conversation")


@app.route("/api/conversations/update_folder", methods=["POST"])
def update_folder() -> Response:
    """Update a folder name."""
    request_json = request.get_json()
    folder_id = request_json["folder_id"]
    folder_name = request_json["name"]
    try:
        db = get_user_conversation_storage()
        db.folder.update_one({"_id": ObjectId(folder_id)},
                             {"$set": {"name": folder_name}})
        return jsonify({"success": True,
                        "message": "Folder name updated to {}.".format(folder_name)})
    except Exception as e:
        return Response(response=None, status=f"{INTERNAL} error update folder")


@app.route("/api/conversations/register_folder", methods=["POST"])
def register_folder() -> Response:
    """Creates a new folder."""
    request_json = request.get_json()
    user_id = request_json.pop("user_id", DEFAULT_USER_ID)
    folder = request_json.get("folder", None)
    if folder:
        try:
            db = get_user_conversation_storage()
            folder = db.folder.insert_one({"name": folder["name"], "user_id": user_id})
            return jsonify({"id": str(folder.inserted_id),
                            "message": "Register folder successfully."})
        except Exception as e:
            return Response(response=None, status=f"{INTERNAL} error register folder")
    else:
        return Response(response=None, status=f"{UNFOUND} missing folder")


@app.route("/api/conversations/register_conversation", methods=["POST"])
def register_conversation() -> Response:
    """Creates a new conversation."""
    request_json = request.get_json()
    user_id = request_json.pop("user_id", DEFAULT_USER_ID)
    conversation = request_json.get("conversation", None)
    if conversation:
        try:
            db = get_user_conversation_storage()
            conversation_id = conversation["id"]
            if conversation_id is not None and db.conversation.find_one(
                    {"_id": ObjectId(conversation_id)}):
                updates = {
                    "name": conversation["name"],
                    "agent": conversation["agent"],
                    "prompt": conversation["prompt"],
                    "temperature": conversation["temperature"],
                    "folder_id": conversation["folderId"],
                    "bookmarked_message_ids": conversation.get("bookmarkedMessagesIds",
                                                               None),
                    "selected_code_interpreter_plugins": conversation[
                        "selectedCodeInterpreterPlugins"],
                    "selected_plugins": conversation["selectedPlugins"],
                }
                db.conversation.update_one({"_id": ObjectId(conversation_id)},
                                           {"$set": updates})
            else:
                conversation = db.conversation.insert_one(
                    {
                        "name": conversation["name"],
                        "agent": conversation["agent"],
                        "prompt": conversation["prompt"],
                        "temperature": conversation["temperature"],
                        "folder_id": conversation["folderId"],
                        "bookmarked_message_ids": conversation.get(
                            "bookmarkedMessagesIds", None),
                        "hashed_api_key": "",
                        "user_id": user_id,
                        "selected_code_interpreter_plugins": conversation[
                            "selectedCodeInterpreterPlugins"],
                        "selected_plugins": conversation["selectedPlugins"],
                        "timestamp": datetime.datetime.utcnow(),
                    }
                )
                conversation_id = str(conversation.inserted_id)
            return jsonify({"id": conversation_id})
        except Exception as e:
            return Response(response=None,
                            status=f"{INTERNAL} error register conversation")
    else:
        return Response(response=None, status=f"{UNFOUND} missing conversation")


@app.route("/api/conversations/delete_conversation", methods=["POST"])
def delete_conversation() -> Response:
    """Deletes a conversation."""
    request_json = request.get_json()
    chat_id = request_json.get("chat_id", None)
    if chat_id:
        try:
            db = get_user_conversation_storage()
            db.conversation.delete_one({"_id": ObjectId(chat_id)})
            db.message.delete_many({"conversation_id": chat_id})
            return jsonify({"success": True, "message": "Conversation is deleted."})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)})
    else:
        return jsonify({"success": False, "message": "chat_id is missing"})


@app.route("/api/conversations/delete_folder", methods=["POST"])
def delete_folder() -> Response:
    """Deletes a folder."""
    request_json = request.get_json()
    folder_id = request_json.get("folder_id", None)
    if folder_id:
        try:
            db = get_user_conversation_storage()
            db.folder.delete_one({"_id": ObjectId(folder_id)})
            return jsonify({"success": True, "message": "Folder is deleted."})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)})
    else:
        return jsonify({"success": False, "message": "folder_id is missing"})


@app.route("/api/conversations/clear", methods=["POST"])
def clear_all_conversation() -> Response:
    """Clears all previous conversations."""
    request_json = request.get_json()
    user_id = request_json.pop("user_id", DEFAULT_USER_ID)
    if user_id:
        try:
            db = get_user_conversation_storage()
            db.conversation.delete_many({"user_id": user_id})
            db.folder.delete_many({"user_id": user_id})
            db.message.delete_many({"user_id": user_id})
            return jsonify({"success": True, "message": "Clear All Conversations."})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)})
    else:
        return jsonify({"success": False, "message": "user_id is missing"})


@app.route("/api/conversations/stop_conversation", methods=["POST"])
def stop_generation() -> Response:
    """Stops the current generation, cut on streaming."""
    try:
        request_json = request.get_json()
        chat_id = request_json["chat_id"]
        threading_pool.kill_thread(chat_id)
    except Exception as e:
        print(e)
        return Response(response={}, status=f"{INTERNAL} error stopping")

    def pack_json(object: Any) -> bytes:
        json_text = json.dumps(object)
        return struct.pack("<i", len(json_text)) + json_text.encode("utf-8")

    def yield_stop() -> Generator[bytes, Any, None]:
        yield pack_json({"success": False, "error": "stop"})

    return Response(response={})
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
from __future__ import annotations

import os
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, inspect

from real_agents.adapters.data_model.base import DataModel
from real_agents.adapters.data_model.table import TableDataModel
from real_agents.adapters.data_model.templates.skg_templates.database_templates import serialize_db
from real_agents.adapters.schema import SQLDatabase


class DatabaseDataModel(DataModel):
    """A data model for database."""

    @classmethod
    def from_table_data_model(cls, table_data_model: TableDataModel) -> DatabaseDataModel:
        os.makedirs(f".db_cache/{table_data_model.id}", exist_ok=True)
        db_path = os.path.join(
            f".db_cache/{table_data_model.id}", os.path.splitext(table_data_model.raw_data_name)[0] + ".db"
        )
        engine = create_engine(f"sqlite:///{db_path}")
        table_data_model.raw_data.to_sql(table_data_model.raw_data_name, engine, if_exists="replace")
        db = SQLDatabase(engine)
        return cls.from_raw_data(raw_data=db, raw_data_name=table_data_model.raw_data_name)

    def insert_table_data_model(self, table_data_model: TableDataModel) -> None:
        engine = self.raw_data.engine
        table_data_model.raw_data.to_sql(table_data_model.raw_data_name, engine)

    def get_llm_side_data(self, serialize_method: str = "database", num_visible_rows: int = 3) -> Any:
        db = self.raw_data
        formatted_db = serialize_db(db, serialize_method, num_visible_rows)
        return formatted_db

    def get_human_side_data(self) -> Any:
        # In the frontend, we show the first few rows of each table
        engine = self.raw_data.engine
        inspector = inspect(engine)
        table_names = inspector.get_table_names()

        # Loop through each table name, creating a DataFrame from the first three rows of each table
        df_dict = {}
        for table_name in table_names:
            query = f"SELECT * FROM {table_name} LIMIT 3"
            df = pd.read_sql(query, engine)
            df_dict[table_name] = df
        return df_dict
import sqlite3
from typing import Dict, Union

import pandas as pd
import tiktoken


from real_agents.adapters.data_model.templates.skg_templates.table_templates import (
    convert as convert_table,
)
from real_agents.adapters.schema import SQLDatabase


def convert(db_input: Union[str, Dict[str, pd.DataFrame]], visible_rows_num: int = 3) -> Dict[str, str]:
    """
    Convert database data to string representations in different formats.

    :param db_input: the path to the sqlite database file, or a pd.DataFrame.
    :param visible_rows_num: the number of rows to be displayed in each table.
    :return: A dictionary with the string database representations in different formats.
    """
    if isinstance(db_input, str):
        conn = sqlite3.connect(db_input)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_names = [name[0] for name in cursor.fetchall()]
        dfs = {table_name: pd.read_sql_query(f"SELECT * FROM {table_name}", conn) for table_name in table_names}
    elif isinstance(db_input, dict) and all(isinstance(df, pd.DataFrame) for df in db_input.values()):
        dfs = db_input
    else:
        raise ValueError("db_input should be either a SQLite database file path or a dictionary of pandas DataFrames")

    representations = {
        "Markdown": "",
        "HTML": "",
        "LaTeX": "",
        "CSV": "",
        "TSV": "",
        "reStructuredText": "",
        "BBCode": "",
        "MediaWiki": "",
        "Org mode": "",
        "PrettyTable": "",
        "SQL": "",
    }

    for table_name, df in dfs.items():
        table_data = {"cols": df.columns.tolist(), "rows": df.values.tolist()}
        table_representations = convert_table(table_data, table_name, visible_rows_num)
        for _format, table_representation in table_representations.items():
            representations[_format] += table_representation + "\n\n"

    return representations


def serialize_db(
    db: SQLDatabase,
    serialize_method: str = "database",
    num_visible_rows: int = 3,
    max_tokens: int = 1000,
) -> str:
    """Convert database engine to a string representation."""
    if serialize_method == "database":
        # TODO: Now access the internal variable
        setattr(db, "_sample_rows_in_table_info", num_visible_rows)
        string = db.get_table_info()
        # Truncate the string if it is too long
        enc = tiktoken.get_encoding("cl100k_base")
        enc_tokens = enc.encode(string)
        if len(enc_tokens) > max_tokens:
            string = enc.decode(enc_tokens[:max_tokens])
    else:
        raise ValueError("Unknown serialization method.")
    return string
import os
from flask import request, Response
from kaggle.api.kaggle_api_extended import KaggleApi

from backend.app import app
from backend.utils.utils import create_personal_folder
from backend.schemas import UNFOUND, INTERNAL, DEFAULT_USER_ID

api = KaggleApi()
api.authenticate()


@app.route("/api/kaggle/download_dataset", methods=["POST"])
def kaggle_dataset_download() -> dict | Response:
    """Use Kaggle-api to connect. """
    request_json = request.get_json()
    user_id = request_json.pop("user_id", DEFAULT_USER_ID)
    url = request_json["url"]
    if url.startswith("http"):
        return {"success": False,
                "message": "Please remove the http in your submitted URL."}
    kaggle_dataset_id = url.replace("www.kaggle.com/datasets/", "")
    if not kaggle_dataset_id:
        return {"success": False, "message": "Please input a valid Kaggle dataset URL."}
    root_path = create_personal_folder(user_id)
    if os.path.exists(root_path) and os.path.isdir(root_path):
        try:
            path = os.path.join(root_path, kaggle_dataset_id)
            api.dataset_download_files(kaggle_dataset_id, path=path, unzip=True)
            return {"success": True, "message": "Download {} successfully.",
                    "data_path": path}
        except Exception as e:
            return Response(response=None,
                            status=f"{INTERNAL} Error Downloading, please try another datasets")
    else:
        return Response(response=None, status=f"{UNFOUND} Missing User folder")
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
from typing import List
from flask import jsonify

from backend.app import app

DATA_TOOLS = [
    {
        "type": "language",
        "id": "1cea1f39-fe63-4b08-83d5-fa4c93db0c87",
        "name": "SQLQueryBuilder",
        "name_for_human": "SQL",
        "pretty_name_for_human": "SQL Query Generation",
        "icon": "",
        "description": "Using SQL as the programming language",
    },
    {
        "type": "language",
        "id": "0c135359-af7e-473b-8425-1393d2943b57",
        "name": "PythonCodeBuilder",
        "name_for_human": "Python",
        "pretty_name_for_human": "Python Code Generation",
        "icon": "",
        "description": "Using Python as the programming language",
    },
    {
        "type": "tool",
        "id": "a86aebe1-a780-4038-a333-fb2a9d2d25fc",
        "name": "Echarts",
        "name_for_human": "Echarts",
        "pretty_name_for_human": "Echarts",
        "icon": "",
        "description": "Enhancing the analyzing experience with interactive charts",
    },
    {
        "type": "tool",
        "id": "c7c826ba-5884-4e2b-b27c-fedea30c1749",
        "name": "KaggleDataLoader",
        "name_for_human": "Kaggle Data Search",
        "pretty_name_for_human": "Kaggle Data Search",
        "icon": "",
        "description": "Search & Connect to kaggle datasets",
    },
    {
        "type": "tool",
        "id": "8f8e8dbc-ae5b-4950-9f4f-7f5238978806",
        "name": "DataProfiling",
        "name_for_human": "Data Profiling",
        "pretty_name_for_human": "Data Profiling",
        "icon": "",
        "description": "Intelligent profiling for your data",
    },
]


@app.route("/api/data_tool_list", methods=["POST"])
def get_data_tool_list() -> List[dict]:
    print("请求数据工具列表")
    """Gets the data tool list. """
    for i, tool in enumerate(DATA_TOOLS):
        cache_path = f"backend/static/images/{tool['name']}.cache"
        with open(cache_path, 'r') as f:
            image_content = f.read()
            DATA_TOOLS[i]["icon"] = image_content

    return jsonify(DATA_TOOLS)
from typing import Dict, Optional, List
import json
import base64
import re
import ast

import mo_sql_parsing
from langchain_core.pydantic_v1 import BaseModel

from real_agents.adapters.data_model import MessageDataModel, DataModel


def is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def split_text_and_code(text: str) -> List:
    pattern = r"(```[\s\S]+?```)"
    result = [x for x in re.split(pattern, text) if x.strip()]

    return result


def detect_code_type(code) -> str:
    # Attempt Python parsing
    try:
        ast.parse(code)
        return "python"
    except SyntaxError:
        pass

    # Attempt SQL parsing
    try:
        mo_sql_parsing.parse(code)
        return "sql"
    except:
        pass

    # If all else fails, it's probably plain text
    return "text"


def add_backticks(text: str) -> str:
    """Add backticks to code blocks."""
    text_type = detect_code_type(text)
    if is_json(text):
        text = "```json\n" + text + "\n```"
    elif text_type == "python":
        if not text.startswith("```") and not text.endswith("```"):
            text = "```python\n" + text + "\n```"
    elif text_type == "sql":
        if not text.startswith("```") and not text.endswith("```"):
            text = "```sql\n" + text + "\n```"
    return text


class DisplayStream(BaseModel):
    """The display stream to parse and render tokens and blocks"""

    streaming_mode: str = "plain"
    action: str = ""
    action_cache: str = ""
    execution_result_max_tokens: int = 1000
    escape: bool = False
    escape_cache: str = ""
    llm_call_id: int = -1

    def reset(self):
        self.streaming_mode = "plain"
        self.action = ""
        self.action_cache = ""
        self.escape = False
        self.escape_cache = ""

    def display(self, token: Dict) -> Optional[List[Dict]]:
        # Reset if the llm_call_id has changed
        if token["llm_call_id"] != self.llm_call_id:
            self.llm_call_id = token["llm_call_id"]
            self.reset()
        # Handle escape characters
        import codecs

        if token["text"] == "\\":
            self.escape = True
            self.escape_cache = "\\"
            return None
        else:
            if self.escape:
                try:
                    token["text"] = codecs.decode(self.escape_cache + token["text"], "unicode_escape")
                    self.escape = False
                    self.escape_cache = ""
                except Exception as e:
                    self.escape_cache += token["text"]
        # Tool selection
        if self.action != "" and token["type"] != "action":
            # An action has been generated
            if self.action != "Final Answer":
                _pretty_name = self.action
                self.action_cache = self.action
                self.action = ""
                return [{"text": _pretty_name, "type": "tool", "final": False}]
        if token["type"] == "plain":
            # Display plain text
            if self.streaming_mode == "identifier":
                return None
            else:
                self.streaming_mode = "plain"
                return [{"text": token["text"], "type": "transition", "final": False}]
        elif token["type"] == "identifier":
            self.streaming_mode = "identifier"
            return None
        elif token["type"] == "key":
            self.streaming_mode = "key"
            return None
        elif token["type"] == "action":
            self.streaming_mode = "action"
            self.action += token["text"]
            return None
        elif token["type"] == "action_input":
            self.streaming_mode = "action_input"
            if self.action == "Final Answer":
                return [{"text": token["text"], "type": "plain", "final": True}]
        elif token["type"] == "block":
            observation = token["text"]
            result = self._display_observation(observation=observation)
            return result
        else:
            raise ValueError("Unknown token type: {}".format(token["type"]))

    def _display_observation(self, observation: Dict) -> Optional[List]:
        """Display the observation, i.e., the response from the tool

        Args:
            observation: Tool response block

        Returns:
            A list of display blocks to the frontend
        """
        tool_response_list = []
        if isinstance(observation, str):
            # Observation is a plain text (not used)
            tool_response_list.append({"text": observation, "type": "plain", "final": False})
            return tool_response_list

        assert isinstance(observation, DataModel), "Observation must be a DataModel object"
        observation = observation.get_human_side_data()

        assert isinstance(observation, Dict), "Observation must be a Dict object"

        result = observation.get("result", "")
        result_str = str(result)
        # Code & Plugin block
        if "intermediate_steps" in observation:
            intermediate_steps = observation["intermediate_steps"]
            if self.action_cache == "PythonCodeBuilder":
                intermediate_steps = "```python\n" + intermediate_steps + "\n```"
            elif self.action_cache == "SQLCodeBuilder":
                intermediate_steps = "```sql\n" + intermediate_steps + "\n```"
            else:
                intermediate_steps = add_backticks(intermediate_steps)
            tool_response_list.append({"text": intermediate_steps, "type": "plain", "final": False})

        # Execution result
        if not observation["success"]:
            tool_response_list.append({"text": result_str, "type": "error", "final": False})
        else:
            result_str = MessageDataModel.truncate_text(result_str, max_token=self.execution_result_max_tokens)

            tool_response_list.append(
                {
                    "text": f"""```console\n{result_str.strip(' ').strip("```")}\n```"""
                    if result_str.strip("```")
                    else "",
                    "type": "execution_result",
                    "final": False,
                }
            )
        # Kaggle search and connect
        if "kaggle_action" in observation:
            kaggle_action = observation["kaggle_action"]
            tool_response_list.append(
                {
                    "text": json.dumps(observation["kaggle_output_info"]),
                    "type": f"kaggle_{kaggle_action}",
                    "final": False,
                }
            )
        # Image result, e.g., matplotlib
        if "images" in observation:
            try:
                for image in observation["images"]:
                    if isinstance(image, str):
                        continue
                    image_data_64 = "data:image/png;base64," + base64.b64encode(
                        base64.b64decode(image.data["image/png"])
                    ).decode("utf-8")
                    tool_response_list.append({"text": image_data_64, "type": "image", "final": False})
            except:
                tool_response_list.append(
                    {"text": "[ERROR]: error rendering image/png", "type": "error", "final": False}
                )
        # Echarts
        if "echarts" in observation:
            chart_json = observation["echarts"]

            if is_json(chart_json):
                tool_response_list.append({"text": chart_json, "type": "echarts", "final": False})
            else:
                tool_response_list.append({"text": f"""```json{chart_json}```""", "type": "plain", "final": False})

        return tool_response_list
