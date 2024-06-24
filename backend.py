import os
import sys
import base64
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import pandas as pd
import tiktoken
from flask import Request
from sqlalchemy import create_engine
from PIL import Image
from loguru import logger

from real_agents.adapters.data_model import (
    DatabaseDataModel,
    DataModel,
    ImageDataModel,
    TableDataModel,
    KaggleDataModel,
)
from real_agents.data_agent import (
    DataSummaryExecutor,
    TableSummaryExecutor,
    ImageSummaryExecutor,
)
from real_agents.adapters.schema import SQLDatabase
from backend.utils.running_time_storage import get_running_time_storage
from backend.app import app
from backend.schemas import DEFAULT_USER_ID

TABLE_EXTENSIONS = {"csv", "xls", "xlsx", "tsv"}
DOCUMENT_EXTENSIONS = {"pdf", "doc", "docx", "txt"}
DATABASE_EXTENSIONS = {"sqlite", "db"}
IMAGE_EXTENSIONS = {"jpg", "png", "jpeg"}
ALLOW_EXTENSIONS = TABLE_EXTENSIONS | DOCUMENT_EXTENSIONS | DATABASE_EXTENSIONS | IMAGE_EXTENSIONS

LOCAL = "local"
REDIS = "redis"


class VariableRegister:
    def __init__(self, name=None, backend=LOCAL) -> None:
        self.backend = backend
        if self.backend == LOCAL:
            self.variables: Dict[int, Any] = {}
            self.counter = 1
        elif self.backend == REDIS:
            assert name is not None
            self.name = name
            self.counter_name = f"{self.name}:counter"
            self.variables_name = f"{self.name}:variables"
            with app.app_context():
                self.redis_client = get_running_time_storage()
            if not self.redis_client.exists(self.counter_name):
                self.redis_client.set(self.counter_name, 0)
            else:
                logger.bind(msg_head="VariableRegister").debug(
                    f"Reuse the {self.counter_name}({self.redis_client.get(self.counter_name)}) and {self.variables_name}."
                )
        else:
            raise ValueError("Unknown backend option: {}".format(self.backend))

    def add_variable(self, variable: Any) -> int:
        if self.backend == LOCAL:
            variable_id = self.counter
            self.variables[variable_id] = variable
            self.counter += 1
            return variable_id
        elif self.backend == REDIS:
            variable_id = self.redis_client.incrby(self.counter_name, 1)
            self.redis_client.hset(self.variables_name, variable_id, variable)
            return variable_id

    def get_variable(self, variable_id: int) -> Any:
        if self.backend == LOCAL:
            return self.variables.get(variable_id, None)
        elif self.backend == REDIS:
            return self.redis_client.hget(self.variables_name, variable_id)

    def get_variables(self) -> Dict[int, Any]:
        if self.backend == LOCAL:
            return self.variables
        elif self.backend == REDIS:
            return self.redis_client.hgetall(self.variables_name)


def get_user_and_chat_id_from_request_json(request_json: Dict) -> Tuple[str, str]:
    user_id = request_json.pop("user_id", DEFAULT_USER_ID)
    chat_id = request_json["chat_id"]
    return user_id, chat_id


def get_user_and_chat_id_from_request(request: Request) -> Tuple[str, str]:
    user_id = request.form.get("user_id", DEFAULT_USER_ID)
    chat_id = request.form.get("chat_id")
    return user_id, chat_id


def load_grounding_source(file_path: str) -> Any:
    # TODO: Maybe convert to DataModel here
    suffix = Path(file_path).suffix
    if Path(file_path).is_dir():
        # Assume it is a collection of csv files, usually downloaded from kaggle.
        grounding_source = {}
        for file in Path(file_path).iterdir():
            if file.suffix == ".csv":
                grounding_source[file.as_posix()] = pd.read_csv(file, index_col=False)
            else:
                raise ValueError("Only csv files are allowed in the directory")
    elif suffix == ".csv":
        grounding_source = pd.read_csv(file_path, index_col=False)
    elif suffix == ".tsv" or suffix == ".txt":
        grounding_source = pd.read_csv(file_path, sep="\t")
    elif suffix == ".xlsx" or suffix == ".xls":
        grounding_source = pd.read_excel(file_path)
    elif suffix == ".db" or suffix == ".sqlite":
        engine = create_engine(f"sqlite:///{file_path}")
        grounding_source = SQLDatabase(engine)
        return grounding_source
    elif suffix == ".png" or suffix == ".jpg" or suffix == ".jpeg":
        img = Image.open(file_path)
        with open(file_path, "rb") as image2string:
            converted_string = "data:image/png;base64," + base64.b64encode(image2string.read()).decode("utf-8")
        grounding_source = {
            "base64_string": converted_string,
            "format": img.format,
            "size": img.size,
            "mode": img.mode,
        }
    else:
        raise ValueError("File type not allowed to be set as grounding source")
    return grounding_source


def get_data_model_cls(file_path: str) -> DataModel:
    suffix = Path(file_path).suffix
    if Path(file_path).is_dir():
        data_model_cls = KaggleDataModel
    elif suffix == ".csv":
        data_model_cls = TableDataModel
    elif suffix == ".tsv" or suffix == ".txt":
        raise NotImplementedError("Not implemented yet")
    elif suffix == ".xlsx" or suffix == ".xls":
        data_model_cls = TableDataModel
    elif suffix == ".sqlite" or suffix == ".db":
        data_model_cls = DatabaseDataModel
    elif suffix == ".jpeg" or suffix == ".png" or suffix == ".jpg":
        data_model_cls = ImageDataModel
    else:
        raise ValueError("File type not allowed to be set as grounding source")
    return data_model_cls


def get_data_summary_cls(file_path: str) -> DataSummaryExecutor:
    suffix = Path(file_path).suffix
    if suffix == ".csv":
        data_summary_cls = TableSummaryExecutor
    elif suffix == ".tsv" or suffix == ".txt":
        raise NotImplementedError("Not implemented yet")
    elif suffix == ".xlsx" or suffix == ".xls":
        data_summary_cls = TableSummaryExecutor
    elif suffix == ".sqlite" or suffix == ".db":
        data_summary_cls = TableSummaryExecutor
    elif suffix == ".jpeg" or suffix == ".png" or suffix == ".jpg":
        data_summary_cls = ImageSummaryExecutor
    else:
        raise ValueError("File type not allowed to be set as grounding source")
    return data_summary_cls


def allowed_file(filename: Union[str, Path]) -> bool:
    if isinstance(filename, str):
        filename = Path(filename)
    suffix = filename.suffix[1:]
    if suffix in ALLOW_EXTENSIONS:
        return True
    else:
        return False


def is_table_file(filename: Union[str, Path]) -> bool:
    if isinstance(filename, str):
        filename = Path(filename)
    suffix = filename.suffix[1:]
    if suffix in TABLE_EXTENSIONS:
        return True
    else:
        return False


def is_document_file(filename: Union[str, Path]) -> bool:
    if isinstance(filename, str):
        filename = Path(filename)
    suffix = filename.suffix[1:]
    if suffix in DOCUMENT_EXTENSIONS:
        return True
    else:
        return False


def is_sqlite_file(filename: Union[str, Path]) -> bool:
    if isinstance(filename, str):
        filename = Path(filename)
    suffix = filename.suffix[1:]
    if suffix in DATABASE_EXTENSIONS:
        return True
    else:
        return False


def is_image_file(filename: Union[str, Path]) -> bool:
    if isinstance(filename, str):
        filename = Path(filename)
    suffix = filename.suffix[1:]
    if suffix in IMAGE_EXTENSIONS:
        return True
    else:
        return False


def remove_nan(file_path: str) -> None:
    """
    We only support csv file in the current version
    By default, we remove columns that contain only nan values
    For columns that have both nan values and non-nan values, we replace nan values with the mean (number type)
    or the mode (other type)
    """
    if file_path.endswith("csv"):
        df = pd.read_csv(file_path)
        columns = list(df.columns)
        nan_columns = []
        for c in columns:
            if all(list(df[c].isnull())):
                nan_columns.append(c)
        df.drop(columns=nan_columns, inplace=True)
        columns = list(df.columns)
        for c in columns:
            try:
                fillin_value = df[c].mean()
            except Exception:
                fillin_value = df[c].mode()
            df[c].fillna(value=fillin_value, inplace=True)
        df.to_csv(file_path)


def is_valid_input(user_intent: str, max_token_limit: int = 2000) -> bool:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = len(enc.encode(user_intent))
    return tokens <= max_token_limit


def error_rendering(error_message: str) -> str:
    """Map (certain) error message to frontend rendering form, otherwise show
    'internal backend  error'. Currently, only handle OpenAI error message.
    """
    if "openai" in error_message:
        if "Timeout" in error_message:
            return "OpenAI timeout error. Please try again."
        elif "RateLimitError" in error_message:
            return "OpenAI rate limit error. Please try again."
        elif "APIConnectionError" in error_message:
            return "OpenAI API connection error. Please try again."
        elif "InvalidRequestError" in error_message:
            return "OpenAI invalid request error. Please try again."
        elif "AuthenticationError" in error_message:
            return "OpenAI authentication error. Please try again."
        elif "ServiceUnavailableError" in error_message:
            return "OpenAI service unavailable error. Please try again."
    else:
        return "Internal backend error. Please try again."


def init_log(**sink_channel):
    """Initialize loguru log information"""

    # Just for sys.stdout log message
    format_stdout = (
        "<g>{time:YYYY-MM-DD HH:mm:ss}</g> | <lvl>{level}</lvl> - {extra[user_id]}++{extra[chat_id]}-><y>{extra[api]}</y> "
        "<LC>{extra[msg_head]}</LC>:{message}"
    )

    # Avoid unexpected KeyError
    # Do not unpack key-value pairs, but save all records.
    format_full_extra = (
        "<g>{time:YYYY-MM-DD HH:mm:ss}</g> | <lvl>{level}</lvl> - <c><u>{name}</u></c> | {message} - {extra}"
    )

    logger.remove()

    logger.configure(
        handlers=[
            dict(sink=sys.stdout, format=format_stdout, level="TRACE"),
            dict(
                sink=sink_channel.get("error"),
                format=format_full_extra,
                level="ERROR",
                diagnose=False,
                rotation="1 week",
            ),
            dict(
                sink=sink_channel.get("runtime"),
                format=format_full_extra,
                level="DEBUG",
                diagnose=False,
                rotation="20 MB",
                retention="20 days",
            ),
            dict(
                sink=sink_channel.get("serialize"),
                level="DEBUG",
                diagnose=False,
                serialize=True,
            ),
        ],
        extra={"user_id": "", "chat_id": "", "api": "", "msg_head": ""},
    )

    return logger


def create_personal_folder(user_id: str) -> str:
    # mkdir user folder
    from backend.main import app

    user_folder = os.path.join(app.config["UPLOAD_FOLDER"], user_id)
    os.makedirs(user_folder, exist_ok=True)
    # mkdir chat folder under user folder
    return user_folder
import json
import re
import struct
import time
from typing import Any, Dict, List, Optional, Literal
import multiprocess
import requests
from bs4 import BeautifulSoup

from backend.display_streaming import DisplayStream
from backend.main import logger, message_pool, threading_pool
from backend.utils.user_conversation_storage import get_user_conversation_storage
from backend.utils.utils import error_rendering
from backend.memory import MessageMemoryManager
from backend.schemas import (
    APP_TYPES,
    TIME_OUT_MAP,
    HEARTBEAT_INTERVAL,
    STREAM_BLOCK_TYPES,
    STREAM_TOKEN_TYPES,
    EXECUTION_RESULT_MAX_TOKENS_MAP,
)
from real_agents.data_agent import DataSummaryExecutor
from real_agents.adapters.callbacks.agent_streaming import AgentStreamingStdOutCallbackHandler
from real_agents.adapters.agent_helpers import Agent, AgentExecutor
from real_agents.adapters.llm import BaseLanguageModel


def check_url_exist(text: str) -> bool:
    """check in a text whether there is a url"""
    # this regex extracts the http(s) with whitespace or () in the beginning and end, since usually the url is surrounded by whitespace or ()
    # e.g. " https://google.com " or "(https://google.com)"
    url_regex = r"(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)"

    links = re.findall(url_regex, text)
    return len(links) > 0


# function to extract links from text
def extract_links(text: str) -> list[Any]:
    url_regex = r"(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)"
    links = re.findall(url_regex, text)
    return links


# function to extract image links from a webpage
def extract_title_and_image_links(url: str) -> (tuple[Literal[''], list] | tuple[Any, list]):
    try:
        res = requests.get(url, timeout=3)
        if res.status_code != 200:
            return "", []
        soup = BeautifulSoup(res.text, "html.parser")
        title_tag = soup.find_all("title")[0].text
        img_tags = soup.find_all("img")
        # List to store image links with large width and height
        large_img_links = []
        # List to store all image links
        all_img_links = []
        for img in img_tags:
            if "src" in img.attrs:
                all_img_links.append(img["src"])
                # Check if width and height attributes exist and add to the large list
                if "width" in img.attrs and "height" in img.attrs:
                    # Ensure the width and height attributes can be converted to integers
                    if int(img["width"]) > 100 and int(img["height"]) > 100:
                        large_img_links.append(img["src"])
                    else:
                        continue
        # If large images were found, return those, otherwise return all images
        img_links = large_img_links if large_img_links else []
        # fixme: handle the case there are no such tags

        return title_tag, img_links
    except requests.exceptions.Timeout:
        print("Request timed out!")
        return "", []
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return "", []


def extract_card_info_from_text(message: str) -> list:
    links = extract_links(message)
    rt = []
    for link in links:
        title, image_links = extract_title_and_image_links(link)
        if len(image_links) > 0:
            selected_image_link = image_links[0]
        else:
            selected_image_link = ""  # no image in this website
        rt.append({"title": title, "web_link": link, "image_link": selected_image_link})
    return rt


def extract_card_info_from_links(links: List[str]) -> list[dict[str, Any]]:
    rt = []
    for link in links:
        if check_url_exist(link):
            title, image_links = extract_title_and_image_links(link)
            if len(image_links) > 0:
                selected_image_link = image_links[0]
            else:
                selected_image_link = ""  # no image in this website
            rt.append({"title": title, "web_link": link, "image_link": selected_image_link})
        else:
            continue
    return rt


def pack_json(object: Any) -> bytes:
    json_text = json.dumps(object)
    return struct.pack("<i", len(json_text)) + json_text.encode("utf-8")


def _streaming_block(
    fancy_block: Dict,
    is_final: bool,
    user_id: str,
    chat_id: str,
) -> bytes:
    """Stream a block to the frontend."""
    render_position = "intermediate_steps" if not is_final else "final_answer"
    return pack_json(
        {
            render_position: [
                {
                    "type": fancy_block["type"],
                    "text": fancy_block["text"],
                }
            ],
            "is_block_first": True,
            "streaming_method": "block",
            "user_id": user_id,
            "chat_id": chat_id,
        }
    )


def _streaming_token(token: Dict, is_final: bool, user_id: str, chat_id: str, is_block_first: bool) -> bytes:
    """Streams a token to the frontend."""
    render_position = "intermediate_steps" if not is_final else "final_answer"
    return pack_json(
        {
            render_position: {
                "type": token["type"],
                "text": token["text"],
            },
            "is_block_first": is_block_first,
            "streaming_method": "char",
            "user_id": user_id,
            "chat_id": chat_id,
        }
    )


def _wrap_agent_caller(
    interaction_executor: Any,
    inputs: Dict[str, Any],
    chat_id: str,
    err_pool: Dict[str, Any],
    memory_pool: Dict[str, Any],
    callbacks: List,
) -> None:
    try:
        _ = interaction_executor(inputs, callbacks=callbacks)
        message_list_from_memory = MessageMemoryManager.save_agent_memory_to_list(interaction_executor.memory)
        memory_pool.update({chat_id: message_list_from_memory})
        del interaction_executor
    except Exception as e:
        import traceback

        traceback.print_exc()

        err_pool[chat_id] = f"{type(e).__name__}: {str(e)}"


def _combine_streaming(stream_list: List) -> List:
    """Combine the streaming tokens/blocks to be saved in database."""
    stream_list_combined = []
    current_type, current_text = None, ""
    for idx, item in enumerate(stream_list):
        if current_type in STREAM_TOKEN_TYPES and (item["type"] != current_type) or idx == len(stream_list) - 1:
            stream_list_combined.append(
                {
                    "type": current_type,
                    "text": current_text,
                }
            )
            current_text = ""
        if item["type"] in STREAM_BLOCK_TYPES:
            stream_list_combined.append(item)
        elif item["type"] in STREAM_TOKEN_TYPES:
            current_text += item["text"]
        current_type = item["type"]
    return stream_list_combined


def _render_preprocess(string: Optional[str] = None) -> str:
    """Preprocess the string to be rendered in frontend."""
    if string is None:  # this is due to openai stop policy or other stream issue
        return ""
    string = string.replace("$", "\$")
    return string


def single_round_chat_with_agent_streaming(
    stream_handler: AgentStreamingStdOutCallbackHandler,
    interaction_executor: AgentExecutor,
    user_intent: str,
    human_message_id: int,
    ai_message_id: int,
    user_id: str,
    chat_id: str,
    message_list: List[Dict[str, Any]],
    parent_message_id: int,
    llm_name: str,
    app_type: str = "plugins",
) -> Any:
    """Streams the response of the agent to the frontend."""
    assert app_type in APP_TYPES, f"app_type should be one of {APP_TYPES}"

    with multiprocess.Manager() as share_manager:
        err_pool: Dict[str, Any] = share_manager.dict()
        memory_pool: Dict[str, Any] = share_manager.dict()
        share_list = share_manager.list()
        memory_pool[chat_id] = []

        stream_handler.for_display = share_list

        chat_thread = multiprocess.Process(
            target=_wrap_agent_caller,
            args=(
                interaction_executor,
                {
                    "input": user_intent,
                },
                chat_id,
                err_pool,
                memory_pool,
                [stream_handler],
            ),
        )

        threading_pool.register_thread(chat_id, chat_thread)
        chat_thread.start()
        empty_s_time: float = -1
        last_heartbeat_time: float = -1
        timeout = TIME_OUT_MAP[app_type]
        LEFT_SIGN = "("
        RIGHT_SIGN = ")"
        start_buffer = False
        streamed_transition_text_buffer = ""
        streamed_links = []
        converted_card_info_list = []
        yield pack_json(
            {
                "human_message_id": human_message_id,
                "ai_message_id": ai_message_id,
            }
        )
        # Display streaming to frontend
        display_stream = DisplayStream(execution_result_max_tokens=EXECUTION_RESULT_MAX_TOKENS_MAP[app_type])
        is_block_first, current_block_type = False, None
        intermediate_list, final_list = [], []  # Only for database storage
        try:
            while chat_thread.is_alive() or len(stream_handler.for_display) > 0:
                # print(memory_pool, err_pool, "out")
                if stream_handler.is_end:
                    # The ending of the streaming is marked by the is_end variable from AgentStreamingStdOutCallbackHandler in agent_streaming.py
                    break

                if len(stream_handler.for_display) == 0:
                    # first time display list is empty
                    if empty_s_time == -1:
                        empty_s_time = time.time()
                    # already empty for some time
                    else:
                        if time.time() - empty_s_time > timeout and chat_thread.is_alive():
                            threading_pool.timeout_thread(chat_id)
                            break

                    if last_heartbeat_time == -1:
                        last_heartbeat_time = time.time()
                    else:
                        if time.time() - last_heartbeat_time > HEARTBEAT_INTERVAL and chat_thread.is_alive():
                            last_heartbeat_time = -1
                            yield _streaming_token(
                                {"text": "🫀", "type": "heartbeat", "final": False}, False, user_id, chat_id, False
                            )

                else:
                    empty_s_time = -1
                    last_heartbeat_time = -1

                while len(stream_handler.for_display) > 0:
                    token = stream_handler.for_display.pop(0)
                    items_to_display = display_stream.display(token)

                    # Skip the "identifier" and "key" token
                    if items_to_display is None:
                        continue

                    for item in items_to_display:
                        # Check if the block type is changed
                        if item["type"] != current_block_type:
                            current_block_type = item["type"]
                            is_block_first = True
                        else:
                            is_block_first = False
                        is_final = item.get("final", False)

                        # Render the item(s)
                        if item["type"] in STREAM_BLOCK_TYPES:
                            # Render image and echarts as block
                            yield _streaming_block(item, is_final, user_id, chat_id)
                        elif item["type"] in STREAM_TOKEN_TYPES:
                            # Render the rest as plain text
                            item["text"] = _render_preprocess(item["text"])
                            yield _streaming_token(item, is_final, user_id, chat_id, is_block_first)
                        # Save the intermediate steps and final answer
                        if is_final:
                            final_list.append(item)
                        else:
                            intermediate_list.append(item)

                        if item["type"] == "transition" and item["text"] == RIGHT_SIGN:
                            start_buffer = False
                            link = streamed_transition_text_buffer
                            streamed_transition_text_buffer = ""
                            card_info_list = extract_card_info_from_text(link)
                            # empty the buffer after extracting card info
                            streamed_transition_text_buffer = ""
                            if len(card_info_list) > 0:
                                streaming_card_info_list: list[dict[str, Any]] = [
                                    {
                                        "final_answer": {
                                            "text": json.dumps(card_info),
                                            "type": "card_info",
                                        },
                                        "is_block_first": False,
                                        "streaming_method": "card_info",
                                        "user_id": user_id,
                                        "chat_id": chat_id,
                                    }
                                    for card_info in card_info_list
                                ]
                                streamed_links.extend([card_info["web_link"] for card_info in card_info_list])
                                converted_card_info_list.extend(
                                    [
                                        {
                                            "text": stream_card_info["final_answer"]["text"],
                                            "type": stream_card_info["final_answer"]["type"],
                                        }
                                        for stream_card_info in streaming_card_info_list
                                    ]
                                )
                                for streaming_card_info in streaming_card_info_list:
                                    yield pack_json(streaming_card_info)

                        if start_buffer == True:
                            streamed_transition_text_buffer += item["text"]

                        if item["type"] == "transition" and item["text"] == LEFT_SIGN:
                            start_buffer = True

        except Exception as e:
            import traceback

            traceback.print_exc()
        # Wait for the chat thread to finish
        chat_thread.join()
        stop_flag, timeout_flag, error_msg = threading_pool.flush_thread(chat_id)
        error_msg = err_pool.pop(chat_id, None)
        # Response Error!!
        if stop_flag:
            yield pack_json({"success": False, "error": "stop"})
            return
        elif timeout_flag:
            yield pack_json({"success": False, "error": "timeout"})
            return
        elif error_msg is not None:
            error_msg_to_render = error_rendering(error_msg)
            yield pack_json({"success": False, "error": "internal", "error_msg": error_msg_to_render})
            return
        elif len(memory_pool[chat_id]) == 0:
            yield pack_json({"success": False, "error": "internal"})
            return
        # Response Success!!
        message_list_from_memory = memory_pool[chat_id]
        del stream_handler
        # share_manager.shutdown()
        del memory_pool, err_pool, share_list, share_manager, interaction_executor

    # Save conversation to memory
    new_human_message = message_list_from_memory[-2]
    new_ai_message = message_list_from_memory[-1]
    new_human_message.update({"message_id": human_message_id, "parent_message_id": parent_message_id})
    new_ai_message.update({"message_id": ai_message_id, "parent_message_id": human_message_id})
    message_list.extend([new_human_message, new_ai_message])

    logger.bind(user_id=user_id, chat_id=chat_id, api="/chat", msg_head="New human message").debug(new_human_message)
    logger.bind(user_id=user_id, chat_id=chat_id, api="/chat", msg_head="New ai message").debug(new_ai_message)

    MessageMemoryManager.set_pool_info_with_id(message_pool, user_id, chat_id, message_list)

    # Save conversation to database
    db = get_user_conversation_storage()
    # Combine the streaming tokens/blocks
    intermediate_list_combined = _combine_streaming(intermediate_list)
    final_list_combined = _combine_streaming(final_list)
    if len(converted_card_info_list) > 0:
        final_list_combined.extend(converted_card_info_list)
    # Insert User Message, if regenerate there is no need to insert again
    db.message.insert_one(
        {
            "conversation_id": chat_id,
            "user_id": user_id,
            "message_id": human_message_id,
            "parent_message_id": parent_message_id,
            "version_id": 0,
            "role": "user",
            "data_for_human": user_intent,
            "data_for_llm": message_list[-2]["message_content"],
            "raw_data": None,
        }
    )
    # Insert AI Message
    db.message.insert_one(
        {
            "conversation_id": chat_id,
            "user_id": user_id,
            "message_id": ai_message_id,
            "parent_message_id": human_message_id,
            "version_id": 0,
            "role": "assistant",
            "data_for_human": {
                "intermediate_steps": intermediate_list_combined,
                "final_answer": final_list_combined,
            },
            "data_for_llm": message_list[-1]["message_content"],
            "raw_data": None,
        }
    )


def _wrap_executor_caller(
    executor: Any, inputs: Any, llm: Any, chat_id: str, err_pool: Dict[str, Any], memory_pool: Dict[str, Any]
) -> None:
    try:
        results = executor.run(inputs, llm)
        message_list_from_memory = results
        memory_pool.update({chat_id: message_list_from_memory})
    except Exception as e:
        import traceback

        traceback.print_exc()

        err_pool[chat_id] = f"{type(e).__name__}: {str(e)}"


def single_round_chat_with_executor(
    executor: Any,
    user_intent: Any,
    human_message_id: int,
    ai_message_id: int,
    user_id: str,
    chat_id: str,
    message_list: List[Dict[str, Any]],
    parent_message_id: int,
    llm: BaseLanguageModel,
    app_type: str = "copilot",
) -> Any:
    """Streams the response of the executor to the frontend."""
    stream_handler = executor.stream_handler
    share_manager = multiprocess.Manager()
    err_pool: Dict[str, Any] = share_manager.dict()
    memory_pool: Dict[str, Any] = share_manager.dict()
    share_list = share_manager.list()
    stream_handler._all = share_list
    memory_pool[chat_id] = []
    chat_thread = multiprocess.Process(
        target=_wrap_executor_caller,
        args=(
            executor,
            user_intent,
            llm,
            chat_id,
            err_pool,
            memory_pool,
        ),
    )
    threading_pool.register_thread(chat_id, chat_thread)

    empty_s_time: float = -1
    timeout = TIME_OUT_MAP[app_type]
    chat_thread.start()
    yield pack_json(
        {
            "human_message_id": human_message_id,
            "ai_message_id": ai_message_id,
        }
    )
    # FIXME: treat data summary as a special tool
    STREAM_TOOL_TYPE = "tool"
    data_summary_tool_item = {
        "text": executor.tool_name,
        "type": STREAM_TOOL_TYPE,
    }
    yield _streaming_block(data_summary_tool_item, is_final=False, user_id=user_id, chat_id=chat_id)
    is_block_first = True
    final_answer = []
    while chat_thread.is_alive() or len(stream_handler._all) > 0:
        if stream_handler.is_end:
            break
        if len(stream_handler._all) == 0:
            # first time display list is empty
            if empty_s_time == -1:
                empty_s_time = time.time()
            # already empty for some time
            else:
                if time.time() - empty_s_time > timeout and chat_thread.is_alive():
                    threading_pool.timeout_thread(chat_id)
                    break
        else:
            empty_s_time = -1

        while len(stream_handler._all) > 0:
            text = stream_handler._all.pop(0)
            final_answer.append(text)
            if is_block_first:
                is_block_first_ = True
                is_block_first = False
            else:
                is_block_first_ = False
            yield pack_json(
                {
                    "final_answer": {
                        "type": "text",
                        "text": text + " ",
                    },
                    "is_block_first": is_block_first_,
                    "streaming_method": "char",
                    "user_id": user_id,
                    "chat_id": chat_id,
                }
            )
            time.sleep(0.035)
    chat_thread.join()
    stop_flag, timeout_flag, error_msg = threading_pool.flush_thread(chat_id)
    error_msg = err_pool.pop(chat_id, None)
    if stop_flag:
        yield pack_json({"success": False, "error": "stop"})
        return
    elif timeout_flag:
        yield pack_json({"success": False, "error": "timeout"})
        return
    elif error_msg is not None:
        error_msg_to_render = error_rendering(error_msg)
        yield pack_json({"success": False, "error": "internal", "error_msg": error_msg_to_render})
        return
    elif len(memory_pool[chat_id]) == 0 or len(final_answer) == 0:
        yield pack_json({"success": False, "error": "internal"})
        return
    # Response Success!!
    del share_list, stream_handler
    del memory_pool, err_pool, share_manager, executor

    # Save conversation to memory
    final_answer_str = " ".join(final_answer)
    message_list.append(
        {
            "message_id": ai_message_id,
            "parent_message_id": parent_message_id,
            "message_type": "ai_message",
            "message_content": final_answer_str,
        }
    )
    logger.bind(user_id=user_id, chat_id=chat_id, api="chat/", msg_head="New data summary message").debug(
        message_list[-1]
    )

    MessageMemoryManager.set_pool_info_with_id(message_pool, user_id, chat_id, message_list)

    # Database Operations
    db = get_user_conversation_storage()
    db.message.insert_one(
        {
            "conversation_id": chat_id,
            "user_id": user_id,
            "message_id": ai_message_id,
            "parent_message_id": parent_message_id,
            "version_id": 0,
            "role": "assistant",
            "data_for_human": {
                "intermediate_steps": [
                    data_summary_tool_item,
                ],
                "final_answer": [
                    {
                        "text": final_answer,
                        "type": "plain",
                    }
                ],
            },
            "data_for_llm": message_list[-1]["message_content"],
            "raw_data": None,
        }
    )
import redis
from flask import g
import os


def get_running_time_storage():
    """Connects to redis."""
    if "running_time_storage" not in g:
        g.running_time_storage = redis.Redis(host=os.getenv("REDIS_SERVER"), port=6379, decode_responses=True)
        # Set maxmemory to 200MB (value is in bytes)
        g.running_time_storage.config_set("maxmemory", "500000000")
        # Set maxmemory policy to allkeys-lru (Least Recently Used)
        g.running_time_storage.config_set("maxmemory-policy", "allkeys-lru")
    return g.running_time_storage
import json


def polish_echarts(echarts_str):
    """Polishes the echarts output into prettier format."""
    try:
        option = json.loads(echarts_str)

        # turn numeric axis into str
        category_flag = False
        for idx, series_data in enumerate(option["series"]):
            if series_data["type"] in ["bar", "line"]:
                category_flag = True
                break
        if category_flag:
            option["xAxis"][0]["data"] = [str(_) for _ in option["xAxis"][0]["data"]]
            for idx, series_data in enumerate(option["series"]):
                try:
                    option["series"][idx]["data"] = [[str(_[0]), _[1]] for _ in series_data["data"]]
                except:
                    continue
        for idx, series_data in enumerate(option["series"]):
            option["series"][idx]["label"]["show"] = False
        # set title position
        option["title"][0]["bottom"] = "bottom"
        option["title"][0]["left"] = "center"
        option["tooltip"]["alwaysShowContent"] = False

        return json.dumps(option)
    except Exception as e:
        print(e)
        return echarts_str
# Python program using
# traces to kill threads
from typing import Dict, Tuple, Optional
from multiprocess import Process


class ThreadManager:
    """Manager class of all user chat threads."""

    def __init__(self) -> None:
        self.thread_pool: Dict[str, Process] = {}
        self.stop_pool: Dict[str, bool] = {}
        self.timeout_pool: Dict[str, bool] = {}
        self.run_error_pool: Dict[str, Optional[str]] = {}

    def register_thread(self, chat_id, thread: Process) -> None:
        self.thread_pool[chat_id] = thread
        self.stop_pool[chat_id] = False
        self.timeout_pool[chat_id] = False
        self.run_error_pool[chat_id] = None

    def flush_thread(self, chat_id) -> Tuple[bool, bool, str]:
        # self.thread_pool[chat_id] = None
        stop_flag = self.stop_pool[chat_id]
        timeout_flag = self.timeout_pool[chat_id]
        run_error = self.run_error_pool[chat_id]
        _ = self.thread_pool.pop(chat_id)
        _.terminate()
        del _
        self.stop_pool.pop(chat_id)
        self.timeout_pool.pop(chat_id)
        self.run_error_pool.pop(chat_id)
        return stop_flag, timeout_flag, run_error

    def kill_thread(self, chat_id) -> None:
        if chat_id in self.thread_pool and self.thread_pool[chat_id] is not None:
            try:
                self.stop_pool[chat_id] = True
                while self.thread_pool[chat_id].is_alive():
                    self.thread_pool[chat_id].terminate()
            except Exception as e:
                if not self.thread_pool[chat_id].is_alive():
                    self.stop_pool[chat_id] = True
                pass

    def timeout_thread(self, chat_id) -> None:
        if chat_id in self.thread_pool and self.thread_pool[chat_id] is not None:
            try:
                self.timeout_pool[chat_id] = True
                while self.thread_pool[chat_id].is_alive():
                    self.thread_pool[chat_id].terminate()
            except Exception as e:
                if not self.thread_pool[chat_id].is_alive():
                    self.timeout_pool[chat_id] = True
                pass

    def error_thread(self, chat_id, e_msg: str) -> None:
        if chat_id in self.thread_pool and self.thread_pool[chat_id] is not None:
            try:
                self.run_error_pool[chat_id] = e_msg
            except:
                pass
import pymongo
from flask import g
import os

def get_user_conversation_storage():
    """Connects to mongodb."""
    if "user_conversation_storage" not in g:
        g.user_conversation_storage = pymongo.MongoClient("mongodb://{0}:27017/".format(os.getenv("MONGO_SERVER")))
    return g.user_conversation_storage["xlang"]


def close_user_conversation_storage():
    """Closes mongodb."""
    user_conversation_storage = g.pop("user_conversation_storage", None)
    if user_conversation_storage is not None:
        user_conversation_storage["xlang"].close()
import os

from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
current_path = os.path.abspath(__file__)
app.config["UPLOAD_FOLDER"] = os.path.dirname(current_path) + "/data"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
# Execute code locally or remotely on docker
app.config["CODE_EXECUTION_MODE"] = os.getenv("CODE_EXECUTION_MODE", "local")
CORS(app)
import traceback
from typing import Dict, List, Union
from flask import Response, request, stream_with_context, Response

from backend.api.file import _get_file_path_from_node
from backend.api.language_model import get_llm
from backend.app import app
from backend.main import (
    grounding_source_pool,
    jupyter_kernel_pool,
    logger,
    message_id_register,
    message_pool,
)
from backend.schemas import DEFAULT_USER_ID
from backend.utils.utils import create_personal_folder
from backend.utils.charts import polish_echarts
from backend.utils.streaming import (
    single_round_chat_with_executor,
    single_round_chat_with_agent_streaming,
)
from backend.utils.utils import get_data_summary_cls
from backend.schemas import OVERLOAD, UNAUTH, NEED_CONTINUE_MODEL
from real_agents.adapters.llm import BaseLanguageModel
from real_agents.adapters.agent_helpers import AgentExecutor, Tool
from real_agents.adapters.callbacks import AgentStreamingStdOutCallbackHandler
from real_agents.adapters.data_model import DatabaseDataModel, DataModel, JsonDataModel, \
    TableDataModel
from real_agents.adapters.executors import ChatExecutor
from real_agents.adapters.interactive_executor import initialize_agent
from real_agents.data_agent import CodeGenerationExecutor, KaggleDataLoadingExecutor
from real_agents.adapters.memory import ConversationReActBufferMemory, \
    ReadOnlySharedStringMemory


def create_interaction_executor(
        grounding_source_dict: Dict[str, DataModel],
        code_interpreter_languages: List[str],
        code_interpreter_tools: List[str],
        llm: BaseLanguageModel,
        llm_name: str,
        user_id: str = None,
        chat_id: str = None,
        code_execution_mode: str = "local",
) -> AgentExecutor:
    """Creates an agent executor for interaction.

    Args:
        grounding_source_dict: A dict of grounding source filepath and data.
        code_interpreter_languages: A string to indicate the programming language to use.
        code_interpreter_tools: A list of augmented data tools.
        llm: A llm model.
        llm_name: A string llm name.
        user_id: A string of user id.
        chat_id: A string chat id.
        code_execution_mode: A string indicating where code is executed.

    Returns:
        An agent executor.

    """
    # Initialize Memory
    memory = ConversationReActBufferMemory(
        memory_key="chat_history", return_messages=True, llm=llm, max_token_limit=3500
    )
    read_only_memory = ReadOnlySharedStringMemory(memory=memory)

    # Initialize tools(executors)
    basic_chat_executor = ChatExecutor()
    python_code_generation_executor = CodeGenerationExecutor(
        programming_language="python", memory=read_only_memory)
    sql_code_generation_executor = CodeGenerationExecutor(programming_language="sql",
                                                          memory=read_only_memory)
    echart_code_generation_executor = CodeGenerationExecutor(
        programming_language="python", memory=read_only_memory, usage="echarts"
    )
    kaggle_data_loading_executor = KaggleDataLoadingExecutor()

    def run_python_code_builder(term: str) -> Union[Dict, DataModel]:
        try:
            # Only TableDataModel are allowed as input to python
            # input_grounding_source = [gs for _, gs in grounding_source_dict.items()
            # if isinstance(gs, TableDataModel)]
            input_grounding_source = [gs for gs in grounding_source_dict.values()]
            # Get the result
            results = python_code_generation_executor.run(
                user_intent=term,
                llm=llm,
                grounding_source=input_grounding_source,
                user_id=user_id,
                chat_id=chat_id,
                code_execution_mode=code_execution_mode,
                jupyter_kernel_pool=jupyter_kernel_pool,
            )

            logger.bind(msg_head=f"PythonCodeBuilder results({llm})").debug(results)

            if results["result"]["success"]:
                if results["result"]["result"] is not None:
                    raw_output = results["result"]["result"]
                elif results["result"]["stdout"] != "":
                    raw_output = results["result"]["stdout"]
                else:
                    raw_output = ""
                observation = JsonDataModel.from_raw_data(
                    {
                        "success": True,
                        "result": raw_output,
                        "images": results["result"]["outputs"] if ".show()" in results[
                            "intermediate_steps"] else [],
                        "intermediate_steps": results["intermediate_steps"],
                    },
                    filter_keys=["images"],
                )
            else:
                observation = JsonDataModel.from_raw_data(
                    {
                        "success": False,
                        "result": results["result"]["error_message"],
                        "intermediate_steps": results["intermediate_steps"],
                    }
                )
            return observation
        except Exception as e:
            logger.bind(msg_head=f"PythonCodeBuilder error({llm})").error(str(e))

            traceback.print_exc()
            results = basic_chat_executor.run(user_intent=term, llm=llm)
            return results["result"]

    def run_sql_code_builder(term: str) -> Union[Dict, DataModel]:
        try:

            def convert_grounding_source_as_db(
                    grounding_source_dict: Dict[str, DataModel]
            ) -> Union[List[TableDataModel], DatabaseDataModel]:
                db_grounding_source = [
                    gs for _, gs in grounding_source_dict.items() if
                    isinstance(gs, DatabaseDataModel)
                ]
                table_grounding_source = [
                    gs for _, gs in grounding_source_dict.items() if
                    isinstance(gs, TableDataModel)
                ]
                assert len(db_grounding_source) <= 1
                if len(table_grounding_source) == 0:
                    # Only DatabaseDataModel. Assume there is at least one grounding
                    # source
                    return db_grounding_source[0]
                else:
                    for t_gs in table_grounding_source:
                        if len(db_grounding_source) == 0:
                            # No DatabaseDataModel, then convert the first TableModel
                            # into DatabaseDataModel.
                            if t_gs.db_view is None:
                                t_gs.set_db_view(
                                    DatabaseDataModel.from_table_data_model(t_gs))
                            db_gs = t_gs.db_view
                            db_grounding_source.append(db_gs)
                        else:
                            # Insert TableDataModel into the existing DatabaseDataModel
                            db_gs = db_grounding_source[0]
                            db_gs.insert_table_data_model(t_gs)
                    return db_gs

            input_grounding_source = convert_grounding_source_as_db(
                grounding_source_dict)
            results = sql_code_generation_executor.run(
                user_intent=term,
                grounding_source=input_grounding_source,
                llm=llm,
            )

            logger.bind(msg_head=f"SQLQueryBuilder results({llm})").debug(results)

            if results["result"]["success"]:
                observation = JsonDataModel.from_raw_data({
                    "success": True,
                    "result": results["result"]["result"],
                    "intermediate_steps": results["intermediate_steps"],
                })
            else:
                observation = JsonDataModel.from_raw_data({
                    "success": False,
                    "result": results["result"]["error_message"],
                    "intermediate_steps": results["intermediate_steps"],
                })
            return observation
        except Exception as e:
            logger.bind(msg_head=f"SQLQueryBuilder results({llm})").error(str(e))

            traceback.print_exc()
            results = basic_chat_executor.run(user_intent=term, llm=llm)
            return results["result"]

    def run_echarts_interactive_plotter(term: str) -> Union[Dict, DataModel]:
        try:
            input_grounding_source = [gs for _, gs in grounding_source_dict.items() if
                                      isinstance(gs, TableDataModel)]
            results = echart_code_generation_executor.run(
                user_intent=term,
                llm=llm,
                grounding_source=input_grounding_source,
                user_id=user_id,
                chat_id=chat_id,
                code_execution_mode=code_execution_mode,
                jupyter_kernel_pool=jupyter_kernel_pool,
            )

            logger.bind(msg_head=f"PlotInteractivePlotter results({llm})").debug(
                results)

            if results["result"]["success"]:
                results = JsonDataModel.from_raw_data(
                    {
                        "success": True,
                        "result": "",
                        "echarts": polish_echarts(results["result"]["stdout"]),
                        "intermediate_steps": results["intermediate_steps"],
                    },
                    filter_keys=["result", "echarts"],
                )
            else:
                results = JsonDataModel.from_raw_data(
                    {
                        "success": False,
                        "result": results["result"]["error_message"],
                        "intermediate_steps": results["intermediate_steps"],
                    }
                )
            return results
        except Exception as e:
            logger.bind(msg_head=f"PlotInteractivePlotter error({llm})").error(str(e))

            results = basic_chat_executor.run(user_intent=term, llm=llm)
            return results["result"]

    def run_kaggle_data_loader(term: str) -> Union[Dict, DataModel]:
        try:
            results = kaggle_data_loading_executor.run(
                user_intent=term,
                llm=llm,
            )
            logger.bind(msg_head=f"KaggleDataLoader results({llm})").debug(results)

            results = JsonDataModel.from_raw_data(
                {
                    "success": True,
                    "kaggle_action": results["kaggle_action"],
                    "kaggle_output_info": results["kaggle_output_info"],
                },
            )
            return results
        except Exception as e:
            logger.bind(msg_head=f"KaggleDataLoader results({llm})").error(str(e))

            traceback.print_exc()
            results = basic_chat_executor.run(user_intent=term, llm=llm)
            return results["result"]

    tool_dict = {
        "PythonCodeBuilder": Tool(
            name="PythonCodeBuilder",
            func=run_python_code_builder,
            description="""
Description: This tool adeptly turns your textual problem or query into Python code & execute it to get results. It shines when dealing with mathematics, data manipulation tasks, general computational problems and basic visualization like matplotlib. Please note it does not generate database queries.
Input: A natural language problem or question.
Output: A Python program + its execution result to solve the presented problem or answer the question.
Note: The tool MUST be used whenever you want to generate & execute Python code.
                """,
        ),
        "SQLQueryBuilder": Tool(
            name="SQLQueryBuilder",
            func=run_sql_code_builder,
            description="""
Description: Specialized for database tasks, this tool converts your natural language query into SQL code & execute it to get results. It's particularly suited for creating database queries, but it doesn't solve mathematical problems or perform data manipulations outside the SQL context. Be sure to specify the table name for successful operation.
Input: A natural language query related to database operations, along with the name of the table on which the query will operate.
Output: A SQL program, ready to execute on the specified database table, and its execution result.
Note: It is ALWAYS preferable to use the tool whenever you want to generate SQL query & execute the SQL query.
            """,
        ),
        "Echarts": Tool(
            name="Echarts",
            func=run_echarts_interactive_plotter,
            description="""
Description: Dive into the world of data visualization with this specialized Echarts tool. It takes your data table and creates Echarts code & show Echarts for four distinct chart types: scatter, bar, line, and pie, selecting the most appropriate labels and titles.
Input: A natural language query detailing your desired visualization, no other words.
Output: An Echarts script, specifically tailored for your data, that generates an interactive chart upon execution.
Note: Currently, this tool supports only the listed chart types. Please ensure your desired visualization aligns with these options to experience the full capabilities of this dynamic Echarts tool.""",
        ),
        "KaggleDataLoader": Tool(
            name="KaggleDataLoader",
            func=run_kaggle_data_loader,
            description="""
Description: The KaggleDataLoader tool allows you to seamlessly connect to Kaggle datasets. It allows you to load specific datasets by providing the exact dataset path, or it can aid in the search and retrieval of datasets based on the information given in your user input, providing you with a vast array of data sources for your projects.
Input: A natural language intent that may mention path of the Kaggle dataset, or some keyword or other relevant information about the dataset you are interested in.
Output: The action you want to perform, and the extracted path or searched relevant datasets depending on your input.
""",
        ),
    }
    # Data profiling is not activated in agent
    IGNORE_TOOLS = ["DataProfiling"]
    # Activate tools according to the user selection
    tools = [tool_dict[lang["name"]] for lang in code_interpreter_languages]
    for tool in code_interpreter_tools:
        if tool["name"] not in IGNORE_TOOLS:
            tools.append(tool_dict[tool["name"]])

    # Build the chat agent with LLM and tools
    continue_model = llm_name if llm_name in NEED_CONTINUE_MODEL else None
    interaction_executor = initialize_agent(tools, llm, continue_model, memory=memory,
                                            verbose=True)
    return interaction_executor


@app.route("/api/chat", methods=["POST"])
def chat() -> Response | Dict:
    """Returns the chat response of data agent."""
    try:
        # Get request parameters
        request_json = request.get_json()
        user_id = request_json.pop("user_id", DEFAULT_USER_ID)
        chat_id = request_json["chat_id"]
        user_intent = request_json["user_intent"]
        parent_message_id = int(request_json["parent_message_id"])
        code_interpreter_languages = request_json.get("code_interpreter_languages", [])
        code_interpreter_tools = request_json.get("code_interpreter_tools", [])
        api_call = request_json.get("api_call", None)
        llm_name = request_json["llm_name"]
        temperature = request_json.get("temperature", 0.7)
        stop_words = ["[RESPONSE_BEGIN]", "TOOL RESPONSE"]
        kwargs = {
            "temperature": temperature,
            "stop": stop_words,
        }

        # Get language model
        stream_handler = AgentStreamingStdOutCallbackHandler()
        llm = get_llm(llm_name, **kwargs)

        logger.bind(user_id=user_id, chat_id=chat_id, api="/chat",
                    msg_head="Request json").debug(request_json)

        if api_call:
            # Load/init grounding source
            grounding_source_dict = grounding_source_pool.get_pool_info_with_id(user_id,
                                                                                chat_id,
                                                                                default_value={})

            # Find the mainstay message list from leaf to root
            activated_message_list = message_pool.get_activated_message_list(
                user_id, chat_id, default_value=list(),
                parent_message_id=parent_message_id
            )
            assert api_call["api_name"] == "DataProfiling"
            ai_message_id = message_id_register.add_variable("")
            file_node = api_call["args"]["activated_file"]

            folder = create_personal_folder(user_id)
            file_path = _get_file_path_from_node(folder, file_node)
            executor = get_data_summary_cls(file_path)()
            gs = grounding_source_dict[file_path]
            return stream_with_context(
                Response(
                    single_round_chat_with_executor(
                        executor,
                        user_intent=gs,
                        human_message_id=None,
                        ai_message_id=ai_message_id,
                        user_id=DEFAULT_USER_ID,
                        chat_id=api_call["args"]["chat_id"],
                        message_list=activated_message_list,
                        parent_message_id=api_call["args"]["parent_message_id"],
                        llm=llm,
                        app_type="copilot",
                    ),
                    content_type="application/json",
                )
            )
        else:
            # Load/init grounding source
            grounding_source_dict = grounding_source_pool.get_pool_info_with_id(user_id,
                                                                                chat_id,
                                                                                default_value={})
            # Build executor and run chat
            interaction_executor = create_interaction_executor(
                grounding_source_dict=grounding_source_dict,
                code_interpreter_languages=code_interpreter_languages,
                code_interpreter_tools=code_interpreter_tools,
                llm=llm,
                llm_name=llm_name,
                user_id=user_id,
                chat_id=chat_id,
                code_execution_mode=app.config["CODE_EXECUTION_MODE"],
            )
            # Find the mainstay message list from leaf to root
            activated_message_list = message_pool.get_activated_message_list(
                user_id, chat_id, default_value=list(),
                parent_message_id=parent_message_id
            )
            message_pool.load_agent_memory_from_list(interaction_executor.memory,
                                                     activated_message_list)
            human_message_id = message_id_register.add_variable(user_intent)
            ai_message_id = message_id_register.add_variable("")
            return stream_with_context(
                Response(
                    single_round_chat_with_agent_streaming(
                        interaction_executor=interaction_executor,
                        user_intent=user_intent,
                        human_message_id=human_message_id,
                        ai_message_id=ai_message_id,
                        user_id=user_id,
                        chat_id=chat_id,
                        message_list=activated_message_list,
                        parent_message_id=parent_message_id,
                        llm_name=llm_name,
                        stream_handler=stream_handler,
                        app_type="copilot"
                    ),
                    content_type="application/json",
                )
            )

    except Exception as e:
        try:
            logger.bind(user_id=user_id, chat_id=chat_id, api="/chat",
                        msg_head="Chat error").error(str(e))
            import traceback

            traceback.print_exc()
        except:
            # if user_id & chat_id not found, unauth err
            return Response(response=None, status=f"{UNAUTH} Invalid Authentication")
        return Response(response=None,
                        status=f"{OVERLOAD} Server is currently overloaded")
import os

from backend.app import app
from real_agents.adapters.models import ChatOpenAI, ChatAnthropic, AzureChatOpenAI
from real_agents.adapters.llm import BaseLanguageModel

LLAMA_DIR = "PATH_TO_LLAMA_DIR"


@app.route("/api/llm_list", methods=["POST"])
def get_llm_list():
    """Gets the whole llm list."""
    return [
        {"id": llm, "name": llm} for llm in [
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo",
            # "gpt-4",
            "claude-v1",
            "claude-2",
            "lemur-chat"
        ]
    ]


def get_llm(llm_name: str, **kwargs) -> BaseLanguageModel:
    """Gets the llm model by its name."""
    if llm_name in ["gpt-3.5-turbo","gpt-3.5-turbo-16k", "gpt-4"]:
        openai_api_type = os.getenv("OPENAI_API_TYPE", "open_ai")
        if openai_api_type == "open_ai":
            chat_openai = ChatOpenAI
            kwargs.update({"model_name": llm_name})
        elif openai_api_type == "azure":
            chat_openai = AzureChatOpenAI
            kwargs.update({"deployment_name": llm_name})
        return chat_openai(
            streaming=True,
            verbose=True,
            **kwargs
        )
    elif llm_name in ["claude-v1", "claude-2"]:
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        return ChatAnthropic(
            model=llm_name,
            streaming=True,
            verbose=True,
            anthropic_api_key=anthropic_api_key,
            **kwargs,
        )
    elif llm_name == "lemur-chat":
        return ChatOpenAI(
            model_name="lemur-70b-chat-v1",
            streaming=True,
            openai_api_base="https://model-api.xlang.ai/v1",
            verbose=True,
            max_tokens=2048,
            **kwargs
        )
    else:
        raise ValueError(f"llm_name {llm_name} not found")
from flask import request, jsonify, Response

from backend.api.chat_plugin import plugins
from backend.main import app, api_key_pool
from backend.schemas import DEFAULT_USER_ID

@app.route("/api/tool_list", methods=["POST"])
def get_tool_list() -> Response:
    """parameters:
    {
      user_id: id of the user
    }
    return value:
    [{
        id: id of a plugin,
        name: name of a plugin,
        description: description of the plugin,
        icon: icon of the plugin,
        require_api_key: whether the plugin requires api_key,
        api_key: the api key of the plugin, None if no api key
    }]
    """
    user_id = DEFAULT_USER_ID
    api_key_info = api_key_pool.get_pool_info_with_id(user_id, [])
    tool_list = []
    for plugin in plugins:
        plugin_info = {
            "id": plugin["id"],
            "name": plugin["name"],
            "name_for_human": plugin["name_for_human"],
            "description": plugin["description"],
            "icon": plugin["icon"],
            "require_api_key": plugin["require_api_key"],
        }
        search_plugin = [i for i in api_key_info if i["tool_id"] == plugin["id"]]
        if len(search_plugin) > 0:
            plugin_info["api_key"] = search_plugin[0]["api_key"]
        else:
            plugin_info["api_key"] = None
        tool_list.append(plugin_info)
    return jsonify(tool_list)


@app.route("/api/api_key", methods=["POST"])
def post_tool_api_key() -> Response:
    """parameters:
    {
      user_id: id of the user,
      tool_id: id of the tool,
      tool_name: name of the tool,
      api_key: api_key of the tool
    }
    """
    request_json = request.get_json()
    user_id = request_json.pop("user_id", DEFAULT_USER_ID)
    tool_id = request_json["tool_id"]
    tool_name = request_json["tool_name"]
    api_key = request_json["api_key"]
    api_key_info = api_key_pool.get_pool_info_with_id(user_id, [])
    flag = False
    for i in api_key_info:
        if i["tool_id"] == tool_id:
            flag = True
            i["api_key"] = api_key
    if not flag:
        api_key_info.append({"tool_id": tool_id, "tool_name": tool_name, "api_key": api_key})
    api_key_pool.set_pool_info_with_id(user_id, api_key_info)
    return Response("Success", status=200)
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
    {
        "type": "tool",
        "id": "000000000000000000000000000000000000",
        "name": "LinePloter",
        "name_for_human": "LinePloter",
        "pretty_name_for_human": "LinePloter",
        "icon": "",
        "description": "LinePloter",
    },
    {
        "type": "tool",
        "id": "000000000000000000000000000000000001",
        "name": "PiePloter",
        "name_for_human": "PiePloter",
        "pretty_name_for_human": "PiePloter",
        "icon": "",
        "description": "PiePloter",
    },
    {
        "type": "tool",
        "id": "000000000000000000000000000000000002",
        "name": "TopicExtractor",
        "name_for_human": "TopicExtractor",
        "pretty_name_for_human": "TopicExtractor",
        "icon": "",
        "description": "TopicExtractor",
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
from time import sleep
import copy
import redis
import json
import pickle
import traceback
from flask import Response, request, stream_with_context
from typing import Dict, Union
import os

from langchain.schema import HumanMessage, SystemMessage

from backend.api.language_model import get_llm
from backend.main import app, message_id_register, message_pool, logger
from backend.utils.streaming import single_round_chat_with_agent_streaming
from backend.schemas import OVERLOAD, NEED_CONTINUE_MODEL
from backend.schemas import DEFAULT_USER_ID
from real_agents.adapters.llm import BaseLanguageModel
from real_agents.adapters.agent_helpers import AgentExecutor, Tool
from real_agents.adapters.callbacks.agent_streaming import \
    AgentStreamingStdOutCallbackHandler
from real_agents.adapters.models import ChatOpenAI
from real_agents.adapters.memory import ConversationReActBufferMemory
from real_agents.adapters.data_model import DataModel, JsonDataModel
from real_agents.adapters.interactive_executor import initialize_webot_agent
from real_agents.web_agent import WebBrowsingExecutor, WebotExecutor

r = redis.Redis(host=os.getenv("REDIS_SERVER"), port=6379, db=0)  # adjust host/port/db as needed


# here webot and webot_status are stored in redis since the two global variable can not be modified and accessed normally in multiprocess
# fixme:now webot is stored without message_id or chat_id info, so it can only be used for one chat at a time
# fixme:now webot_status is stored with chat_id info, if the status is not reset after a message ended abnormally e.g. the message is interrupted, it will be reused wrongly for the next chat
def get_webot_from_redis(user_id: str, chat_id: str, ) -> WebBrowsingExecutor:
    data = r.get(f'webot_{user_id}_{chat_id}')
    if data is not None:
        webot = pickle.loads(data)
    else:
        # initialize a webot with None instrucition if webot does not exist
        webot = WebBrowsingExecutor(None)
        save_webot_to_redis(user_id, chat_id, webot)
    return webot


def save_webot_to_redis(user_id: str, chat_id: str, webot: WebBrowsingExecutor, ):
    r.set(f'webot_{user_id}_{chat_id}', pickle.dumps(webot))


def get_webot_status_from_redis(user_id: str, chat_id: str):
    webot_status_json = r.get(f'webot_status_{user_id}_{chat_id}')
    if webot_status_json is not None:
        webot_status = json.loads(webot_status_json)
        return webot_status
    else:
        return {}


def save_webot_status_to_redis(user_id: str, chat_id: str, webot_status: Dict):
    r.set(f'webot_status_{user_id}_{chat_id}', json.dumps(webot_status))


def reset_webot(user_id: str, chat_id: str):
    webot = WebBrowsingExecutor(None)
    save_webot_to_redis(user_id, chat_id, webot)


def reset_webot_status(user_id: str, chat_id: str):
    webot_status = {"webot_status": "idle", "url": None}
    save_webot_status_to_redis(user_id, chat_id, webot_status)


# this function has been deprecated
def get_plan(instruction: str, start_url: str, chat_llm: ChatOpenAI):
    # fixme: Move this into a separate chain or executors to decompose the LLMs
    system_message = f"""
You are a planner to assist another browser automation assistant.

Here is the instruction for the other assistant:
```
You MUST take one of the following actions. NEVER EVER EVER make up actions that do not exist:

1. click(element): Clicks on an element
2. setValue(element, value: string): Focuses on and sets the value of an input element
3. finish(): Indicates the task is finished
4. fail(): Indicates that you are unable to complete the task
You will be be given a task to perform and the current state of the DOM. You will also be given previous actions that you have taken. You may retry a failed action up to one time.

This is an example of an action:

<Thought>I should click the add to cart button</Thought>
<Action>click(223)</Action>

You MUST always include the <Thought> and <Action> open/close tags or else your response will be marked as invalid.

Rules you MUST follow:
1. You must only take one step at a time. You cannot take multiple actions in a single response.
2. You should not consider the action to present the result to the user. You only need to do available actions. If info in current page is enough for the user to solve the problem, you should finish.
```
Now your responsibility is to give a step-by-step plan according to user's instruction. This plan will be given to the assistant as a reference when it is performing tasks.
""".strip()

    human_message = f"""
The user requests the following task:

{instruction}

Now you are at {start_url}

Provide a plan to do this (you can use pseudo description as below to describe the item).

Here is an example case:

request: Go to google calendar to schedule a meeting

current url: "https://google.com"

example plan:

1. setValue(searchBar, "google calendar")
2. click(search)
3. click(the item with title of google calendar)
4.1 if user has loginned 
    do nothing 
4.2 if user hasn't loginned 
    do login 
5. click(create event button) 
6. setValue(event title input bar, "meeting") 
7. click(save event button)
8. finish()
""".strip()

    messages = [SystemMessage(content=system_message),
                HumanMessage(content=human_message)]
    response = chat_llm(messages).content
    return response


def create_webot_interaction_executor(
        llm: BaseLanguageModel,
        llm_name: str,
        user_id: str,
        chat_id: str
) -> AgentExecutor:
    """Creates an agent executor for interaction.

    Args:
        llm: A llm model.
        llm_name: A string llm name.
        user_id: A string of user id.
        chat_id: A string chat id.

    Returns:
        An agent executor.

    """
    # Initialize memory
    memory = ConversationReActBufferMemory(memory_key="chat_history",
                                           return_messages=True, max_token_limit=10000)

    class RunWebot:
        def __init__(self, webot: WebotExecutor, llm: BaseLanguageModel, user_id: str,
                     chat_id: str):
            self.llm = llm
            self.webot = webot
            self.user_id = user_id
            self.chat_id = chat_id

        def run(self, term: str) -> Union[str, Dict, DataModel]:
            try:
                user_id = self.user_id
                chat_id = self.chat_id
                reset_webot(user_id=user_id, chat_id=chat_id)
                reset_webot_status(user_id=user_id, chat_id=chat_id)
                raw_observation = self.webot.run(user_intent=term, llm=self.llm)
                instruction, start_url = raw_observation["instruction"], \
                    raw_observation["start_url"]
                webot = get_webot_from_redis(user_id=user_id, chat_id=chat_id)
                webot.instruction = instruction
                # webot.plan = get_plan(instruction, start_url)
                webot.plan = ""
                save_webot_to_redis(user_id=user_id, chat_id=chat_id, webot=webot)
                webot_status = {
                    "webot_status": "running",
                    "url": start_url
                }
                save_webot_status_to_redis(user_id=user_id, chat_id=chat_id,
                                           webot_status=webot_status)
                while True:
                    webot = get_webot_from_redis(user_id=user_id, chat_id=chat_id)
                    if webot.finish or webot.interrupt or webot.error or webot.fail:
                        break
                    else:
                        sleep(0.5)
                save_webot_status_to_redis(user_id=user_id, chat_id=chat_id,
                                           webot_status={"webot_status": "idle",
                                                         "url": None})
                webot = get_webot_from_redis(user_id=user_id, chat_id=chat_id)
                webot.instruction = None
                save_webot_to_redis(user_id=user_id, chat_id=chat_id, webot=webot)

                if webot.finish:
                    webot = get_webot_from_redis(user_id=user_id, chat_id=chat_id)
                    action_history = webot.action_history
                    last_page = webot.pages_viewed[-1]
                    observation = JsonDataModel.from_raw_data(
                        {
                            "success": True,
                            "result": json.dumps({"action_history": action_history,
                                                  "last_page": last_page}, indent=4),
                            "intermediate_steps": json.dumps(
                                {"instruction": instruction, "start_url": start_url},
                                indent=4)
                        }
                    )
                    return observation

                if webot.fail:
                    observation = JsonDataModel.from_raw_data(
                        {
                            "success": True,
                            "result": "The webot failed to execute the instruction.",
                            "intermediate_steps": json.dumps(
                                {"instruction": instruction, "start_url": start_url},
                                indent=4)
                        }
                    )
                    return observation

                if webot.interrupt:
                    observation = JsonDataModel.from_raw_data(
                        {
                            "success": False,
                            "result": "The web browsing is interrupted by user.",
                            "intermediate_steps": json.dumps(
                                {"instruction": instruction, "start_url": start_url},
                                indent=4)
                        }
                    )
                    return observation

                if webot.error:
                    observation = JsonDataModel.from_raw_data(
                        {
                            "success": False,
                            "result": "Error occurs during web browsing.",
                            "intermediate_steps": json.dumps(
                                {"instruction": instruction, "start_url": start_url},
                                indent=4)
                        }
                    )
                    return observation

            except Exception as e:
                print(traceback.format_exc())
                observation = JsonDataModel.from_raw_data(
                    {
                        "success": False,
                        "result": f"Failed in web browsing with the input: {term}, please try again later.",
                        "intermediate_steps": json.dumps({"error": str(e)})
                    }
                )
                return observation

    webot = WebotExecutor.from_webot()
    llm = copy.deepcopy(llm)
    run_webot = RunWebot(webot, llm, chat_id=chat_id, user_id=user_id)
    tools = [Tool(name=webot.name, func=run_webot.run, description=webot.description)]

    continue_model = llm_name if llm_name in NEED_CONTINUE_MODEL else None
    interaction_executor = initialize_webot_agent(
        tools, llm, continue_model, memory=memory, verbose=True
    )
    return interaction_executor


@app.route("/api/chat_xlang_webot", methods=["POST"])
def chat_xlang_webot() -> Dict:
    """Returns the chat response of web agent."""
    try:
        # Get request parameters
        request_json = request.get_json()
        user_id = request_json.pop("user_id", DEFAULT_USER_ID)
        chat_id = request_json["chat_id"]
        user_intent = request_json["user_intent"]
        parent_message_id = request_json["parent_message_id"]
        llm_name = request_json["llm_name"]
        temperature = request_json.get("temperature", 0.4)
        stop_words = ["[RESPONSE_BEGIN]", "TOOL RESPONSE"]
        kwargs = {
            "temperature": temperature,
            "stop": stop_words,
        }

        # Get language model
        llm = get_llm(llm_name, **kwargs)

        logger.bind(user_id=user_id, chat_id=chat_id, api="/chat",
                    msg_head="Request json").debug(request_json)

        human_message_id = message_id_register.add_variable(user_intent)
        ai_message_id = message_id_register.add_variable("")

        stream_handler = AgentStreamingStdOutCallbackHandler()
        # Build executor and run chat

        # reset webot and status
        reset_webot(user_id=user_id, chat_id=chat_id)
        reset_webot_status(user_id=user_id, chat_id=chat_id)

        interaction_executor = create_webot_interaction_executor(
            llm=llm,
            llm_name=llm_name,
            chat_id=chat_id,
            user_id=user_id
        )

        activated_message_list = message_pool.get_activated_message_list(user_id,
                                                                         chat_id,
                                                                         list(),
                                                                         parent_message_id)
        message_pool.load_agent_memory_from_list(interaction_executor.memory,
                                                 activated_message_list)
        return stream_with_context(
            Response(
                single_round_chat_with_agent_streaming(
                    interaction_executor=interaction_executor,
                    user_intent=user_intent,
                    human_message_id=human_message_id,
                    ai_message_id=ai_message_id,
                    user_id=user_id,
                    chat_id=chat_id,
                    message_list=activated_message_list,
                    parent_message_id=parent_message_id,
                    stream_handler=stream_handler,
                    llm_name=llm_name,
                    app_type="webot",
                ),
                content_type="application/json",
            )
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return Response(response=None,
                        status=f"{OVERLOAD} backend is currently overloaded")
from typing import Dict
from flask import request, jsonify, Response

from backend.main import message_pool
from backend.app import app
from backend.api.language_model import get_llm
from backend.utils.utils import get_user_and_chat_id_from_request_json
from real_agents.adapters.executors import QuestionSuggestionExecutor
from real_agents.adapters.memory import ConversationReActBufferMemory


@app.route("/api/recommend", methods=["POST"])
def recommend() -> dict | Response:
    """Recommends potential inputs for users. """
    try:
        request_json = request.get_json()
        (user_id, chat_id) = get_user_and_chat_id_from_request_json(request_json)
        parent_message_id = int(request_json["parent_message_id"])
        user_intent = request_json["user_intent"]

        # Find the mainstat message list from leaf to root
        activated_message_list = message_pool.get_activated_message_list(
            user_id, chat_id, default_value=list(), parent_message_id=parent_message_id
        )
        chat_memory = ConversationReActBufferMemory(memory_key="chat_history", return_messages=True)
        message_pool.load_agent_memory_from_list(chat_memory, activated_message_list)
        question_suggestion_executor = QuestionSuggestionExecutor()
        
        llm_name = request_json["llm_name"]
        temperature = request_json.get("temperature", 0.7)
        kwargs = {
            "temperature": temperature,
        }

        # Get language model
        llm = get_llm(llm_name, **kwargs)
        follow_questions = question_suggestion_executor.run(
            user_intent=user_intent,
            llm=llm,
            chat_memory=chat_memory,
            mode="chat_memory",
        )

        return jsonify({
            "recommend_questions": follow_questions["questions"], 
            "user_id": user_id,
            "chat_id": chat_id,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "recommend_questions": [],
            "user_id": user_id,
            "chat_id": chat_id,
        }
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
from flask import request, jsonify, Response

from backend.main import app
from backend.schemas import DEFAULT_USER_ID
from backend.api.chat_webot import get_webot_from_redis, \
    get_webot_status_from_redis, reset_webot_status


@app.route("/api/webot/instructions", methods=["POST"])
def get_instruction() -> Response:
    request_json = request.get_json()
    user_id = request_json.pop("user_id", DEFAULT_USER_ID)
    chat_id = request_json["chat_id"]
    webot = get_webot_from_redis(user_id=user_id, chat_id=chat_id)
    return jsonify({
        "chat_id": chat_id,
        "user_id": user_id,
        "instructions": webot.instruction
    })


@app.route("/api/webot/webot_status", methods=["POST"])
def get_webot_status() -> Response:
    request_json = request.get_json()
    user_id = request_json.pop("user_id", DEFAULT_USER_ID)
    chat_id = request_json["chat_id"]
    webot_status_json = get_webot_status_from_redis(user_id=user_id, chat_id=chat_id)
    return jsonify(webot_status_json) if webot_status_json is not None else jsonify(
        {"webot_status": None, "url": None})


@app.route("/api/webot/reset_status", methods=["POST"])
def reset_status() -> Response:
    request_json = request.get_json()
    user_id = request_json.pop("user_id", DEFAULT_USER_ID)
    chat_id = request_json["chat_id"]
    reset_webot_status(user_id=user_id, chat_id=chat_id)
    return jsonify({
        "chat_id": chat_id,
        "user_id": user_id,
    })
import base64
import copy
import json
import os
import random
import traceback
from typing import Dict, List, Union

import requests
from flask import Response, request, stream_with_context
from retrying import retry

from backend.api.language_model import get_llm
from backend.app import app
from backend.main import message_id_register, message_pool, logger
from backend.utils.streaming import single_round_chat_with_agent_streaming
from backend.schemas import OVERLOAD, NEED_CONTINUE_MODEL, DEFAULT_USER_ID
from backend.main import api_key_pool
from real_agents.adapters.llm import BaseLanguageModel
from real_agents.adapters.agent_helpers import AgentExecutor, Tool
from real_agents.adapters.callbacks.agent_streaming import \
    AgentStreamingStdOutCallbackHandler
from real_agents.adapters.data_model import DataModel, JsonDataModel
from real_agents.adapters.interactive_executor import initialize_plugin_agent
from real_agents.adapters.memory import ConversationReActBufferMemory
from real_agents.plugins_agent.plugins.utils import load_all_plugins_elements
from real_agents.plugins_agent.plugins.tool_selector import ToolSelector
from real_agents.plugins_agent import PluginExecutor

# The plugins list
global plugins
plugins = []

# Set up the tool selector for automatically selecting plugins
try:
    tool_selector = ToolSelector(tools_list=plugins, mode="embedding", api_key_pool=api_key_pool)
except Exception as e:
    print(e, "The auto selection feature of plugins agent will return random elements.")
    tool_selector = None

# Load plugin info and icon image
for plugin_type, plugin_info in load_all_plugins_elements().items():
    @retry(stop_max_attempt_number=10,
           wait_fixed=2000)  # Retry 3 times with a 2-second delay between retries
    def make_request(_image_url) -> Response:
        response = requests.get(_image_url)  # Replace with your actual request code
        response.raise_for_status()  # Raise an exception for unsuccessful response status codes
        return response


    # Load icon image
    image_url = plugin_info["meta_info"]["manifest"]["logo_url"]

    # If image is base64 encoded
    if image_url.startswith("data:image"):
        plugins.append(
            {
                "id": plugin_type,
                "name": plugin_type,
                "name_for_human": plugin_info["meta_info"]["manifest"][
                    "name_for_human"],
                "description": plugin_info["description"],
                "icon": image_url,
                "require_api_key": plugin_info["need_auth"],
            }
        )
        continue

    image_format = image_url.split(".")[-1]

    try:
        # Check if in cache
        os.makedirs("backend/static/images", exist_ok=True)
        if os.path.exists(f"backend/static/images/{plugin_type}.cache"):
            with open(f"backend/static/images/{plugin_type}.cache", "rb") as f:
                image_content = f.read()
        else:
            response = make_request(image_url)
            image_content = response.content
            # Make a .cache file for the image
            with open(f"backend/static/images/{plugin_type}.cache", "wb") as f:
                f.write(image_content)
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to make the request {plugin_type}:", e)
        continue

    if image_format == "svg":
        encoded_image = "data:image/svg+xml;base64,  ".format(
            image_format) + base64.b64encode(image_content).decode(
            "utf-8"
        )
    else:
        encoded_image = "data:image/{};base64,  ".format(
            image_format) + base64.b64encode(image_content).decode(
            "utf-8"
        )

    plugins.append(
        {
            "id": plugin_type,
            "name": plugin_type,
            "name_for_human": plugin_info["meta_info"]["manifest"]["name_for_human"],
            "description": plugin_info["description"],
            "icon": encoded_image,
            "require_api_key": plugin_info["need_auth"],
        }
    )


def create_plugins_interaction_executor(
        selected_plugins: List[str],
        api_key_info: List[Dict],
        llm: BaseLanguageModel,
        llm_name: str,
) -> AgentExecutor:
    """Creates an agent executor for interaction.

    Args:
        selected_plugins: A list of selected plugins.
        api_key_info: A list of plugin api keys.
        llm: A llm model.
        llm_name: A string llm name.

    Returns:
        An agent executor.

    """
    # Initialize memory
    memory = ConversationReActBufferMemory(memory_key="chat_history",
                                           return_messages=True, style="plugin",
                                           max_token_limit=10000)

    class RunPlugin:
        def __init__(self, plugin: PluginExecutor, llm: BaseLanguageModel):
            self.plugin = plugin
            self.llm = llm

        def run(self, term: str) -> Union[str, Dict, DataModel]:
            try:
                raw_observation = self.plugin.run(user_intent=term, llm=self.llm)
                input_json, output = raw_observation["input_json"], raw_observation[
                    "api_output"]
                observation = JsonDataModel.from_raw_data(
                    {
                        "success": True,
                        "result": json.dumps(output, indent=4) if isinstance(output,
                                                                             dict) else output,
                        "intermediate_steps": json.dumps(input_json, indent=4),
                    }
                )
                return observation

            except Exception as e:
                observation = JsonDataModel.from_raw_data(
                    {
                        "success": False,
                        "result": str(e),
                    }
                )
                print(traceback.format_exc())
                return observation

    # Load plugins from selected names
    _plugins = []
    for selected_plugin in selected_plugins:
        plugin = PluginExecutor.from_plugin_name(selected_plugin)
        llm = copy.deepcopy(llm)

        if len([i for i in api_key_info if i["tool_name"] == plugin.name]) != 0:
            plugin.api_key = \
            [i for i in api_key_info if i["tool_name"] == plugin.name][0]["api_key"]
            # For some plugins, we need to reload the plugin to update personal data
            plugin.load_personnel_info()  # warning: this will change the plugin object every time we make a new query

        run_plugin = RunPlugin(plugin, llm)

        _plugins.append(Tool(name=plugin.name, func=run_plugin.run,
                             description=plugin.full_description))

    continue_model = llm_name if llm_name in NEED_CONTINUE_MODEL else None
    interaction_executor = initialize_plugin_agent(
        _plugins, llm, continue_model, memory=memory, verbose=True
    )

    return interaction_executor


@app.route("/api/chat_xlang_plugin", methods=["POST"])
def chat_xlang_plugin() -> Dict:
    """Returns the chat response of plugins agent."""
    try:
        # Get request parameters
        request_json = request.get_json()
        user_id = request_json.pop("user_id", DEFAULT_USER_ID)
        chat_id = request_json["chat_id"]
        user_intent = request_json["user_intent"]
        parent_message_id = request_json["parent_message_id"]
        selected_plugins = request_json["selected_plugins"]
        llm_name = request_json["llm_name"]
        temperature = request_json.get("temperature", 0.4)
        stop_words = ["[RESPONSE_BEGIN]", "TOOL RESPONSE"]
        kwargs = {
            "temperature": temperature,
            "stop": stop_words,
        }

        # pass user id and chat id to tool selector
        if tool_selector:
            tool_selector.user_id = user_id
            tool_selector.chat_id = chat_id

        # Get language model
        llm = get_llm(llm_name, **kwargs)

        logger.bind(user_id=user_id, chat_id=chat_id, api="/chat",
                    msg_head="Request json").debug(request_json)

        # Get API key for plugins
        api_key_info = api_key_pool.get_pool_info_with_id(user_id,
                                                          default_value=[])  # fixme: mock user_id: 1

        activated_message_list = message_pool.get_activated_message_list(user_id,
                                                                         chat_id,
                                                                         list(),
                                                                         parent_message_id)

        # Flag for auto retrieving plugins
        if len(selected_plugins) == 1 and selected_plugins[0].lower() == "auto":

            if tool_selector:
                # this will return a list of plugin names sorted by relevance (lower case and the same as their dir name)
                query = tool_selector.load_query_from_message_list(activated_message_list,
                                                                   user_intent)
                selected_plugins = tool_selector.select_tools(query=query, top_k=5)
            else:
                selected_plugins = [_plugin['id'] for _plugin in random.sample(plugins, 5)]

        # Build executor and run chat
        stream_handler = AgentStreamingStdOutCallbackHandler()
        interaction_executor = create_plugins_interaction_executor(
            selected_plugins=selected_plugins,
            api_key_info=api_key_info,
            llm=llm,
            llm_name=llm_name,
        )

        message_pool.load_agent_memory_from_list(interaction_executor.memory,
                                                 activated_message_list)

        human_message_id = message_id_register.add_variable(user_intent)
        ai_message_id = message_id_register.add_variable("")

        return stream_with_context(
            Response(
                single_round_chat_with_agent_streaming(
                    interaction_executor=interaction_executor,
                    user_intent=user_intent,
                    human_message_id=human_message_id,
                    ai_message_id=ai_message_id,
                    user_id=user_id,
                    chat_id=chat_id,
                    message_list=activated_message_list,
                    parent_message_id=parent_message_id,
                    stream_handler=stream_handler,
                    llm_name=llm_name,
                    app_type="plugins",
                ),
                content_type="application/json",
            )
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return Response(response=None,
                        status=f"{OVERLOAD} backend is currently overloaded")
from flask import request, jsonify, Response

from backend.api.chat_webot import get_webot_from_redis, save_webot_to_redis
from backend.main import app
from backend.schemas import DEFAULT_USER_ID
from backend.api.language_model import get_llm


@app.route("/api/webot/action", methods=["POST"])
def get_action() -> Response:
    """Gets the next action to take for a given the current page HTML."""
    request_json = request.get_json()
    user_id = request_json.pop("user_id", DEFAULT_USER_ID)
    chat_id = request_json["chat_id"]
    webot = get_webot_from_redis(user_id=user_id, chat_id=chat_id)
    # Get request parameters
    request_json = request.get_json()
    processed_html = request_json["processed_html"]
    llm = get_llm("gpt-4")
    result = webot.run(processed_html, llm=llm)
    save_webot_to_redis(user_id=user_id, chat_id=chat_id, webot=webot)

    return jsonify({
        "chat_id": chat_id,
        "user_id": user_id,
        "action_response": result,
    })


@app.route("/api/webot/interrupt", methods=["POST"])
def interrupt() -> Response:
    """Interrupts the current webot."""
    request_json = request.get_json()
    user_id = request_json.pop("user_id", DEFAULT_USER_ID)
    chat_id = request_json["chat_id"]
    interrupt = request_json["interrupt"]
    if interrupt:
        webot = get_webot_from_redis(user_id=user_id, chat_id=chat_id)
        webot.actions_taken.append("interrupt")
        save_webot_to_redis(user_id=user_id, chat_id=chat_id, webot=webot)
        return jsonify({
            "chat_id": chat_id,
            "user_id": user_id,
        })
    return jsonify({"message": "No interrupt signal received."})


@app.route("/api/webot/error", methods=["POST"])
def error() -> Response:
    """Appends action 'error' to the current webot."""
    request_json = request.get_json()
    user_id = request_json.pop("user_id", DEFAULT_USER_ID)
    chat_id = request_json["chat_id"]
    error = request_json["error"]
    if error:
        webot = get_webot_from_redis(user_id=user_id, chat_id=chat_id)
        webot.actions_taken.append("error")
        save_webot_to_redis(user_id=user_id, chat_id=chat_id, webot=webot)
        return jsonify({
            "chat_id": chat_id,
            "user_id": user_id,
        })
    return jsonify({"message": "No error signal received."})
import json
import os
import shutil
from typing import Dict, Any
from flask import Response, jsonify, request, send_file

from backend.app import app
from backend.main import (
    grounding_source_pool,
    logger,
    message_id_register,
    message_pool,
)
from backend.schemas import DEFAULT_USER_ID
from backend.utils.utils import create_personal_folder
from backend.utils.user_conversation_storage import get_user_conversation_storage
from backend.utils.utils import (
    allowed_file,
    get_data_model_cls,
    get_user_and_chat_id_from_request,
    get_user_and_chat_id_from_request_json,
    is_sqlite_file,
    is_table_file,
    is_image_file,
    load_grounding_source,
)
from backend.schemas import INTERNAL, UNFOUND

TABLE_HUMAN_SIDE_FORMAT = "material-react-table"


def _path_tree_for_react_dnd_treeview(tree: list, id_to_path_dict: dict, path: str,
                                      parent: int,
                                      highlighted_files: list = []) -> list:
    """
    {
        "id": 1,
        "parent": 0,
        "droppable": true,
        "text": "Folder 1"
    },
    {
        "id": 2,
        "parent": 1,
        "text": "File 1-1",
        "data": {
            "fileType": "csv",
            "fileSize": "0.5MB"
        }
    },
    """
    for item in os.listdir(path):
        if item.startswith("."):
            continue
        item_path = os.path.join(path, item)
        droppable = os.path.isdir(item_path)
        idx = len(tree) + 1
        tree.append({
            "id": idx,
            "parent": parent,
            "droppable": droppable,
            "text": item,
            "highlight": True if item_path in highlighted_files else False})
        id_to_path_dict[idx] = item_path
        if os.path.isdir(item_path):
            _path_tree_for_react_dnd_treeview(tree, id_to_path_dict, item_path, idx)

    return []

def secure_filename(filename: str) -> str:
    keep_characters = ('.', '_')
    filename = "".join(
        c for c in filename if c.isalnum() or c in keep_characters).rstrip()
    return filename


@app.route("/api/upload", methods=["POST"])
def create_upload_file() -> dict | Response:
    """Uploads a new file."""
    try:
        if "file" not in request.files:
            return {"error": "No file part in the request"}
        file = request.files["file"]
        (user_id, chat_id) = get_user_and_chat_id_from_request(request)
        folder = create_personal_folder(user_id)

        # Check if the file is allowed
        if not allowed_file(str(file.filename)):
            return {"error": "File type not allowed"}

        # Save and read the file
        file.filename = secure_filename(str(file.filename))
        file_path = os.path.join(folder, file.filename)
        file.save(file_path)
        response = {"success": file.filename}

        logger.bind(user_id=user_id, chat_id=chat_id, api="/upload",
                    msg_head="Upload file success").debug(file_path)

        return jsonify(response)
    except Exception as e:
        logger.bind(user_id=user_id, chat_id=chat_id, api="/upload",
                    msg_head="Upload file error").error(str(e))

        return Response(response=None, status=f"{INTERNAL} Upload File Error: {str(e)}")


def _generate_human_side_data_from_file(filename: str, data_model: Any) -> Dict:
    if is_table_file(filename):
        # Determine the format of the human side(frontend) table
        human_side_data = data_model.get_human_side_data(mode="FULL")
        if TABLE_HUMAN_SIDE_FORMAT == "markdown":
            human_side_data = human_side_data.to_markdown(index=False)
            human_side_data_type = "plain"
        elif TABLE_HUMAN_SIDE_FORMAT == "material-react-table":
            columns = list(map(lambda item: {"accessorKey": item, "header": item},
                               human_side_data.columns.tolist()))
            data = human_side_data.fillna("").to_dict(orient="records")
            human_side_data = json.dumps({"columns": columns, "data": data})
            human_side_data_type = "table"
        data = {"success": filename, "content": human_side_data,
                "type": human_side_data_type}
    elif is_sqlite_file(filename):
        data = {"success": filename, "content": filename, "type": "table"}
    elif is_image_file(filename):
        # Determine the format of human side(frontend) image
        human_side_data = data_model.get_human_side_data()
        data = {"success": filename, "content": human_side_data, "type": "image"}
    else:
        return {"error": "Document file type not supported"}
    return data


def _get_file_path_from_node(folder: str, file_node: dict) -> Any:
    path_tree_list: list = []
    id_to_path_dict = {0: folder}
    _path_tree_for_react_dnd_treeview(path_tree_list, id_to_path_dict, folder, 0)
    path = id_to_path_dict[file_node["id"]]
    return path


@app.route("/api/file_system/apply", methods=["POST"])
def apply_to_conversation() -> Response:
    """Applies data to the conversation."""
    try:
        request_json = request.get_json()
        (user_id, chat_id) = get_user_and_chat_id_from_request_json(request_json)
        file_node = request_json["activated_file"]
        parent_message_id = request_json["parent_message_id"]
        folder = create_personal_folder(user_id)

        # Modify the selected grounding sources
        grounding_source_dict = grounding_source_pool.get_pool_info_with_id(user_id,
                                                                            chat_id,
                                                                            default_value={})
        file_path = _get_file_path_from_node(folder, file_node)
        filename = file_node["text"]
        filename_no_ext = os.path.splitext(filename)[0]
        if file_path not in grounding_source_dict:
            data = load_grounding_source(file_path)
            data_model = get_data_model_cls(filename).from_raw_data(
                raw_data=data,
                raw_data_name=filename_no_ext,
                raw_data_path=file_path,
            )
            grounding_source_dict[file_path] = data_model
            # Add uploaded file in chat memory
            message_list = message_pool.get_pool_info_with_id(user_id, chat_id,
                                                              default_value=list())
            llm_side_data = data_model.get_llm_side_data()
            human_message_content = "[User uploaded a file {}]\n{}".format(filename,
                                                                           llm_side_data)
            human_message_id = message_id_register.add_variable(human_message_content)
            message_list.append(
                {
                    "message_id": human_message_id,
                    "parent_message_id": parent_message_id,
                    "message_type": "human_message",
                    "message_content": human_message_content,
                }
            )
            data = _generate_human_side_data_from_file(filename, data_model)
            message_pool.set_pool_info_with_id(user_id, chat_id, message_list)
            grounding_source_pool.set_pool_info_with_id(user_id, chat_id,
                                                        grounding_source_dict)
            # Dump to database
            db = get_user_conversation_storage()
            db_message = {
                "conversation_id": chat_id,
                "user_id": user_id,
                "message_id": human_message_id,
                "parent_message_id": parent_message_id,
                "version_id": 0,
                "role": "user",
                "data_for_human": {
                    "intermediate_steps": [],
                    "final_answer": [
                        {
                            "type": data["type"],
                            "text": data["content"],
                            "final": True,
                        }
                    ],
                },
                "data_for_llm": message_list[-1]["message_content"],
                "raw_data": None,
            }
            db.message.insert_one(db_message)
            response = {
                "success": True,
                "message_id": human_message_id,
                "parent_message_id": parent_message_id,
                "message": "Successfully apply {} to conversation {}".format(filename,
                                                                             chat_id),
                "content": {
                    "intermediate_steps": [],
                    "final_answer": [
                        {
                            "type": data["type"],
                            "text": data["content"],
                            "final": True,
                        }
                    ],
                },
            }

            logger.bind(user_id=user_id, chat_id=chat_id, api="/apply",
                        msg_head="Apply file success").debug(file_path)
            del db_message["data_for_human"]

            return jsonify(response)
        else:
            logger.bind(user_id=user_id, chat_id=chat_id, api="/apply",
                        msg_head="Apply file failed").debug(file_path)

            return jsonify({"success": False,
                            "message": "You have already import {} to the conversation".format(
                                filename)})
    except Exception as e:
        logger.bind(user_id=user_id, chat_id=chat_id, api="/apply",
                    msg_head="Apply file failed").error(file_path)
        import traceback
        traceback.print_exc()

        return Response(response=None,
                        status=f"{INTERNAL} Fail to apply file to chat: {str(e)}")


@app.route("/api/file_system/move", methods=["POST"])
def move_files() -> Response:
    """Moves file from source path from target source."""
    request_json = request.get_json()
    (user_id, chat_id) = get_user_and_chat_id_from_request_json(request_json)
    root_path = create_personal_folder(user_id)
    nodes = request_json["nodes"]
    try:
        if os.path.exists(root_path) and os.path.isdir(root_path):
            current_path_tree_list: list = []
            id_to_path_dict = {0: root_path}
            _path_tree_for_react_dnd_treeview(current_path_tree_list, id_to_path_dict,
                                              root_path, 0)
        for node in nodes:
            old_path = id_to_path_dict[node["id"]]
            new_path = id_to_path_dict[node["parent"]]
            shutil.move(old_path, new_path)

            logger.bind(user_id=user_id, chat_id=chat_id, api="/move",
                        msg_head="Move file success").debug(
                f"from {old_path} to {new_path}"
            )

            return jsonify({"success": True, "message": "File moved successfully"})
    except Exception as e:
        logger.bind(user_id=user_id, chat_id=chat_id, api="/move",
                    msg_head="Move file failed").error(str(e))

        return jsonify({"success": False, "message": str(e)})
    return Response(response=None, status=f"{INTERNAL} Fail to move file")


@app.route("/api/file_system/delete", methods=["POST"])
def delete_files() -> Response:
    """Deletes a file from the filesystem."""
    request_json = request.get_json()
    (user_id, chat_id) = get_user_and_chat_id_from_request_json(request_json)
    root_path = create_personal_folder(user_id)
    node = request_json["node"]
    try:
        if os.path.exists(root_path) and os.path.isdir(root_path):
            current_path_tree_list: list = []
            id_to_path_dict = {0: root_path}
            _path_tree_for_react_dnd_treeview(current_path_tree_list, id_to_path_dict,
                                              root_path, 0)
            path = id_to_path_dict[node["id"]]
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

        logger.bind(user_id=user_id, chat_id=chat_id, api="/delete",
                    msg_head="Delete file success").debug(path)

        return jsonify({"success": True, "message": "File is deleted successfully"})
    except Exception as e:
        logger.bind(user_id=user_id, chat_id=chat_id, api="/delete",
                    msg_head="Delete file failed").error(str(e))

        return Response(response=None,
                        status=f"{INTERNAL} Delete file failed: {str(e)}")


@app.route("/api/file_system/download", methods=["POST"])
def download_files() -> Response:
    """Downloads a file to local."""
    request_json = request.get_json()
    user_id = request_json.pop("user_id", DEFAULT_USER_ID)
    root_path = create_personal_folder(user_id)
    node = request_json["node"]

    try:
        if os.path.exists(root_path) and os.path.isdir(root_path):
            current_path_tree_list: list = []
            id_to_path_dict = {0: root_path}
            _path_tree_for_react_dnd_treeview(current_path_tree_list, id_to_path_dict,
                                              root_path, 0)
            path = id_to_path_dict[node["id"]]

            if os.path.exists(path):
                logger.bind(user_id=user_id, api="/download",
                            msg_head="download file success").debug(path)
                return send_file(path, as_attachment=True)

        logger.bind(user_id=user_id, api="/download",
                    msg_head="download file failed").debug(path)
        return Response(response=None,
                        status=f"{INTERNAL} Download file failed: file not correctlt sent")

    except Exception as e:
        print(str(e))
        import traceback
        traceback.print_exc()

        logger.bind(user_id=user_id, api="/download",
                    msg_head="download file failed").error(str(e))

        return Response(response=None,
                        status=f"{INTERNAL} Download file failed: {str(e)}")


def _generate_directory_name(name: str, x:int=0) -> Any:
    dir_name = (name + ("_" + str(x) if x != 0 else "")).strip()
    if not os.path.exists(dir_name):
        return dir_name
    else:
        return _generate_directory_name(name, x + 1)


@app.route("/api/file_system/create_folder", methods=["POST"])
def create_folders() -> Response:
    """Creates a folder in the filesystem."""
    request_json = request.get_json()
    (user_id, chat_id) = get_user_and_chat_id_from_request_json(request_json)
    root_path = create_personal_folder(user_id)
    if os.path.exists(root_path) and os.path.isdir(root_path):
        try:
            new_path = _generate_directory_name(os.path.join(root_path, "Folder"))
            os.makedirs(new_path, exist_ok=False)

            logger.bind(
                user_id=user_id, chat_id=chat_id, api="/create_folder",
                msg_head="Create folder success"
            ).debug(new_path)

            return jsonify({"success": True, "message": "Folder created successfully"})
        except Exception as e:
            logger.bind(user_id=user_id, chat_id=chat_id, api="/create_folder",
                        msg_head="Create folder failed").error(
                str(e)
            )

            return jsonify({"success": False, "message": str(e)})
    else:
        logger.bind(user_id=user_id, chat_id=chat_id, api="/create_folder",
                    msg_head="Create folder failed").error(
            "Root path does not exist."
        )

        return Response(response=None, status=f"{INTERNAL} Root path does not exist")


@app.route("/api/file_system/update", methods=["POST"])
def rename_folder() -> Response:
    """Renames a folder in the filesystem."""
    request_json = request.get_json()
    (user_id, chat_id) = get_user_and_chat_id_from_request_json(request_json)
    root_path = create_personal_folder(user_id)
    node = request_json["node"]
    rename_value = request_json["rename_value"]
    if os.path.exists(root_path) and os.path.isdir(root_path):
        try:
            current_path_tree_list: list = []
            id_to_path_dict = {0: root_path}
            _path_tree_for_react_dnd_treeview(current_path_tree_list, id_to_path_dict,
                                              root_path, 0)
            path = id_to_path_dict[node["id"]]
            new_path = os.path.join(os.path.dirname(path), rename_value)
            shutil.move(path, new_path)

            logger.bind(user_id=user_id, chat_id=chat_id, api="/update",
                        msg_head="Rename folder success").debug(
                f"{path} to {new_path}"
            )

            return jsonify({"success": True, "message": "Folder created successfully"})
        except Exception as e:
            logger.bind(user_id=user_id, chat_id=chat_id, api="/update",
                        msg_head="Rename folder failed").error(str(e))

            return jsonify({"success": False, "message": str(e)})
    else:
        logger.bind(user_id=user_id, chat_id=chat_id, api="/update",
                    msg_head="Rename folder failed").error(
            "Root path does not exist."
        )

        return Response(response=None, status=f"{INTERNAL} Root path does not exist")


@app.route("/api/file_system/get_path_tree", methods=["POST"])
def get_path_tree() -> Response:
    """Gets a file path tree of one file."""
    try:
        request_json = request.get_json()
        user_id = request_json.pop("user_id", DEFAULT_USER_ID)
        if user_id == "":  # front-end may enter empty user_id
            return jsonify([])
        root_path = create_personal_folder(user_id)
        highlighted_files = request_json.get("highlighted_files", [])
        if root_path is None:
            return {"error": "root_path parameter is required", "error_code": 404}
        if os.path.exists(root_path) and os.path.isdir(root_path):
            current_path_tree_list: list = []
            id_to_path_dict = {0: root_path}
            _path_tree_for_react_dnd_treeview(current_path_tree_list, id_to_path_dict,
                                              root_path, 0,
                                              highlighted_files=highlighted_files)
            return jsonify(current_path_tree_list)
        else:
            return Response(response=None, status=f"{UNFOUND} Directory not found")
    except Exception as e:
        print(e)
        return Response(response=None, status=f"{INTERNAL} Directory not found")


@app.route("/api/set_default_examples", methods=["POST"])
def set_default_examples() -> Response:
    """Sets default files for each user."""
    try:
        # Should be called after auth is verified
        request_json = request.get_json()
        user_id = request_json.pop("user_id", DEFAULT_USER_ID)
        root_path = create_personal_folder(user_id)
        example_dir = os.path.dirname(os.path.dirname(app.config["UPLOAD_FOLDER"]))
        example_path = os.path.join(example_dir, "data/examples/")
        if os.path.exists(example_path):
            shutil.copytree(example_path, root_path, dirs_exist_ok=True)
            return jsonify(
                {"success": True, "message": "Default examples are set successfully"})
        else:
            return Response(response=None,
                            status=f"{UNFOUND} Directory not found at {example_dir}")
    except Exception as e:
        return Response(response=None,
                        status=f"{INTERNAL} Fail to Set Default Examples")
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
import redis
from typing import Any

from backend.utils.utils import logger
import os

r = redis.Redis(host=os.getenv("REDIS_SERVER"), port=6379, decode_responses=True)


# Set the queue and pending key
QUEUE_RUNNING = "kernel_running_queue"
QUEUE_PENDING = "kernel_pending_queue"
SUBMIT_EVENT = "job_submitted"
RUNNING_EVENT = "job_started"
COMPLETE_EVENT = "job_completed"

MAX_CONCURRENT_KERNELS = 300


def add_job_to_pending(job: Any) -> None:
    # always add the jobn to pending
    r.rpush(QUEUE_PENDING, job)


def move_pending_to_running() -> None:
    """Move pending jobs to the running queue."""
    # Get the first pending job
    job = r.lindex(QUEUE_PENDING, 0)
    if job is not None:
        logger.bind(msg_head="Job running").debug(job)
        # Move the job from pending queue to running queue
        r.rpush(QUEUE_RUNNING, job)
        # Notify the running channel
        r.publish(RUNNING_EVENT, job)
        # Remove the job from pending queue
        r.lpop(QUEUE_PENDING)


# Subscribe to job completion events
def handle_job_completion(message: dict) -> None:
    # the data should be the chat id
    chat_id = message["data"]
    logger.bind(msg_head="Job completed").debug(chat_id)
    # here we only care about the capacity, not caring about which one is poped now
    logger.bind(msg_head="Queue running").debug(r.lrange(QUEUE_RUNNING, 0, -1))
    r.lrem(QUEUE_RUNNING, 0, chat_id)
    move_pending_to_running()


def handle_new_job(message: dict) -> None:
    # the data should be the chat id
    chat_id = message["data"]
    logger.bind(msg_head="Job submitted").debug(chat_id)
    # all submitted jobs into pending queue
    add_job_to_pending(chat_id)
    # push the id to the pending queue
    logger.bind(msg_head="Queue pending").debug(r.lrange(QUEUE_PENDING, 0, -1))
    if r.llen(QUEUE_RUNNING) < MAX_CONCURRENT_KERNELS:
        move_pending_to_running()


def start_kernel_publisher() -> None:
    # Connect to Redis
    r.delete(QUEUE_RUNNING)
    r.delete(QUEUE_PENDING)
    # Start the publisher & subscriber
    p = r.pubsub()
    p.subscribe(**{COMPLETE_EVENT: handle_job_completion, SUBMIT_EVENT: handle_new_job})

    p.run_in_thread(sleep_time=0.1)
APP_TYPES = ["copilot", "plugins", "webot"]
TIME_STEP = 0.035
TIME_OUT_MAP = {"copilot": 90, "plugins": 300, "webot": 600}
STREAM_BLOCK_TYPES = ["image", "echarts"]
STREAM_TOKEN_TYPES = ["tool", "transition", "execution_result", "error", "kaggle_search", "kaggle_connect", "plain"]
EXECUTION_RESULT_MAX_TOKENS_MAP = {"copilot": 1000, "plugins": 2000, "webot": 20000}

HEARTBEAT_INTERVAL = 10

# define error code
UNAUTH = 401
UNFOUND = 404
OVERLOAD = 503
INTERNAL = 500
UNSUPPORTED = 403

# define models which need extra continue flag
NEED_CONTINUE_MODEL = {"claude-v1", "claude-2"}
DEFAULT_USER_ID = "DefaultUser"
import os
import warnings
import threading

from backend.app import app
from backend.kernel_publisher import start_kernel_publisher
from backend.utils.threading import ThreadManager
from backend.utils.utils import VariableRegister, init_log
from backend.memory import (
    ChatMemoryManager,
    MessageMemoryManager,
    UserMemoryManager,
)

warnings.filterwarnings("ignore", category=UserWarning)

logger = init_log(
    error=os.path.join(".logging", "error.log"),
    runtime=os.path.join(".logging", "runtime.log"),
    serialize=os.path.join(".logging", "serialize.log"),
    trace=os.path.join(".logging", "trace.log"),
)

VARIABLE_REGISTER_BACKEND = os.environ.get("VARIABLE_REGISTER_BACKEND", "local")
MESSAGE_MEMORY_MANAGER_BACKEND = os.environ.get("MESSAGE_MEMORY_MANAGER_BACKEND", "local")
API_KEY_MEMORY_MANAGER_BACKEND = os.environ.get("API_KEY_MEMORY_MANAGER_BACKEND", "local")
JUPYTER_KERNEL_MEMORY_MANAGER_BACKEND = os.environ.get("JUPYTER_KERNEL_MEMORY_MANAGER_BACKEND", "local")

message_pool: MessageMemoryManager = MessageMemoryManager(name="message_pool", backend=MESSAGE_MEMORY_MANAGER_BACKEND)
grounding_source_pool: ChatMemoryManager = ChatMemoryManager()
api_key_pool: UserMemoryManager = UserMemoryManager(name="api_key_pool", backend=API_KEY_MEMORY_MANAGER_BACKEND)
jupyter_kernel_pool: ChatMemoryManager = ChatMemoryManager(
    name="jupyter_kernel_pool", backend=JUPYTER_KERNEL_MEMORY_MANAGER_BACKEND
)
threading_pool: ThreadManager = ThreadManager()

message_id_register = VariableRegister(name="message_id_register", backend=VARIABLE_REGISTER_BACKEND)

# Monitor kernel and kill long running kernels
if app.config["CODE_EXECUTION_MODE"] == "docker":
    threading.Thread(target=start_kernel_publisher, args=(), daemon=True).start()

if __name__ == "__main__":
    import multiprocess

    multiprocess.set_start_method("spawn", True)
    app.run(debug=True)
from typing import Any, Dict, List, Union
from loguru import logger
import json

from backend.app import app
from backend.utils.running_time_storage import get_running_time_storage
from backend.utils.user_conversation_storage import get_user_conversation_storage
from real_agents.adapters.memory import BaseChatMemory

HUMAN_MESSAGE_KEY = "human_message"
AI_MESSAGE_KEY = "ai_message"

LOCAL = "local"
DATABASE = "database"


class UserMemoryManager:
    """A class to manage the global memory including messages, grounding_sources,
    etc. on user level"""

    # api_key_pool:
    # {
    #   "user_id": [{
    #       "tool_id": the id of the tool,
    #       "tool_name": the name of the tool,
    #       "api_key": the api_key of the tool,
    # }]
    # }

    def __init__(self, name: str = None, backend: str = LOCAL, memory_pool: Dict = None):
        self.backend = backend
        self.name = name
        if self.backend == LOCAL:
            if memory_pool is None:
                memory_pool = {}
            self.memory_pool = memory_pool
        elif self.backend == DATABASE:
            with app.app_context():
                self.redis_client = get_running_time_storage()
                self.db_client = get_user_conversation_storage()
        else:
            raise ValueError("Unknown backend option: {}".format(self.backend))

    def get_pool_info_with_id(
        self,
        user_id: str,
        default_value: Union[List, Dict],
    ) -> Any:
        """Gets the information with user_id and chat_id from the pool."""
        if self.backend == LOCAL:
            pool = self.memory_pool
            if user_id in pool:
                return pool[user_id]
            else:
                return default_value
        elif self.backend == DATABASE:
            memory_pool_name = f"{self.name}:{user_id}"
            if self.redis_client.exists(memory_pool_name):
                # In cache
                info = json.loads(self.redis_client.get(memory_pool_name))
            else:
                # Cache miss
                try:
                    # api_keys are not stored in database
                    if self.name == "api_key_pool":
                        info = default_value
                    else:
                        raise NotImplementedError(f"Currently only support message pool in database, not {self.name}")
                except Exception as e:
                    # Not in database
                    logger.bind(user_id=user_id, msg_head="Cache miss but not in database").warning(
                        "Failed to get pool info from database: {}".format(e)
                    )
                    info = default_value
            return info

    def set_pool_info_with_id(self, user_id: str, info: Any) -> None:
        """Sets the information with user_id and chat_id to the pool."""
        if self.backend == LOCAL:
            pool = self.memory_pool
            if user_id not in pool:
                pool[user_id] = info
        elif self.backend == DATABASE:
            # As db has its own updating logic, we only need to update the cache here (write-through).
            memory_pool_name = f"{self.name}:{user_id}"
            self.redis_client.set(memory_pool_name, json.dumps(info))

    def __iter__(self):
        """Iterates over the memory pool."""
        if self.backend == LOCAL:
            for user_id, info in self.memory_pool.items():
                yield user_id, info
        elif self.backend == DATABASE:
            raise NotImplementedError("Currently not support UserMemoryManager iteration in database mode.")


class ChatMemoryManager:
    """A class to manage the global memory including messages, grounding_sources, etc. on chat level"""

    # memory_pool:
    # {user_id: {chat_id: [
    #                           {"message_id": the id of this pair of messages,
    #                            "parent_message_id": the id of the parent message,
    #                            "message_type": type of the message, possible values: human_message / ai_message
    #                            "message_content": content of the message
    #                           }
    #                     ]
    #           }
    # }
    # grounding_source_pool:
    # {user_id: {chat_id: {"filenames": List of filenames,
    #                      "activated_filenames": List of user-selected activated names}}

    def __init__(self, name: str = None, backend: str = LOCAL, memory_pool: Dict = None):
        """
        This ChatMemoryManager can not be applied to grounding_source_pool in database mode.
        """
        self.backend = backend
        self.name = name
        if self.backend == LOCAL:
            if memory_pool is None:
                memory_pool = {}
            self.memory_pool = memory_pool
        elif self.backend == DATABASE:
            with app.app_context():
                self.redis_client = get_running_time_storage()
                self.db_client = get_user_conversation_storage()
        else:
            raise ValueError("Unknown backend option: {}".format(self.backend))

    def get_pool_info_with_id(
        self,
        user_id: str,
        chat_id: str,
        default_value: Union[List, Dict],
    ) -> Any:
        """Gets the information with user_id and chat_id from the pool."""
        if self.backend == LOCAL:
            pool = self.memory_pool
            if user_id in pool and chat_id in pool[user_id]:
                return pool[user_id][chat_id]
            else:
                return default_value
        elif self.backend == DATABASE:
            memory_pool_name = f"{self.name}:{user_id}:{chat_id}"
            if self.redis_client.exists(memory_pool_name):
                # In cache
                info = json.loads(self.redis_client.get(memory_pool_name))
            else:
                # Cache miss
                try:
                    # Found in database
                    if self.name == "message_pool":
                        info = []
                        response = self.db_client.message.find({"conversation_id": chat_id})
                        if response is None:
                            # Not in database (new chat)
                            info = default_value
                        else:
                            # In database
                            for message in response:
                                if message["role"] == "user":
                                    message_type = HUMAN_MESSAGE_KEY
                                elif message["role"] == "assistant":
                                    message_type = AI_MESSAGE_KEY
                                else:
                                    raise ValueError("Unknown role: {}".format(message["role"]))
                                info.append(
                                    {
                                        "message_id": message["message_id"],
                                        "parent_message_id": message["parent_message_id"],
                                        "message_type": message_type,
                                        "message_content": message["data_for_llm"],
                                    }
                                )
                            self.redis_client.set(memory_pool_name, json.dumps(info))
                    elif self.name == "jupyter_kernel_pool":
                        info = default_value
                    else:
                        raise NotImplementedError(f"Currently only support message pool in database, not {self.name}")
                except Exception as e:
                    # Not in database
                    logger.bind(user_id=user_id, chat_id=chat_id, msg_head="Cache miss but not in database").warning(
                        "Failed to get pool info from database: {}".format(e)
                    )
                    info = default_value
            return info

    def set_pool_info_with_id(self, user_id: str, chat_id: str, info: Any) -> None:
        """Sets the information with user_id and chat_id to the pool."""
        if self.backend == LOCAL:
            pool = self.memory_pool
            if user_id not in pool:
                pool[user_id] = {}
            pool[user_id][chat_id] = info
        elif self.backend == DATABASE:
            # As db has its own updating logic, we only need to update the cache here (write-through).
            memory_pool_name = f"{self.name}:{user_id}:{chat_id}"
            self.redis_client.set(memory_pool_name, json.dumps(info))

    def __iter__(self):
        """Iterates over the memory pool."""
        if self.backend == LOCAL:
            for user_id, chat_id_info in self.memory_pool.items():
                for chat_id, info in chat_id_info.items():
                    yield user_id, chat_id, info
        elif self.backend == DATABASE:
            if self.name == "jupyter_kernel_pool":
                iterator = self.redis_client.scan_iter("jupyter_kernel_pool:*")
                for key in iterator:
                    user_id, chat_id = key.split(":")[1:]
                    yield user_id, chat_id, self.get_pool_info_with_id(user_id, chat_id, {})
            else:
                raise NotImplementedError("Currently only support jupyter kernel pool iteration in database mode.")

    def drop_item_with_id(self, user_id: str, chat_id: str):
        if self.backend == LOCAL:
            # drop item under one user
            if user_id in self.memory_pool:
                self.memory_pool[user_id].pop([chat_id], None)
        elif self.backend == DATABASE:
            if self.name == "jupyter_kernel_pool":
                self.redis_client.delete(f"{self.name}:{user_id}:{chat_id}")
            else:
                raise NotImplementedError("Currently only support jupyter kernel pool delete in database mode.")


class MessageMemoryManager(ChatMemoryManager):
    """A class to manage the memory of messages."""

    @staticmethod
    def load_agent_memory_from_list(agent_memory: BaseChatMemory, message_list: List[Dict[str, str]]) -> None:
        """Load agent's memory from a list."""
        agent_memory.clear()
        for message in message_list:
            if message.get("message_type", None) == HUMAN_MESSAGE_KEY:
                agent_memory.chat_memory.add_user_message(message["message_content"])
            elif message.get("message_type", None) == AI_MESSAGE_KEY:
                agent_memory.chat_memory.add_ai_message(message["message_content"])
        try:
            agent_memory.fit_max_token_limit()
        except Exception as e:
            import traceback

            traceback.print_exc()
            pass

    @staticmethod
    def save_agent_memory_to_list(agent_memory: BaseChatMemory) -> List[Dict[str, str]]:
        """Saves agent's memory to a list"""
        messages = agent_memory.chat_memory.messages
        message_list = []
        for message in messages:
            if message.type == "human":
                message_list.append(
                    {
                        "message_type": "human_message",
                        "message_content": message.content,
                    }
                )
            elif message.type == "ai":
                message_list.append(
                    {
                        "message_type": "ai_message",
                        "message_content": message.content,
                    }
                )
        return message_list

    def get_activated_message_list(
        self,
        user_id: str,
        chat_id: str,
        default_value: Union[List, Dict],
        parent_message_id: Union[int, str],
    ) -> List:
        """Gets the activated message list from leaf to root."""
        # ONLY work for messages
        message_list = self.get_pool_info_with_id(user_id, chat_id, default_value)
        activated_message_list = []
        end_point = parent_message_id
        while len(message_list) > 0 and end_point != -1:
            flag = False
            for msg in message_list:
                if msg["message_id"] == end_point:
                    if end_point == msg["parent_message_id"]:
                        flag = False
                        break
                    activated_message_list = [msg] + activated_message_list
                    end_point = msg["parent_message_id"]
                    flag = True
                    break
            if not flag:
                break
        logger.bind(msg_head=f"get_activated_message_list").debug(activated_message_list)
        return activated_message_list
# flake8: noqa
# mypy: ignore-errors

from backend.api import (
    chat_copilot,
    chat_plugin,
    chat_webot,
    conversation,
    file,
    recommend,
    tool,
    language_model,
    webot_actions,
    webot_instructions,
    data_tools,
    data_connector,
)
