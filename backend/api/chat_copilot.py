import traceback
import os
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
from backend.utils.utils import (
    get_data_model_cls,
    load_grounding_source,
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


# self code
from backend.topic.topic_new import topic_analysis
# import multiprocessing

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
    logger.bind(user_id=user_id, chat_id=chat_id, api="/chat",
                    msg_head="create grounding source").debug(grounding_source_dict)

    
    

    # Initialize tools(executors)
    # 这几个执行器的run函数都来自chain的执行函数，其实就是多个chain
    basic_chat_executor = ChatExecutor()
    python_code_generation_executor = CodeGenerationExecutor(
        programming_language="python", memory=read_only_memory)
    sql_code_generation_executor = CodeGenerationExecutor(programming_language="sql",
                                                          memory=read_only_memory)
    echart_code_generation_executor = CodeGenerationExecutor(
        programming_language="python", memory=read_only_memory, usage="echarts"
    )
    line_code_generation_executor = CodeGenerationExecutor(
        programming_language="python", memory=read_only_memory, usage="line")

    pie_code_generation_executor = CodeGenerationExecutor(
        programming_language="python", memory=read_only_memory, usage="pie")
    
    topic_extract_executor = CodeGenerationExecutor(
        programming_language="python", memory=read_only_memory, usage="topic_extract")

    kaggle_data_loading_executor = KaggleDataLoadingExecutor()

    def run_topic_extract_builder(term: str) -> Union[Dict, DataModel]:

        try:
            # raw_data_path = grounding_source_dict[]
            raw_data_path = '/data/llmagents/data/llm_agent/tai_news_0526/tw_news2023_r3k.json'
            stats, topic_dict = topic_analysis(raw_data_path)
            logger.bind(api="/chat",
                        msg_head="Analysis news from ").debug(stats['path'])

            data_summary = f"We retrieve related data that contains {stats['count']} news from different {stats['url_count']} \
websites during {stats['mintime']} to {stats['maxtime']}.\n\n"

            topics = topic_dict['Topic']
            counts = topic_dict['Count']
            summary = topic_dict['Summary']
            topic_prompt = ""
            for t, c, s in zip(topics, counts, summary):
                topic_prompt += f"{t}\t{c}\t{s}\n"
            new_user_intent = f"Now we are analyizing topic of related data. {data_summary} \
We have obtained the main topic of these news. Here are all data about topic analysis.\n\n\
Topic\tCount\tSummary\n{topic_prompt}\
Please provide a succinct yet meaningful summary for the topic, count and summary.\n\n"
# Finally, generate Pie code using TopicAnalyzer Tool for visulaizing topic count distribution.\n\n"

            # # 自动加载news数据
            file_path = topic_dict['path']
            filename = file_path.split('/')[-1]
            filename_no_ext = os.path.splitext(filename)[0]
            
            data = load_grounding_source(file_path)
            data_model = get_data_model_cls(filename).from_raw_data(
                raw_data=data,
                raw_data_name=filename_no_ext,
                raw_data_path=file_path,
            )
            grounding_source_dict[file_path] = data_model

            input_grounding_source = [gs for gs in grounding_source_dict.values()]
            
            # Get the result
            results = basic_chat_executor.run(
                # user_intent=term,
                user_intent=new_user_intent,
                llm=llm,
                # grounding_source=input_grounding_source,
                # user_id=user_id,
                # chat_id=chat_id,
                # code_execution_mode=code_execution_mode,
                # jupyter_kernel_pool=jupyter_kernel_pool,
            )

            logger.bind(msg_head=f"TopicExtractor results({llm})").debug(results)
            return results["result"]
            # if results["result"]["success"]:
            #     if results["result"]["result"] is not None:
            #         raw_output = results["result"]["result"]
            #     elif results["result"]["stdout"] != "":
            #         raw_output = results["result"]["stdout"]
            #     else:
            #         raw_output = ""
            #     observation = JsonDataModel.from_raw_data(
            #         {
            #             "success": True,
            #             "result": raw_output,
            #             "images": results["result"]["outputs"] if ".show()" in results[
            #                 "intermediate_steps"] else [],
            #             "intermediate_steps": results["intermediate_steps"],
            #         },
            #         filter_keys=["images"],
            #     )
            # else:
            #     observation = JsonDataModel.from_raw_data(
            #         {
            #             "success": False,
            #             "result": results["result"]["error_message"],
            #             "intermediate_steps": results["intermediate_steps"],
            #         }
            #     )
            # return observation
        except Exception as e:
            logger.bind(msg_head=f"TopicExtractor error({llm})").error(str(e))

            traceback.print_exc()
            results = basic_chat_executor.run(user_intent=term, llm=llm)
            return results["result"]

    def run_line_builder(term: str) -> Union[Dict, DataModel]:

        try:
            # term += "using topic_analysis.csv"
            # # 自动加载topic datamodel数据
            # file_path = '/data/llmagents/code/OpenAgents/backend/data/DefaultUser/topic_hot_time.csv'
            # filename = 'topic_hot_time.csv'
            file_path = '/data/llmagents/code/OpenAgents/backend/data/DefaultUser/sentiment.csv'
            filename = 'sentiment.csv'
            filename_no_ext = os.path.splitext(filename)[0]
            data = load_grounding_source(file_path)
            data_model = get_data_model_cls(filename).from_raw_data(
                raw_data=data,
                raw_data_name=filename_no_ext,
                raw_data_path=file_path,
            )
            grounding_source_dict[file_path] = data_model
            logger.bind(user_id=user_id, chat_id=chat_id, api="/chat",
                    msg_head="line grounding source").debug(grounding_source_dict)

            input_grounding_source = [gs for gs in grounding_source_dict.values()]
            
            # Get the result
            results = line_code_generation_executor.run(
                user_intent=term,
                llm=llm,
                grounding_source=input_grounding_source,
                user_id=user_id,
                chat_id=chat_id,
                code_execution_mode=code_execution_mode,
                jupyter_kernel_pool=jupyter_kernel_pool,
            )

            logger.bind(msg_head=f"LineCodeBuilder results({llm})").debug(results)

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
                        "result": "",
                        "echarts": polish_echarts(raw_output),
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
            logger.bind(msg_head=f"LinePloter error({llm})").error(str(e))

            traceback.print_exc()
            results = basic_chat_executor.run(user_intent=term, llm=llm)
            return results["result"]

    def run_pie_builder(term: str) -> Union[Dict, DataModel]:

        try:
            # term += "using topic_analysis.csv"
            # # 自动加载topic datamodel数据
            # file_path = '/data/llmagents/code/OpenAgents/backend/data/DefaultUser/topic_analysis.csv'
            # filename = 'topic_analysis.csv' 
            # filename_no_ext = os.path.splitext(filename)[0]
            
            # data = load_grounding_source(file_path)
            # data_model = get_data_model_cls(filename).from_raw_data(
            #     raw_data=data,
            #     raw_data_name=filename_no_ext,
            #     raw_data_path=file_path,
            # )
            # grounding_source_dict[file_path] = data_model
            logger.bind(user_id=user_id, chat_id=chat_id, api="/chat",
                    msg_head="pie grounding source").debug(grounding_source_dict)

            input_grounding_source = [gs for gs in grounding_source_dict.values()]
            
            # Get the result
            results = pie_code_generation_executor.run(
                user_intent=term,
                llm=llm,
                grounding_source=input_grounding_source,
                user_id=user_id,
                chat_id=chat_id,
                code_execution_mode=code_execution_mode,
                jupyter_kernel_pool=jupyter_kernel_pool,
            )

            logger.bind(msg_head=f"PieCodeBuilder results({llm})").debug(results)

            if results["result"]["success"]:
                if results["result"]["result"] is not None:
                    raw_output = results["result"]["result"]
                elif results["result"]["stdout"] != "":
                    raw_output = results["result"]["stdout"]
                else:
                    raw_output = ""
                # observation = JsonDataModel.from_raw_data(
                #     {
                #         "success": True,
                #         "result": raw_output,
                #         "images": results["result"]["outputs"] if ".show()" in results[
                #             "intermediate_steps"] else [],
                #         "intermediate_steps": results["intermediate_steps"],
                #     },
                #     filter_keys=["images"],
                # )
                observation = JsonDataModel.from_raw_data(
                    {
                        "success": True,
                        "result": "",
                        "echarts": polish_echarts(raw_output),
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
            logger.bind(msg_head=f"PiePloter error({llm})").error(str(e))

            traceback.print_exc()
            results = basic_chat_executor.run(user_intent=term, llm=llm)
            return results["result"]

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
    #TODO 查看一下langchain定义tool的过程
    tool_dict = {

        # 添加自定义的tool，例如pie图和折线图
#         "TopicAnalyzer": Tool(
#             name="TopicAnalyzer",
#             func=run_topic_analysis_builder,
#             description="""
# Description: This tool aims to analyze hot topics in news data and extract these topics to visualize them into pie charts by using pyecharts. Firstly, it takes your original news data and creates cluster algorithms in python to extract hot topics and their corresponding number of news articles. Secondly, make a summary about the news data. Finally, use the pie charts to visualize hot topics and their count.
# Input: A natural question and the path of news data.
# Output: A Python dealing data program + its execution result to visualize hot topics + a summary ahout news data + an Echarts script, specifically tailored for generated topics, that generates an interactive chart upon execution.
# Note: The tool MUST be used whenever you want to analyze topic from news.
# """,
#         ),
 "LinePloter": Tool(
                    name="LinePloter",
                    func=run_line_builder,
                    description="""
Description: This tool aims to visualize the chnages of hot topics, different sentiments or stances following real time into line charts by using pyecharts. Firstly, it takes topics (sentiments/stances), number and their corresponding time. Secondly, make a summary about the data. Finally, use the line charts to visualize chnages of hot topics, different sentiments or stances following real time.
Input: A natural question and the path of topic (sentiment/stance), count and the corresponding time.
Output: A summary ahout data distribution + an Echarts script, specifically tailored for main topics (sentiments/stances), that generates an interactive chart upon execution.
Note: The tool MUST be used whenever you want to visualize the changes of hot topics, sentiments or stances following real time.
        """,
                ),
        "PiePloter": Tool(
                    name="PiePloter",
                    func=run_pie_builder,
                    description="""
Description: This tool aims to visualize hot topics, different sentiments or stances into pie charts by using pyecharts. Firstly, it takes topics (sentiments/stances) and their corresponding number. Secondly, make a summary about the data distribution. Finally, use the pie charts to visualize hot topics (sentiments/stances) and their count.
Input: A natural question and the path of topic (sentiment/stance) and the corresponding count.
Output: A summary ahout data distribution + an Echarts script, specifically tailored for main topics (sentiments/stances), that generates an interactive chart upon execution.
Note: The tool MUST be used whenever you want to visualize hot topics, sentiments or stances distribution.
        """,
                ),
        "TopicExtractor": Tool(
            name="TopicExtractor",
            func=run_topic_extract_builder,
            description="""
Description: This tool aims to extract hot topics from news data. Firstly, it extracts hot topics and their corresponding number from news articles. Secondly, make a summary about the data.
Input: A natural question.
Output: A summary ahout topics of news data.
Note: The tool MUST be used whenever you want to extract topics from news.
""",
        ),

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

    logger.bind(user_id=user_id, chat_id=chat_id, api="/chat",
                    msg_head="Tools").debug(tools)

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

        logger.bind(user_id=user_id, chat_id=chat_id, api="/chat",
                    msg_head="Test").debug('!!!'*100)

        grounding_source_dict = grounding_source_pool.get_pool_info_with_id(user_id,
                                                                                chat_id,
                                                                                default_value={})

        logger.bind(user_id=user_id, chat_id=chat_id, api="/chat",
                    msg_head="grounding source").debug(grounding_source_dict)

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
            
            #{'droppable': False, 'highlight': False, 'id': 1, 'parent': 0, 'text': 'test.csv'}
            file_node = api_call["args"]["activated_file"]
            #获取用户本地存储文件夹
            folder = create_personal_folder(user_id)
            #获取本地存储apply文件的文件路径
            file_path = _get_file_path_from_node(folder, file_node)
            #获取数据摘要类，根据不同数据类型，本质上是一个执行数据摘要的chain
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

