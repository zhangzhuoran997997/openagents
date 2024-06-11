TOPIC_EXTRACT_REF_CODE = """
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


TOPIC_EXTRACT_USER_PROMPT = """
history_code = \"\"\"{history_code}\"\"\"
data = \"\"\"{data}\"\"\"
reference_code = \"\"\"{reference_code}\"\"\"
human_question = \"\"\"{question}\"\"\"

history_dict = {{
    "history code": history_code,
    "human question": human_question,
    "data": data,
    "reference_code": reference_code,
}}
"""

# TOPIC_EXTRACT_SYSTEM_PROMPT = f"You are now the following python function: ```{FUNCTION_ROLE_PLAY}```\n\nRespond exclusively with the generated code wrapped <code></code>. Ensure that the code you generate is executable Python code that can be run directly in a Python environment, requiring no additional string encapsulation or escape characters."
TOPIC_EXTRACT_SYSTEM_PROMPT=""