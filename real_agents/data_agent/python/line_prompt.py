LINE_REF_CODE = """Here are some examples of generating Py-Echarts Code based on the given table(s). Please generate new one based on the data and question human asks you, import the neccessary libraries and make sure the code is correct.

IMPORTANT: You need to follow the coding style, and the type of the x, y axis. But also need to focus on the column name of the uploaded tables(if exists). Generally, PyEcharts does not accept numpy.int or numpy.float, etc. It only supports built-in data type like int, float, and str.


Given the following database:
topic_hot_time.csv
   Time     Topic     Count
0        2021-01     climate change     33
1        2021-01     air pollution     55
2        2021-01     covid-19     26
3        2021-02     climate change     39
4        2021-02     air pollution     63
5        2021-02     covid-19     46
6        2021-03     climate change     63
7        2021-03     air pollution     75
8        2021-03     covid-19     52

Q: A line chart about topic over time would be useful. Could you help plot it?
<code>
import pandas as pd
from pyecharts.charts import Line
from pyecharts import options as opts
from collections import defaultdict

df = pd.read_csv("topic_hot_time.csv", sep="\\t")

times = []

dict = defaultdict(list)
for i in range(df.shape[0]):
    if df['Time'][i] not in times:
        times.append(df['Time'][i])
    dict[df['Topic'][i]].append(df['Count'][i])

line = Line(init_opts=opts.InitOpts(width="1500px"))

line.add_xaxis(times)
for key in dict.keys():
    line.add_yaxis(key, [int(i) for i in dict[key]])
    print(dict[key])

line.set_global_opts(title_opts=opts.TitleOpts(title='Topic River'),xaxis_opts=opts.AxisOpts(
            axislabel_opts={"interval":"0"}
        ))
line.set_series_opts(
     areastyle_opts=opts.AreaStyleOpts(opacity=0.5), # 透明度
     label_opts=opts.LabelOpts(is_show=False), # 是否显示标签
 )
ret_json = line.dump_options()
print(ret_json)
</code>

Given the following database:
sentiment.csv
   Time     Sentiment     Count
0        2021-01     positive     33
1        2021-01     negative     55
2        2021-01     neutral     26
3        2021-02     positive     39
4        2021-02     negative     63
5        2021-02     neutral     46
6        2021-03     positive     63
7        2021-03     negative     75
8        2021-03     neutral     52

Q: I would like to visularize the changes of sentiments over time.
<code>
import pandas as pd
from pyecharts.charts import Line
from pyecharts import options as opts
from collections import defaultdict

df = pd.read_csv("sentiment.csv", sep="\\t")

times = []

dict = defaultdict(list)
for i in range(df.shape[0]):
    if df['Time'][i] not in times:
        times.append(df['Time'][i])
    dict[df['Sentiment'][i]].append(df['Count'][i])

line = Line(init_opts=opts.InitOpts(width="1500px"))

line.add_xaxis(times)
for key in dict.keys():
    line.add_yaxis(key, [int(i) for i in dict[key]])
    print(dict[key])

line.set_global_opts(title_opts=opts.TitleOpts(title='Sentiment River'),xaxis_opts=opts.AxisOpts(
            axislabel_opts={"interval":"0"}
        ))
line.set_series_opts(
     areastyle_opts=opts.AreaStyleOpts(opacity=0.5), # 透明度
     label_opts=opts.LabelOpts(is_show=False), # 是否显示标签
 )
ret_json = line.dump_options()
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


LINE_USER_PROMPT = """
history_code = \"\"\"{history_code}\"\"\"
data = \"\"\"{data}\"\"\"
reference_code = \"\"\"{reference_code}\"\"\"
human_question = \"\"\"{question}
# MUST follow reference_code, and only use Line function from pyecharts to show the change of topic, sentiment or stance following time.\"\"\"

history_dict = {{
    "history code": history_code,
    "human question": human_question,
    "data": data,
    "reference_code": reference_code,
}}
"""

LINE_SYSTEM_PROMPT = f"You are now the following python function: ```{FUNCTION_ROLE_PLAY}```\n\nRespond exclusively with the generated code wrapped <code></code>. Ensure that the code you generate is executable Python code that can be run directly in a Python environment, requiring no additional string encapsulation or escape characters."
# And make a detailed summary about the topic distribution.
