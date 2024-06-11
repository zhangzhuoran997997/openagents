#!/bin/bash

SOURCE_DIRECTORIES=(
    "/data/zhuoran/code/OpenAgents/backend"
    "/data/zhuoran/code/OpenAgents/real_agents/adapters"
    "/data/zhuoran/code/OpenAgents/real_agents/data_agent"
)
DESTINATION_DIRECTORY="/data/zhuoran/code/OpenAgents/combined"
COMBINED_FILES=(
    "backend.py"
    "adapters.py"
    "data_agent.py"
)

# 循环遍历源目录和合并后的文件路径
for ((i=0; i<${#SOURCE_DIRECTORIES[@]}; i++)); do
    SOURCE_DIRECTORY="${SOURCE_DIRECTORIES[i]}"
    COMBINED_FILE="${COMBINED_FILES[i]}"

    # 使用find命令来查找文件
    find "$SOURCE_DIRECTORY" -name "*.py" -exec sh -c '
        # 复制文件到目标目录
        cp "$1" "$2/$(basename "$1")"
        # 将文件内容添加到合并后的文件
        cat "$1" >> "$3"
    ' sh {} "$DESTINATION_DIRECTORY" "$COMBINED_FILE" \;

    echo "All Python files in $SOURCE_DIRECTORY have been copied to $DESTINATION_DIRECTORY and combined into $COMBINED_FILE."
done