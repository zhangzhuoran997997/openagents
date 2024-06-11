#!/bin/bash

# 目标目录，你需要将这个路径替换为你实际的路径
DIRECTORY="/data/zhuoran/code/OpenAgents/combined"

# 获取目录中的文件数量
file_count=$(ls -1 $DIRECTORY | wc -l)

# 当文件数量大于50时，进行循环
while ((file_count > 50)); do
    # 获取目录中的前两个文件
    file1=$(ls -1 $DIRECTORY | head -n 1)
    file2=$(ls -1 $DIRECTORY | head -n 2 | tail -n 1)

    # 合并这两个文件
    cat "$DIRECTORY/$file1" "$DIRECTORY/$file2" > "$DIRECTORY/combined"
    
    # 删除原始的两个文件
    rm "$DIRECTORY/$file1" "$DIRECTORY/$file2"

    # 重新获取文件数量
    file_count=$(ls -1 $DIRECTORY | wc -l)
done

echo "Files in $DIRECTORY have been combined until the number of files is less than 50."