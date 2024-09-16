# 错误
def transform_list(original_list):
    # 定义一个空列表来存储转换后的数据
    transformed_list = []
    # 定义一个临时列表来存储图片路径和描述
    temp_list = []

    # 遍历原始列表
    for item in original_list:
        if isinstance(item[0], tuple):  # 如果当前项是一个元组
            # 将元组添加到临时列表的图片路径部分
            temp_list.append(item[0])
        else:  # 如果当前项不是元组，即是一个描述
            # 检查临时列表是否不为空
            if temp_list:
                # 将临时列表的图片路径和当前描述组合成一个元组，添加到转换列表
                transformed_list.append((temp_list, item[1]))
                # 清空临时列表以备下一个描述使用
                temp_list = []
            # 将当前描述添加到转换列表
            transformed_list.append(item)

    # 如果临时列表在最后不为空，说明最后一组图片没有对应的描述
    if temp_list:
        # 将最后一组图片的路径添加到转换列表，描述为空字符串
        transformed_list.append((temp_list, ""))

    return transformed_list


# 原始列表
original = [
    ["你是谁", "我是你的小助手"],
    [("./images/0001.jpg",), None],
    ["", "这张图片中有一只猫"],
    [("./images/0002.jpg",), None],
    ["这张图片展示的什么内容?", "这张图片中也有一只猫"],
    [("./images/0003.jpg",), None],
    [("./images/0004.jpg",), None],
    [
        "这2张图片展示的什么内容?",
        "第一张图片中有一个人在滑雪，第二张图片中有一个人坐在长椅上休息。",
    ],
    [("./images/0005.jpg",), None],
    [("./images/0006.jpg",), None],
    ["", "这两张图片显示了雪山上的景色"],
]

# 转换列表
transformed = transform_list(original)
print(transformed)
