tokens = [
    # 4 个特殊 token
    "<pad>",   # 填充：用于补齐序列长度
    "<unk>",   # 未知：表示词典中没有的词
    "<bos>",   # 序列开始
    "<eos>",   # 序列结束

    # 4 个标点
    ".", ",", "?", "!",

    # 人称代词与物主限定词
    "i", "you", "he", "she", "it", "we", "they",
    "my", "your",

    # 冠词与指示词
    "a", "an", "the", "this", "that",

    # 常用动词形式、情态动词与否定词
    "am", "is", "are", "was", "were",
    "do", "does", "have", "has",
    "can", "will", "not",

    # 连词与介词
    "and", "or", "but",
    "in", "on", "at", "to", "from", "with", "for",

    # 动作
    "like", "love", "want", "need", "see",
    "go", "come", "eat", "drink", "read",
    "write", "play", "run", "sleep",

    # 名词
    "cat", "dog", "bird", "fish", "boy", "girl", "friend",
    "apple", "water", "milk", "food", "book", "ball",
    "home", "school", "park", "sun",

    # 形容词
    "good", "bad", "big", "small", "happy", "sad",
    "hot", "cold", "red", "blue", "new", "old",

    # 程度、位置、时间与回答
    "very", "here", "there", "now", "today", "tomorrow",
    "yes", "no",

    # 疑问词
    "what", "who", "where", "when", "how",
]

# token → ID，编号为 0～99
token_to_id = {token: idx for idx, token in enumerate(tokens)}

# ID → token
id_to_token = {idx: token for idx, token in enumerate(tokens)}

