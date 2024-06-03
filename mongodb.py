from pymongo import MongoClient

# 连接到MongoDB
client = MongoClient('10.3.20.117', 27017)

# 选择数据库和集合
db = client['chatgpt']
collection = db['chathistory']

def add_data(username, data):
    # 查询是否已存在该用户名的文档
    existing_doc = collection.find_one({'username': username})
    if existing_doc:
        # 如果用户名已存在，则将数据合并到已有文档
        collection.update_one({'_id': existing_doc['_id']}, {'$push': {'history': data}})
    else:
        # 如果用户名不存在，则创建一个新的文档
        collection.insert_one({'username': username, 'history': [data]})

def get_last_10(username):
    # 查询用户名对应数组的最后的10条数据，若不足10条则查询全部
    result = collection.find_one({'username': username})
    if result:
        history = result['history']
        if len(history) <= 10:
            return history
        else:
            return history[-10:]
    else:
        return []


if __name__ == '__main__':
    # 示例用法
    username = "wangjuntong"
    data = {"role":"user","content":"那我是啥"}  # 新增的数据
    add_data(username, data)


    last_10_history = get_last_10(username)
    print(last_10_history)
