import re

def No_hashtag(content):
    return content.count("#") 

def No_url(content):
    return content.count("https://")

def is_retweet(content):
    pattern = re.compile(r"RT.*?:")
    m = pattern.match(content)
    if m!= None:
        return m.group(0)[4:-1]
    else:
        return False
    
def mention_list(content):
    content = content.replace("\n"," ")
    pattern = re.compile(r"@[a-zA-Z0-9_]*")
    result = pattern.findall(content)
    if result != None:
        result = [m[1:] for m in result if m!="@"]
        if is_retweet(content) == False:
            return result
        else:
            return result[1:]
    else:
        return []

def exact_hashtag(content):
    
    content = content.replace("\n"," ")
    pattern = re.compile(r"#[a-zA-Z0-9_]*")
    result = pattern.findall(content)
    if result != None:
        result = [m.split("\\")[0] for m in result]
    else:
        return None
    
    return result

def exact_URL(content):
    
    content = content.replace("\n"," ")
    pattern = re.compile(r"https://t.co/[a-zA-Z0-9]*")
    result = pattern.findall(content)
    if result != None:
        result = [m.split("\\")[0] for m in result]
    else:
        return None
    
    return result

def exact_u(content):
    
    content = content.replace("\n"," ")
    pattern = re.compile(r"\\u[a-zA-Z0-9]*")
    result = pattern.findall(content)
    if result != None:
        result = [m.split("\\")[0] for m in result]
    else:
        return None
    
    return result

def clear_data(content):
    
    content = content.replace("\n","")
    content = re.sub(r"\\u.{4}","<emoji>",content.__repr__())
    
    hashtags = exact_hashtag(content)
    mentions = mention_list(content)
    urls = exact_URL(content)
    rt = is_retweet(content)
    
    if rt != False:
        content = content.replace(rt, "<user>")
    if mentions != None:
        for m in mentions:
            content = content.replace(m, "<user>")
    if urls != None:
        for u in urls:
            content = content.replace(u, "<url>")
    if hashtags != None:
        for h in hashtags:
            content = content.replace(h, "<hashtag>")
 
    return content

def tweet_cate_encode(content):
    
    cate_encode = None
    if is_retweet(content) != False:
        cate_encode = [0, 1]
    else:
        cate_encode = [1, 0]
    entity_encode = [0, 0, 0, 0]
    if len(mention_list(content)) > 0:
        entity_encode[3] = 1
    if No_url(content) > 0:
        entity_encode[1] = 1
    if No_hashtag(content) > 0:
        entity_encode[2] = 1 
    cate_encode.extend(entity_encode)
    return cate_encode

def find_index(node, node_list):
    low = 0
    high = len(node_list)
    mid = 0
    while low < high:
        mid_tmp = mid
        mid = (low + high)//2
        if mid_tmp-mid == 0:
            return "unknown"
        temp = node_list[mid]
        if temp == node:
            return mid
        elif temp > node:
            high = mid
        else:
            low = mid
        # if low >= high:
        #     print(node)
    return "unknown"

