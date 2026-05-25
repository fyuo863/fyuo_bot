from core.chat import AgentChat



def main():
    print("AI: ", end="", flush=True)
    
    # 【关键修改】接收端也必须用 for 循环，一边接收一边打印
    # 不能再写 reply = chat(...) 了！
    for chunk in AgentChat.chat(
        "用一句话介绍 Python",
        "你是一个python糕手,回答时先表明自己的身份",
        stream=True
        ):
        print(chunk, end="", flush=True)
        
    print() # 打印完毕后换行

if __name__ == "__main__":
    main()