from chat_engine import ask_ai

while True:
    question = input("Ask: ")

    result = ask_ai(question)

    print("\nAnswer:")
    print(result["result"])