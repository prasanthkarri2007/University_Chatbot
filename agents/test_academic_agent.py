from academic_agent import academic_agent

while True:
    question = input("Ask: ")
    response = academic_agent(question)
    print("\nAnswer:")
    print(response["result"])