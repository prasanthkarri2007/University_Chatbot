from agents.router_agent import router_agent

while True:

    question = input("\nAsk: ")

    if question.lower() in ["exit", "quit"]:
        break

    router_agent(question)