# Update the chat history with the latest message based on the role and content
def update_history(history):
    full_history = ""
    index = 0

    if len(history) > 0:
        for message in history:
            print(index)
            print(message)
            full_history = full_history + "\n" + message["role"].capitalize() + ": " + message["content"]

            index = index + 1

    return full_history
