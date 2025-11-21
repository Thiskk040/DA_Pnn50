import os
import google.generativeai as genai

API_KEY = "AIzaSyBr7DeC-e1yOwHlkYQXaFjNAzU3MCrLBt8"
genai.configure(api_key=API_KEY)


def read_all_text_files(folder="."):
    content_list = []
    allowed_extensions = [".txt", ".py", ".md", ".json", ".csv"]

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        if os.path.isfile(file_path) and any(filename.endswith(ext) for ext in allowed_extensions):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    content_list.append(f"\n--- FILE: {filename} ---\n{content}")
            except Exception as e:
                print(f" Error can not read file: {filename} → {e}")

    return "\n".join(content_list)


def start_gemini_chat(initial_context):
    model = genai.GenerativeModel("gemini-1.5-flash")

    chat = model.start_chat(history=[
        {"role": "user", "parts": "These all data in my folder"},
        {"role": "model", "parts": "Okay just send me"},
        {"role": "user", "parts": initial_context}
    ])

    return chat


def main():
    print("📁 Loading folder...")
    folder_data = read_all_text_files()

    if not folder_data:
        print("⚠️ Not found Folder")
        return

    print("✅ Loading complete you can use ai (Type exit to quit)\n")

    chat = start_gemini_chat(folder_data)

    while True:
        user_input = input("Kay: ")
        if user_input.strip().lower() in ['exit', 'quit']:
            print("Exit ciao..")
            break

        try:
            response = chat.send_message(user_input)
            print("AI :", response.text.strip(), "\n")
        except Exception as e:
            print("❌ Error:", e)


if __name__ == "__main__":
    main()
