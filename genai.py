import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# โหลดโมเดลและ tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
tokenizer.pad_token = tokenizer.eos_token
def read_all_text_files(folder="."):
    content_list = []
    allowed_extensions = [".txt", ".py", ".md", ".json", ".csv"]

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) and any(filename.endswith(ext) for ext in allowed_extensions):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    content_list.append(f"--- FILE: {filename} ---\n{content[:1000]}")  # ตัดไฟล์ละ 1000 ตัวอักษร
            except Exception as e:
                print(f"Error reading file {filename}: {e}")

    return "\n\n".join(content_list)

def chat_with_bot(chat_history_ids, user_input):
    encoded_dict = tokenizer.encode_plus(
        user_input + tokenizer.eos_token,
        return_tensors='pt',
        padding=True,
        truncation=True,
    )
    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']

    if chat_history_ids is not None:
        input_ids = torch.cat([chat_history_ids, input_ids], dim=-1)
        attention_mask = torch.ones_like(input_ids)  # ไม่มี padding จริง จึง mask เป็น 1

    chat_history_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )

    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

def main():
    print("💬 พิมพ์ 'read data' เพื่อให้ AI อ่านไฟล์ในโฟลเดอร์")
    print("💬 พิมพ์ข้อความอื่นเพื่อแชตกับ AI")
    print("💬 พิมพ์ 'exit' หรือ 'quit' เพื่อออก\n")

    chat_history_ids = None
    folder_data = ""

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("👋 บ๊ายบายครับ!")
            break

        if user_input.lower() == "read data":
            print("📁 กำลังอ่านไฟล์ในโฟลเดอร์...")
            folder_data = read_all_text_files()
            if not folder_data:
                print("⚠️ ไม่พบไฟล์ที่อ่านได้ในโฟลเดอร์นี้")
            else:
                prompt = f"ช่วยสรุปข้อมูลต่อไปนี้ให้หน่อย:\n{folder_data[:2000]}"  # ตัดให้สั้นพอประมาณ
                chat_history_ids = None  # เริ่มแชตใหม่
                response, chat_history_ids = chat_with_bot(chat_history_ids, prompt)
                print("AI:", response, "\n")

        else:
            # ถ้าเริ่มต้นด้วย hi ให้รีเซ็ตแชตและทักทาย
            if user_input.lower() == "hi":
                chat_history_ids = None
                response, chat_history_ids = chat_with_bot(chat_history_ids, "Hi, I am your assistant. How can I help you?")
                print("AI:", response, "\n")
            else:
                # แชตต่อจากประวัติเดิม
                response, chat_history_ids = chat_with_bot(chat_history_ids, user_input)
                print("AI:", response, "\n")

if __name__ == "__main__":
    main()

