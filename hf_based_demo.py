# coding=utf-8
from src.logger import LoggerFactory
from src.prompt_concat import GetManualTestSamples, CreateTestDataset
from src.utils import decode_csv_to_json, load_json, save_to_json
from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextIteratorStreamer,
)
from typing import List

import gradio as gr
import logging
import os
import shutil
import torch
import warnings
import pdb
import requests

logger = LoggerFactory.create_logger(name="test", level=logging.INFO)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

config_data = load_json("config/config.json")
model_path = config_data["huggingface_local_path"]
character_path = "./character"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,trust_remote_code=True)

def generate_tts(input_text):
    input_text = input_text.replace('</s>','')
    url = 'http://127.0.0.1:4569/tts'
    response = requests.post(url, json={
        "text": input_text,
    }).json()
    if(response["message"]=="Success."):
        file_url = response["url"]
        file_url = file_url.split(".",1)
        new_file_url = "./GPT-SoVITS"+file_url[1]
        return new_file_url
    else:
        return response["message"]

def update_model(role_name):
    available_role = ['丁小真','冰糖']
    if role_name not in available_role:
        return ''
    url = 'http://127.0.0.1:4569/update_weights'
    response = requests.post(url, json={
        "user_name": role_name,
    }).json()
    if(response["message"]=="Success."):
        return response["message"]
    else:
        return response["message"]

def generate_with_question(question, role_name):
    dialogs = []
    for item in question:
        if(isinstance(item[0], str)):
            dialogs.append(item[0])
        if(isinstance(item[1], str)):
            dialogs.append(item[1])
    question_in = "\n".join(dialogs)

    g = GetManualTestSamples(
        role_name=role_name,
        role_data_path=f"./character/{role_name}.json",
        save_samples_dir="./character",
        save_samples_path=role_name + "_rag.json",
        prompt_path="./prompt/dataset_character.txt",
        max_seq_len=4000
    )
    g.get_qa_samples_by_query(
        questions_query=question_in,
        keep_retrieve_results_flag=True
    )


def create_datasets(role_name):
    testset = []
    role_samples_path = os.path.join("./character", role_name + "_rag.json")

    c = CreateTestDataset(role_name=role_name,
                          role_samples_path=role_samples_path,
                          role_data_path=role_samples_path,
                          prompt_path="./prompt/dataset_character.txt"
                          )
    res = c.load_samples()
    testset.extend(res)
    save_to_json(testset, f"./character/{role_name}_测试问题.json")


def hf_gen(dialog: List, role_name, top_k, top_p, temperature, repetition_penalty, max_dec_len):
    generate_with_question(dialog, role_name)
    create_datasets(role_name)

    json_data = load_json(f"{character_path}/{role_name}_测试问题.json")[0]
    text = json_data["input_text"]
    inputs = tokenizer(text, return_tensors="pt")

    if torch.cuda.is_available():
        model.to("cuda")
        inputs.to("cuda")

    streamer = TextIteratorStreamer(tokenizer, **tokenizer.init_kwargs)
    generation_kwargs = dict(
        inputs,
        do_sample=True,
        top_k=int(top_k),
        top_p=float(top_p),
        temperature=float(temperature),
        repetition_penalty=float(repetition_penalty),
        max_new_tokens=int(max_dec_len),
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    answer = ""
    for new_text in streamer:
        answer += new_text
    yield answer[len(text):]


def generate(chat_history: List, query, role_name, role_desc, top_k, top_p, temperature, repetition_penalty, max_dec_len):
    """generate after hitting "submit" button

    Args:
        chat_history (List): [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n]]. list that stores all QA records
        query (str): query of current round
        top_p (float): only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature (float): strictly positive float value used to modulate the logits distribution.
        max_dec_len (int): The maximum numbers of tokens to generate.

    Yields:
        List: [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n], [q_n+1, a_n+1]]. chat_history + QA of current round.
    """
    assert query != "", "Input must not be empty!!!"
    # apply chat template
    chat_history.append([f"user:{query}", ""])
    for answer in hf_gen(chat_history, role_name, top_k, top_p, temperature, repetition_penalty, max_dec_len):
        chat_history[-1][1] = role_name + ":" + answer
        available_role = ['丁小真','冰糖']
        if role_name in available_role:
            tts_path = generate_tts(answer)
            chat_history.append([None,(tts_path,)])
        yield gr.update(value=""), chat_history


def regenerate(chat_history: List,role_name,role_desc, top_k, top_p, temperature, repetition_penalty, max_dec_len):
    """re-generate the answer of last round's query

    Args:
        chat_history (List): [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n]]. list that stores all QA records
        top_p (float): only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature (float): strictly positive float value used to modulate the logits distribution.
        max_dec_len (int): The maximum numbers of tokens to generate.

    Yields:
        List: [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n]]. chat_history
    """
    assert len(chat_history) >= 1, "History is empty. Nothing to regenerate!!"
    if len(chat_history[-1]) > 1:
        chat_history[-1][1] = ""
    # apply chat template
    for answer in hf_gen(chat_history, role_name, top_k, top_p, temperature, repetition_penalty, max_dec_len):
        available_role = ['丁小真','冰糖']
        if role_name in available_role:
            tts_path = generate_tts(answer)
            chat_history[-2][1] = role_name + ":" + answer
            chat_history[-1][1] = (tts_path,)
        else:
            chat_history[-1][1] = role_name + ":" + answer
        yield gr.update(value=""), chat_history


def clear_history():
    """clear all chat history

    Returns:
        List: empty chat history
    """
    torch.cuda.empty_cache()
    return []


def generate_file(file_obj, role_info):
    role_name = os.path.basename(file_obj).split(".")[0]
    file_name = f"{role_name}.csv"
    new_path = os.path.join(character_path, file_name)
    shutil.copy(file_obj, new_path)
    decode_csv_to_json(os.path.join(character_path, role_name + ".csv"), role_name, role_info,
                       os.path.join(character_path, role_name + ".json"))
    gr.Info(f"{role_name}生成成功")

def main():
    # launch gradio demo
    with gr.Blocks(theme="soft") as demo:
        gr.Markdown("""# Index-1.9B RolePlay Gradio Demo""")

        with gr.Row():
            with gr.Column(scale=1):
                top_k = gr.Slider(0, 10, value=5, step=1, label="top_k")
                top_p = gr.Slider(0.0, 1.0, value=0.8, step=0.1, label="top_p")
                temperature = gr.Slider(0.1, 2.0, value=0.85, step=0.1, label="temperature")
                repetition_penalty = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="repetition_penalty")
                max_dec_len = gr.Slider(1, 4096, value=512, step=1, label="max_dec_len")
                file_input = gr.File(label="上传角色对话语料(.csv)")
                role_description = gr.Textbox(label="Role Description", placeholder="输入角色描述", lines=2)
                upload_button = gr.Button("生成角色!")
                upload_button.click(generate_file, inputs=[file_input, role_description])
            with gr.Column(scale=10):
                chatbot = gr.Chatbot(bubble_full_width=False, height=500, label='Index-1.9B')
                with gr.Row():
                    # role_name = gr.Textbox(label="Role name", placeholder="Input your rolename here!", lines=2, value="冰糖")
                    role_name = gr.Dropdown(
                        ['冰糖','丁小真','牢大','龙玉涛'],value="冰糖"
                    )
                user_input = gr.Textbox(label="User", placeholder="Input your query here!", lines=2, interactive=True)
                with gr.Row():
                    submit = gr.Button("🚀 Submit")
                    clear = gr.Button("🧹 Clear")
                    regen = gr.Button("🔄 Regenerate")

        submit.click(generate, inputs=[chatbot, user_input, role_name, role_description, top_k, top_p, temperature,
                                       repetition_penalty, max_dec_len],
                     outputs=[user_input, chatbot])
        regen.click(regenerate,
                    inputs=[chatbot, role_name, role_description, top_k, top_p, temperature, repetition_penalty,
                            max_dec_len],
                    outputs=[user_input, chatbot])
        clear.click(clear_history, inputs=[], outputs=[chatbot])
        role_name.input(update_model,inputs=role_name,outputs=[]).then(clear_history,inputs=[], outputs=[chatbot])

    demo.queue()
    demo.launch(server_name="0.0.0.0",
                server_port=8002,
                show_error=True,
                share=False)


if __name__ == "__main__":
    main()
