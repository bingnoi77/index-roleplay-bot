import pdb
import requests

url = 'http://127.0.0.1:4568/tts'
response = requests.post(url, json={
        "text": "一个男人半夜遇上抢劫，却反过来将小混混打得落花流水，并且从他们身上得到了一件宝物。这个宝物能够让他看到人体内的秘密，他因此来到了一家古玩店，却发现这件宝物连二十都不值。就在他准备换一家古玩店的时候，紫色的提示框出现在他的眼前，告诉他问题出在佛像内部。",
            "cid": "0",
                "eid": "3",
                }).json()
if (response["message"] == "Success."):
        print(response["url"])
else:  # response["url"]==""
        print(response["message"])


