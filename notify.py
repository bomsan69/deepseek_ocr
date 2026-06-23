import json
import os

import requests

def send_mail(to, title, message):
    url = os.environ["MAIL_URL"]
    headers = {
        'Content-Type': 'application/json',
        'api_key': os.environ["MAIL_API_KEY"],
    }
    param = {'to': to, 'title': title, 'content': message}
    res = requests.post(url, headers=headers, data=json.dumps(param))
    return res.json()


