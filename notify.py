import json
import os

import requests

_MAIL_URL = os.environ["MAIL_URL"]
_MAIL_HEADERS = {
    'Content-Type': 'application/json',
    'api_key': os.environ["MAIL_API_KEY"],
}



def send_mail(to, title, message):
    param = {'to': to, 'title': title, 'content': message}
    res = requests.post(_MAIL_URL, headers=_MAIL_HEADERS, data=json.dumps(param))
    return res.json()


