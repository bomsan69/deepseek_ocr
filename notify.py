import json
import requests


_MAIL_URL = 'http://apisvr.boranet.net:3300/api/v1/send'
_MAIL_HEADERS = {
    'Content-Type': 'application/json',
    'api_key': 'JDJiJDEyJHdOTk96N1lJSzJUZXFGQVRwbHhSeS5GZmxuNGtYaURSVGRwaFlFNS5uelo1LlNUR25xL2tx',
}



def send_mail(to, title, message):
    param = {'to': to, 'title': title, 'content': message}
    res = requests.post(_MAIL_URL, headers=_MAIL_HEADERS, data=json.dumps(param))
    return res.json()


