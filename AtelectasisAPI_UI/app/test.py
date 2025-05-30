import requests

url = "http://localhost:8000/upload/"
with open("1.png", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print(response.status_code, response.json())
