import requests

payload = open('client_request/public/payload').read()

response = requests.post("http://localhost:5000", headers={'content-type': 'application/json'}, data=payload)

with open('response.json', 'w') as outfile:
    outfile.write(response.text)
