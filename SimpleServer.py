import time
from pythonping import ping
from flask import Flask, request
import threading

app = Flask(__name__)

def createFile(filename: str) -> None:
    with open("/app/data/"+filename, "+w") as f:
        f.write("test")

# def check_dns_servers():
#     dns_servers = ['1.1.1.1', '8.8.8.8']
#     while True:
#         for dns_server in dns_servers:
#             ping(dns_server, verbose = True, count = 2)
#         time.sleep(5)

@app.route('/createfile')
def health_check():
    createFile(request.args.get("filename"))
    return 'OK', 200

@app.route('/')
def start():
    return 'OK', 200

if __name__ == '__main__':
    # flask_thread = threading.Thread(target=check_dns_servers)
    # flask_thread.start()
    app.run(debug=True, port=8080, host='0.0.0.0')