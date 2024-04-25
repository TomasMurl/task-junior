import requests
import json
from multiprocessing import Process

data = "London is the capital of Great Britain"
data = {"message":data}
data = json.dumps(data)
headers = {"Content-type": "application/json"}

def f(y):
	for i in range(10):
		# requests.post("http://127.0.0.1:80/predict", data=data, headers=headers)
		# try:
		r = requests.post("http://localhost:8080/predict", data=data, headers=headers)
		print(f"[{r.content}]")
		# except Exception:
		# 	print("Error!")
		


if __name__ == '__main__':
	print('start')
	p = []
	print('thread creating')
	for i in range(30):
		print(f'creating {i}')
		p.append(Process(target=f, args=(i,)))
	for i in range(30):
		print(f'starting {i}')
		p[i].start()
	for i in range(30):
		print(f'waiting {i}')
		p[i].join()

