import asyncio
from aiohttp import web
from CustomModel import Model
from elasticsearch import Elasticsearch as es
import os

def init_func(argv):
    app = web.Application()
    app.router.add_post('/predict', predict)
    app.router.add_get('/', index)

    model_path   = os.environ["model_path"]

    app["model"] = Model(model_path, model_path)
    app["elk"]   = es(
        {
            'host': 'elastic.local',
            'port': 9200,
            'scheme': 'https',
            'use_ssl':True,
        },
       ca_certs='/home/cert/ca.crt',
       http_auth=("elastic", "+IwGE5bHI34G-4pIrVOc")
    )

    return app

def strip_event(json_obj):
    temp_data = str(json_obj["message"])
    temp_data = temp_data.replace("\t"," ").replace("\n","")
    return(temp_data)

async def index(request):
    return web.json_response({"state":"alive"}, status=200)

async def predict(request):
    try:
        raw_json = await request.json()
        data     = strip_event(raw_json)
        print(f"data {data}")
        answer   = request.app["model"].predict(data)
        raw_json["ML answer"] = answer
        print(f"answer {answer}")
        request.app["elk"].index(index="ml_index", document=raw_json)

        return web.json_response({"state":"alive"}, status=200)

    except Exception as e:
        print(e)
        return web.json_response({'error': 'An error occurred while processing the request.'}, status=500)

