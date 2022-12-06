from docarray import Document
from jina import Flow, DocumentArray
import os
import glob
from jina.types.request.data import DataRequest


def config():
    os.environ['JINA_PORT'] = '45680'  # the port for accessing the RESTful service, i.e. http://localhost:45678/docs
    os.environ['JINA_WORKSPACE'] = './workspace'  # the directory to store the indexed data
    os.environ['TOP_K'] = '20'  # the maximal number of results to return


config()

f = (
    Flow(protocol="grpc", port=os.environ['JINA_PORT'])
    .add(
        uses='imageLoader/config.yml',
        # uses_requests={"/index": "extract"},
        name='image_loader',
    )
    .add(
        name='image_encoder',
        uses="customClipImage/config.yml",
        # uses_requests={"/index": "encode",
        #                "/search": "encode" #bug
        #                }
        # needs='image_loader'
    ).add(
        name='text_encoder',
        uses="customClipText/config.yml",
        uses_requests={"/search": "encode"},
    ).add(
        name='indexer',
        uses="customIndexer/config.yml",
        uses_metas={"workspace": os.environ['JINA_WORKSPACE']},
        # needs=['image_encoder', 'text_encoder']
    )
)
# todo   replicas=3
f.save_config('flow.yml')
with f:
    f.block()  # run forever
