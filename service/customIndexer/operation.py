"""
fast way to manupalte executor without flow
"""
import sys
import time
import click
from jina import Document, DocumentArray, Flow, Client
from tqdm import tqdm
sys.path.append('/home/gusongen/mm_search_baseline')
sys.path.append('/home/gusongen/mm_search_baseline/service')
from SimpleIndexer import SimpleIndexer
import os
from dataset import input_index_data
import torch

from service.customClipImage.clip_image import CLIPImageEncoder
from service.customClipText.clip_text import CLIPTextEncoder

MAX_DOCS = int(os.environ.get("JINA_MAX_DOCS", 10000))
cur_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_QUERY_IMAGE = 'toy-data/images/1000268201_693b08cb0e.jpg'
DEFAULT_QUERY_TEXT = 'a black dog and a spotted dog are fighting'


def config():
    os.environ.setdefault('JINA_WORKSPACE', os.path.join(cur_dir, 'workspace'))
    os.environ.setdefault(
        'JINA_WORKSPACE_MOUNT',
        f'{os.environ.get("JINA_WORKSPACE")}:/workspace/workspace')
    os.environ.setdefault('JINA_LOG_LEVEL', 'INFO')
    os.environ.setdefault('JINA_PORT', str(45678))


def test_index_and_query(indexer=None):
    tic = time.time()
    # f = Flow().add(name='myexec1', uses=SimpleIndexer)
    if indexer is None:
        indexer = SimpleIndexer()
    # indexer.clear()
    # TODO  multi  image
    image_doc = Document(uri='toy-data/images/1000268201_693b08cb0e.jpg',
                         modality='image')
    image_doc.load_uri_to_image_tensor()
    docs = DocumentArray(image_doc)
    # image_doc.convert_image_buffer_to_blob()
    # image_doc.blob = np.array(image_doc.blob).astype(np.uint8)
    img_executor = CLIPImageEncoder()
    img_executor.encode(docs, parameters={"uid": '111111', "maxCount": 1})

    # with f:
    #     f.post(on='/index', inputs=docs, on_done=print)
    # off line index
    indexer.index(docs)

    # 搜索
    query_text = 'a girl run to a wooden house'
    text = DocumentArray([Document(text=query_text)])
    text_executor = CLIPTextEncoder()
    text_executor.encode(text, parameters={"uid": '111111', "maxCount": 1})
    # with f:
    #     resp = f.post(on='/search', inputs=docs, on_done=print)
    resp = indexer.search(text, parameters={"uid": '111111', "maxCount": 1})
    print(resp)
    print('@ total time ', time.time() - tic)
    print('@ db size', len(indexer.get_db()))


def test_query(indexer=None):
    tic = time.time()
    if indexer is None:
        indexer = SimpleIndexer()
    # query
    query_text = 'this is a photo of a dog'
    text = DocumentArray([Document(text=query_text)])
    text_executor = CLIPTextEncoder()
    text_executor.encode(text, parameters={})
    # with f:
    #     resp = f.post(on='/search', inputs=docs, on_done=print)
    resp = indexer.search(text, parameters={})
    print(resp)
    print('@ total time ', time.time() - tic)
    print('@ db size', len(indexer.get_db()))


def test_img_query(indexer=None):
    tic = time.time()
    if indexer is None:
        indexer = SimpleIndexer()
    # query
    doc = DocumentArray([Document(uri='/home/gusongen/mm_search_baseline/toy-data/images/image_0021.jpg',
                                  modality='image').load_uri_to_image_tensor()])

    text_executor = CLIPImageEncoder()
    text_executor.encode(doc, parameters={'add_dummy_unknow_prompt': True})
    text_executor = CLIPTextEncoder()
    text_executor.encode(doc, parameters={'add_dummy_unknow_prompt': True})
    # with f:
    #     resp = f.post(on='/search', inputs=docs, on_done=print)
    resp = indexer.search(doc, parameters={'add_dummy_unknow_prompt': True})
    print(resp)
    print('@ total time ', time.time() - tic)
    print('@ db size', len(indexer.get_db()))


def clear(indexer=None):
    tic = time.time()
    if indexer is None:
        indexer = SimpleIndexer()
    print('@ db size', len(indexer.get_db()))
    indexer.clear()
    print('@ db size after cleaning', len(indexer.get_db()))
    print('@ total time ', time.time() - tic)
    return indexer


def get_db_size(indexer=None):
    tic = time.time()
    if indexer is None:
        indexer = SimpleIndexer()
    print('@ db size', len(indexer.get_db()))
    print('@ total time ', time.time() - tic)
    return indexer


@click.command()
@click.option("--indexer", default=None)
@click.option("--num_docs", "-n", default=None)
@click.option('--request_size', '-s', default=64)
@click.option('--data_set', '-d', type=click.Choice(['f30k', 'f8k', 'toy-data'], case_sensitive=False), default='toy-data')
@click.option("--device", default='cuda:1' if torch.cuda.is_available() else 'cpu')
@click.option("--by_flow", default=True, type=bool)
def index(indexer, num_docs, request_size, data_set, device, by_flow):
    """
    num_docs : import all image if num_docs is None
    """
    tic = time.time()
    if indexer is None and not by_flow:
        indexer = SimpleIndexer(device=device)
        clear(indexer)
    img_executor = CLIPImageEncoder(device=device, batch_size=64)
    # todo make it batch
    total_size = 0
    if by_flow:
        port = 45680
        # 创建 Jina 客户端
        c = Client(host=f"grpc://localhost:{port}")
    for idx, docs in enumerate(tqdm(input_index_data(num_docs, request_size, data_set))):
        print(f'@batch {idx}')
        # print(img, _)
        # docs = DocumentArray(img_doc)
        img_executor.encode(docs, parameters={})
        if by_flow:
            c.post('/index', inputs=docs, parameters={})
            print('@ db size', len(c.post('/get_db')))
        else:
            indexer.index(docs)
            print('@ db size', len(indexer.get_db()))

        total_size += sum([1 if d.tensor is not None else 0 for d in docs])
    print('@finish')

    print('@ total size ', total_size)
    print('@ total time ', time.time() - tic)
    print(f'@ avg time { (time.time() - tic) / total_size:.2f} s per img',)


if __name__ == '__main__':
    # todo
    os.environ['JINA_DEFAULT_WORKSPACE_BASE'] = './workspace'  # the directory to store the indexed data
    config()  # set environment first ,or jina can't load sqlite appropriate
    # clear()
    # index()
    # get_db_size()
    # test_query()
    test_img_query()
    # test_query()
    # test_query()

# todo search threshold
