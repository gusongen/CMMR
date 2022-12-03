"""dev operation"""

from jina import DocumentArray, Executor, requests, Flow


def clear():
    flow = Flow.load_config('service/customIndexer/config.yml')
    with flow:
        flow.post(on='/search')


def get_db_size():
    flow = Flow.load_config('service/customIndexer/config.yml')
    with flow:
        resp = flow.post(on='/get_db')
        print(resp)

if __name__ == '__main__':
    get_db_size()
