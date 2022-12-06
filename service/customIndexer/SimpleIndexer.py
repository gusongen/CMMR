import inspect
import os
from typing import Dict, Optional

from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
import clip
from torch import Tensor
import torch
import time
import numpy as np


class SimpleIndexer(Executor):
    """
    A simple indexer that stores all the Document data together in a DocumentArray,
    and can dump to and load from disk.

    To be used as a unified indexer, combining both indexing and searching
    """

    FILE_NAME = 'index.db'
    # FILE_NAME = 'index_f8k.db'

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'ViT-B/32',
        match_args: Optional[Dict] = None,
        table_name: str = 'simple_indexer_table2',
        traversal_right: str = '@r',  # https://docarray.jina.ai/fundamentals/documentarray/access-elements/#index-by-nested-structure
        traversal_left: str = '@r',
        # device: str = 'cpu',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs,
    ):
        """
        Initializer function for the simple indexer

        To specify storage path, use `workspace` attribute in executor `metas`
        :param match_args: the arguments to `DocumentArray`'s match function
        :param table_name: name of the table to work with for the sqlite backend
        :param traversal_right: the default traversal path for the indexer's
        DocumentArray
        :param traversal_left: the default traversal path for the query
        DocumentArray
        """
        super().__init__(**kwargs)

        self._match_args = match_args or {}
        # workspace = self.workspace if self.workspace is not None else './workspace'
        self._index = DocumentArray(
            storage='sqlite',
            config={
                'connection': os.path.join(self.workspace, SimpleIndexer.FILE_NAME),
                # 'connection': ,# TODO fix one specific db
                'table_name': table_name,
                # 'synchronous': 'NORMAL'
                # 'journal_mode': 'WAL'
            },
        )  # with customize config
        print('@@@@ _index len', len(self._index))
        self.logger = JinaLogger(self.metas.name)
        self.default_traversal_right = traversal_right
        self.default_traversal_left = traversal_left

        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        self.device = device

        model, preprocessor = clip.load(self.pretrained_model_name_or_path, device=device)

        self.preprocessor = preprocessor
        self.model = model

    @property
    def table_name(self) -> str:
        return self._index._table_name

    @requests(on='/index')
    def index(
        self,
        docs: 'DocumentArray',
        **kwargs,
    ):
        """All Documents to the DocumentArray
        :param docs: the docs to add
        """
        t1 = time.time()
        if docs:
            self._index.extend(docs)  # 将doc往sqlite里存
        t2 = time.time()
        print(t2 - t1)
        print(t1)
        print(t2)
        print('@@@@ _index len', len(self._index))

    @requests(on='/search')
    def search(
        self,
        docs: 'DocumentArray',
        parameters: Optional[Dict] = None,
        **kwargs,
    ):
        """Perform a vector similarity search and retrieve the full Document match

        :param docs: the Documents to search with
        :param parameters: the runtime arguments to `DocumentArray`'s match
        function. They overwrite the original match_args arguments.
        """
        match_args = (
            {**self._match_args, **parameters}
            if parameters is not None
            else self._match_args
        )
        # use_batch = True  # todo add to param
        # batch_size = 64
        traversal_right = parameters.get(
            'traversal_right', self.default_traversal_right
        )
        traversal_left = parameters.get('traversal_left', self.default_traversal_left)
        match_args = SimpleIndexer._filter_match_params(docs, match_args)
        # print('in indexer',docs[traversal_left].embeddings.shape, self._index[traversal_right])
        queries: DocumentArray = docs[traversal_left]
        stored_docs: DocumentArray = self._index[traversal_right].batch(batch_size=parameters.get('batch_size', 256))
        add_dummy_unknow_prompt = parameters.get("add_dummy_unknow_prompt", False)
        relative_score = parameters.get("relative_score", True)
        maxCount = parameters.get('maxCount', None)
        if maxCount != None:
            maxCount = int(maxCount)
        print('@maxCount', maxCount)
        thod = parameters.get("thod", 0)
        if add_dummy_unknow_prompt:
            unknow_prompt_embedding = queries[-1].embedding
            queries = queries[:-1]
        doc_ids = parameters.get("doc_ids")
        t1 = time.time()
        with torch.inference_mode():
            t1_00 = time.time()
            for query in queries:  # 将每个text和所有feature计算相似度
                result = []
                query_features = query.embedding
                if not isinstance(query_features, torch.Tensor):
                    query_features = torch.Tensor(query_features)

                if query_features.dim() == 1:
                    query_features = query_features.unsqueeze(0)
                if add_dummy_unknow_prompt:
                    query_features = torch.cat([query_features, unknow_prompt_embedding], dim=0)
                query_features = query_features.float().to(self.device)
                for batch_sd in stored_docs:
                    # if doc_ids is not None and sd.uri not in doc_ids:
                    #     continue
                    images_features = [sd.embedding for sd in batch_sd]
                    print('images len', len(images_features))
                    t1_0 = time.time()
                    # 将np的emb转tensor
                    # tensor_images_features = [Tensor(image_features) for image_features in images_features]
                    # tensor_images_features = Tensor(images_features)
                    # if tensor_images_features.dim() == 1:
                    #     tensor_images_features = torch.unsqueeze(tensor_images_features, dim=0)
                    tensor_images_features = torch.from_numpy(np.vstack(images_features)).to(self.device)
                    t1_1 = time.time()
                    probs = self.score(tensor_images_features, query_features, relative=relative_score)
                    # print("@probs", probs)
                    result.extend([{
                        "score": prob[0],
                        "unkown_score":prob[1] if len(prob) > 1 else 0,
                        # "index": i, #frame id
                        "uri": sd.uri,
                        "id": sd.id
                    }for (sd, prob) in zip(batch_sd, probs)])
                    t1_2 = time.time()
                    print("tensor cost:", t1_1 - t1_0)
                    print("part score cost:", t1_2 - t1_1)
                    print(t1_0)
                    print(t1_1)
                    print(t1_2)
                t2 = time.time()
                print('score cost:', t2 - t1)
                print('@result len ', len(result))
                ranked_result = self.getRankedSearchResult(result, thod=thod, maxCount=maxCount)
                t3 = time.time()
                print('rank cost:', t3 - t2)
                docArr = DocumentArray.empty(len(ranked_result))
                for i, (doc, res) in enumerate(zip(docArr, ranked_result)):
                    doc.tags["score"] = float(res["score"])
                    doc.tags["uri"] = res["uri"]
                    doc.tags["id"] = res["id"]
                query.matches = docArr
        print('@@@@ _index len', len(self._index))

    def getRankedSearchResult(self, result: list, maxCount: Optional[int] = None, thod: float = 0, filter_unknow: bool = True):

        return sorted(
            filter(lambda x: x["score"] >= thod and (not filter_unknow or x["unkown_score"] < x["score"]), result
                   ), key=lambda x: -x["score"])[:maxCount]
        # tic = time.time()
        # tmp = np.array(list(filter(lambda x: x["score"] >= thod and (not filter_unknow or x["unkown_score"] < x["score"]), result
        #                            )))
        # print('@rank.filter cost', time.time() - tic)
        # tic = time.time()
        # if maxCount is not None:
        #     idx = np.argpartition([-i["score"] for i in tmp], maxCount)[:maxCount]
        #     tmp = tmp[idx]
        # print('@rank.sort cost', time.time() - tic)
        # return tmp

    def score(self, image_features, queries_feature, relative=True):  # TODO fine tune change here
        """_summary_

        Args:
            image_features (_type_): _description_
            text_features (_type_): _description_
            relative (bool, optional): socre is relative value which softmax by queries,
            and the sum of each row is 1
            eg. relative=False get tensor([0.1000, 0.8000, 0.9000]) 
                relative=True get tensor([0.1909, 0.3844, 0.4248]
            . Defaults to True. 

        Returns:
            _type_: _description_
        """
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        queries_feature = queries_feature / queries_feature.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        similarity_matrix = image_features @ queries_feature.t()
        # print("@ similarity_matrix", similarity_matrix)
        if relative:
            logit_scale = self.model.logit_scale.exp()
            # print('@logit_scale', logit_scale)
            logits_per_image = logit_scale * similarity_matrix
            probs = logits_per_image.softmax(dim=-1)
        else:
            probs = similarity_matrix

        probs = probs.cpu().detach().numpy()

        # print(" img Label probs:", probs)
        return probs

    @staticmethod
    def _filter_match_params(docs, match_args):
        # get only those arguments that exist in .match
        args = set(inspect.getfullargspec(docs.match).args)
        args.discard('self')
        match_args = {k: v for k, v in match_args.items() if k in args}
        return match_args

    @requests(on='/delete')
    def delete(self, parameters: Dict, **kwargs):
        """Delete entries from the index by id

        :param parameters: parameters to the request
        """
        deleted_ids = parameters.get('ids', [])
        if len(deleted_ids) == 0:
            return
        del self._index[deleted_ids]
        print('@@@@ _index len', len(self._index))

    @requests(on='/update')
    def update(self, docs: DocumentArray, **kwargs):
        """Update doc with the same id, if not present, append into storage

        :param docs: the documents to update
        """

        for doc in docs:
            try:
                self._index[doc.id] = doc
            except IndexError:
                self.logger.warning(
                    f'cannot update doc {doc.id} as it does not exist in storage'
                )
        print('@@@@ _index len', len(self._index))

    @requests(on='/fill_embedding')
    def fill_embedding(self, docs: DocumentArray, **kwargs):
        """retrieve embedding of Documents by id

        :param docs: DocumentArray to search with
        """
        for doc in docs:
            doc.embedding = self._index[doc.id].embedding
        print('@@@@ _index len', len(self._index))

    @requests(on='/clear')
    def clear(self, **kwargs):
        """clear the database"""
        self._index.clear()
        print('@@@@ _index len', len(self._index))

    @requests(on='/get_db')
    def get_db(self, **kwargs):
        """check the database"""
        print('@@@@ _index len', len(self._index))
        return self._index

    @requests(on='/echo')
    def get_db(self, **kwargs):
        """check the database"""
        print('@@@@ _index len', len(self._index))

    @requests(on='/get_db_len')
    def get_db(self, **kwargs):
        """check the database"""
        return {'len': len(self._index)}
