from typing import Optional, Tuple, Dict

import torch
from docarray import DocumentArray, Document
from jina import Executor, requests
from jina.logging.logger import JinaLogger
from transformers import CLIPFeatureExtractor, CLIPModel
import numpy as np
import clip
from PIL import Image
import time


class CLIPImageEncoder(Executor):
    """Encode image into embeddings using the CLIP model."""

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'ViT-B/32',
        # device: str = 'cpu',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        batch_size: int = 32,
        traversal_paths: str = '@r',
        *args,
        **kwargs,
    ):
        """
        :param pretrained_model_name_or_path: Can be either:
            - A string, the model id of a pretrained CLIP model hosted
                inside a model repo on huggingface.co, e.g., 'openai/clip-vit-base-patch32'
            - A path to a directory containing model weights saved, e.g.,
                ./my_model_directory/
        :param base_feature_extractor: Base feature extractor for images.
            Defaults to ``pretrained_model_name_or_path`` if None
        :param use_default_preprocessing: Whether to use the `base_feature_extractor` on
            images (tensors) before encoding them. If you disable this, you must ensure
            that the images you pass in have the correct format, see the ``encode``
            method for details.
        :param device: Pytorch device to put the model on, e.g. 'cpu', 'cuda', 'cuda:1'
        :param traversal_paths: Default traversal paths for encoding, used if
            the traversal path is not passed as a parameter with the request.
        :param batch_size: Default batch size for encoding, used if the
            batch size is not passed as a parameter with the request.
        """
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.traversal_paths = traversal_paths
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        self.logger = JinaLogger(self.__class__.__name__)

        self.device = device

        model, preprocessor = clip.load(self.pretrained_model_name_or_path, device=device)

        self.preprocessor = preprocessor
        self.model = model
        # self.model.to(self.device).eval()

    @requests(on=["/index",
                  "/search"])
    def encode(self, docs: DocumentArray, parameters: dict, **kwargs):
        t1 = time.time()
        print('clip_image encode', t1)
        document_batches_generator = DocumentArray(
            filter(
                lambda x: x is not None and x.tensor is not None and x.modality == 'image', docs[parameters.get('traversal_paths', self.traversal_paths)],
            )
        ).batch(batch_size=parameters.get('batch_size', self.batch_size))
        with torch.inference_mode():
            for batch_docs in document_batches_generator:
                print('in for')
                images_embedding = self.model.encode_image(torch.stack(
                    [self.preprocessor(Image.fromarray(d.tensor))for d in batch_docs]
                ).to(self.device)).cpu().numpy()
                for doc, image_embedding in zip(batch_docs, images_embedding):
                    doc.embedding = np.array(image_embedding).astype('float32')
        t2 = time.time()
        print('clip_image encode end', t2 - t1, t2)


if __name__ == '__main__':
    image_doc = Document(uri='toy-data/images/1000268201_693b08cb0e.jpg',
                         modality='image')
    image_doc.load_uri_to_image_tensor()
    executor = CLIPImageEncoder()
    executor.encode(DocumentArray(image_doc), parameters={"uid": '111111', "maxCount": 1})
