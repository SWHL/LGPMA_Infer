# -*- encoding: utf-8 -*-
from davarocr.davar_common.apis import inference_model, init_model


class LGPMA(object):
    def __init__(self, config_path, model_path):
        self.model = init_model(config_path, model_path)

    def __call__(self, img):
        result = inference_model(self.model, img)[0]
        return result
