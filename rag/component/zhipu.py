from zhipuai import ZhipuAI


class Zhipu:
    def __init__(self, api_key="a3a531fa24a2da257c09b9b8fd85d826.P6MoZZTHWTOPpkaQ"):
        self.model = ZhipuAI(api_key=api_key)

    def get_model(self):
        return self.model
