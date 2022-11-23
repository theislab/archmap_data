import pre
import post


class Process:

    def __init__(self, adata, config):
        self.config = config
        self.adata = adata
        self.post = post.Post(adata, config)
        self.pre = pre.Pre(config)
