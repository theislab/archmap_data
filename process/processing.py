import pre
import post


class Process:

    def __init__(self, adata):
        self.post = post.Post(adata)
        self.pre = pre.Pre(adata)
