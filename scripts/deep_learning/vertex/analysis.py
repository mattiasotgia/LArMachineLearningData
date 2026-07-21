# analysis.py

from functools import partial

def flatten_model(module):
    children = list(module.children())
    if len(children) == 0:
        return [module]
    else:
        flat_model = []
        for child in children:
            flat_model += flatten_model(child)
        return flat_model


class Hook:
    def __init__(self, id, module, func):
        self.id = id
        self.name = module.__class__.__name__
        self.hook = module.register_forward_hook(partial(func, self))
    
    def remove(self):
        self.hook.remove()
    
    def __del__(self):
        self.remove()


def append_stats(hook, module, input, output):
    if not module.training:
        return
    if not hasattr(hook, 'stats'):
        hook.stats = ([],[],[])
    means, stds, hists = hook.stats
    means.append(output.data.mean())
    stds.append(output.data.std())
    hists.append(output.data.histc(40, -5, 5))