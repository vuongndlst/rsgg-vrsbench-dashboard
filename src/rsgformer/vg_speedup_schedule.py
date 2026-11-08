"""Ordered microbatches and original effective-batch update boundaries."""
def grouped(items, microbatch):
    assert microbatch > 0
    group = []
    for item in items:
        group.append(item)
        if len(group) == microbatch:
            yield group
            group = []
    if group:
        yield group

def step_boundary(images, effective_batch):
    return images % effective_batch == 0
