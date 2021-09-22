from dgl.function.reducer import SimpleReduceFunction

def prod(msg, out):
    return SimpleReduceFunction("prod", msg, out)
