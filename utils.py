import time
def time_since(since):
    now = time.time()
    s = now - since
    m = s // 60
    s -= 60 * m

    return "%d m %d s" % (m, s)