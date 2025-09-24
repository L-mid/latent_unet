
def enforce_timeout(holdoff=5):
    import torch, time
    
    # code
    time.sleep(holdoff)
    return 0.0