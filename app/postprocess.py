import torch.nn as nn

def postprocess(x, store_name, customer_name):
    output_x = x.replace("#@상호명#", store_name)
    output_x = output_x.replace("#@고객이름#", customer_name)
    output_x = output_x.replace("#@위치#", "")
    output_x = output_x.replace("[a-zA-Z]+", "")
    output_x = output_x.replace("\n", " ")
    return output_x