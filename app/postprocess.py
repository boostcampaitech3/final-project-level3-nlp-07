import torch.nn as nn

def postprocess(x, store_name, customer_name):
    output_x = x.replace("#@상호명#", store_name)
    output_x = output_x.replace("#@고객이름#", customer_name)
    output_x = output_x.replace("#@위치#", "")
    output_x = output_x.replace("<unk>", "")
    output_x = output_x.replace("</s>", "")
    output_x = output_x.replace("\n", " ")
    output_x = output_x.replace("[a-zA-Z]+", "")
    return output_x