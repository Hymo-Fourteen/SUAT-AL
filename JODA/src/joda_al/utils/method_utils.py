from joda_al.Selection_Methods.method_regestry import LOSS_MODULE_METHODS


def is_loss_learning(method):
    if "->" in method:
        methods = method.split("->")
        if any([method in LOSS_MODULE_METHODS for method in methods]):
            return True
    else:
        if method in LOSS_MODULE_METHODS:
            return True
    return False


def is_method_of(method, method_list):
    if "->" in method:
        methods = method.split("->")
        if any([method in method_list for method in methods]):
            return True
    else:
        if method in method_list:
            return True
    return False