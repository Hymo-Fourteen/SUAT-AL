from joda_al.Selection_Methods.query_method_def import QueryMethod

import importlib, inspect
from setuptools import find_packages
from pkgutil import iter_modules
import sys
import pathlib

__cache__ = {}

__method__registry__ = {}

class SelectionMethodFactor():

    def __init__(self):
        modules=find_modules()
        module_classes = find_classes(modules)
        self.query_map, self.name_map = create_method_map(module_classes)

    def create_query_method(self,keyword):
        return self.query_map[keyword]

    def create_or_get_query_method(self,keyword,kwargs=None):
        if keyword in __cache__:
            return __cache__[keyword]
        __cache__[keyword]=self.query_map[keyword](keyword,kwargs)
        return __cache__[keyword]

    def register_query(self,query_strategy: QueryMethod,keyword :str  ):
        self.query_map[keyword]=query_strategy
        self.name_map[keyword]=query_strategy.__name__

    def deregister_query(self,keyword):
        del self.query_map[keyword]
        del self.name_map[keyword]


"""
def get_modules_in_package(package_name: str):
    files = os.listdir(package_name)
    for file in files:
        if file not in ['__init__.py', '__pycache__']:
            if file[-3:] != '.py':
                continue

            file_name = file[:-3]
            module_name = package_name + '.' + file_name
            for name, cls in inspect.getmembers(importlib.import_module(module_name), inspect.isclass):
                if cls.__module__ == module_name:
                    yield cls

"""


def find_modules():
    modules = set()
    path=str(pathlib.Path(__file__).parent.resolve())
    for pkg in find_packages(path):
        modules.add(pkg)
        pkgpath = path + '/' + pkg.replace('.', '/')
        if sys.version_info.major == 2 or (sys.version_info.major == 3 and sys.version_info.minor < 6):
            for _, name, ispkg in iter_modules([pkgpath]):
                if not ispkg:
                    modules.add(pkg + '.' + name)
        else:
            for info in iter_modules([pkgpath]):
                if not info.ispkg:
                    modules.add(pkg + '.' + info.name)
    return modules


def find_classes(modules):
    queries=[]
    if __name__ != "__main__":
        root=__name__.split(".SelectionMethodFactory")[0]+"."
    else:
        root=""
    for module in modules:
        for name,cls in inspect.getmembers(importlib.import_module(root+module), inspect.isclass):
            if cls.__module__ == root+module and issubclass(cls,QueryMethod):
                try:
                    if cls.keywords is None:
                        continue
                    queries.append((name,cls,cls.keywords))
                except:
                    print(f"{cls} has not keywords ")
                    queries.append((name, cls, [str(cls.__name__)]))
    return queries


def create_method_map(query_tuples):
    method_dict={}
    name_dict={}
    for (name,cls,keywords) in query_tuples:
        for key in keywords:
            method_dict[key]=cls
            name_dict[key]=name

    return method_dict, name_dict

def check_existence(keyword):
    """
    Check if a keyword exists in the query map.
    :param keyword: The keyword to check.
    :return: True if the keyword exists, False otherwise.
    """
    return keyword in __method__registry__
    modules = find_modules()
    module_classes = find_classes(modules)
    query_map, name_map = create_method_map(module_classes)
    if keyword in query_map:
        return True
    else:
        return False

if __name__ =="__main__":
    find_modules()