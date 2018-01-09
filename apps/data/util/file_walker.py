import os
import json


class FileWalker:
    def get_dir_tree_json(self, path):
        """
        遍历文件树，获得json结果
        e.g.
        {
        "label": ".DS_Store",
         "util": {
                "label": "PageHelper.java"
                },
         "dao": {
                "impl": {
                        "label": "UserDAOImpl.java"
                        },
                "label": "DAOHelper.java"
        },
        "models": {
            "label": "User.java"
            },
        "service": {
            "impl": {
                "label": "UserServiceImpl.java"
                },
                "label": "OrderService.java"
        },
        "factory": {
            "label": "DAOFactory.java"
        }
        }
        :param path:
        :return:
        """
        root = {}
        self.get_dir_tree_dict(path, root)
        return json.dumps(root)

    def get_dir_tree_dict(self, path, root):
        """
        遍历文件树，将遍历节点以字典的形式保存到root
        :param path:
        :param root:
        :return:
        """
        path_list = os.listdir(path)
        for _, item in enumerate(path_list):
            if os.path.isdir(os.path.join(path, item)):
                path = os.path.join(path, item)
                root[item] = {}
                self.get_dir_tree_dict(path, root[item])
                path = '/'.join(path.split('/')[:-1])
            else:
                root['label'] = item
