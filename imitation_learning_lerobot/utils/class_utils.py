class ClassUtils:
    @staticmethod
    def get_leaf_subclasses(base_class) -> list:
        leaf_subclasses = []
        for subclass in base_class.__subclasses__():
            if subclass.__subclasses__():
                leaf_subclasses.extend(ClassUtils.get_leaf_subclasses(subclass))
            else:
                leaf_subclasses.append(subclass)
        return leaf_subclasses

    @staticmethod
    def get_all_subclasses(base_class) -> list:
        """獲取所有子類別（包括非葉節點），用於需要註冊所有類別的情況"""
        all_subclasses = []
        for subclass in base_class.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(ClassUtils.get_all_subclasses(subclass))
        return all_subclasses

