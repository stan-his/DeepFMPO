# Simple implementaion of a binary tree.

class Tree(object):

    def __init__(self, left=None, right=None, value=None):
        self.left = left
        self.right = right
        self.value = value


    def __str__(self):
        return self.to_string(0)


    __repr__ = __str__


    def to_string(self, n):
        s = ""
        space = "".join(["    "] * n)
        if not self.value is None:
            s = str(self.value)
        else:
            if not self.left is None:
                s +=  "----"  + self.left.to_string(n+1)
            if not self.right is None:
                s += "\n" + space + "`---" + self.right.to_string(n+1)

        return s



    def get_depth(self):
        d1, d2 = 0, 0
        if not self.left is None:
            d1 = 1 + self.left.get_depth()
        if not self.right is None:
            d2 = 1 + self.right.get_depth()

        return max(d1, d2)



    def encode_leafs(self):
        return self.encode_r([])


    def encode_r(self, encoding):
        if not self.value is None:
            return [(self.value, "".join(encoding))]

        else:
            if not self.left is None:
                encoding.append("0")
                l1 = self.left.encode_r(encoding)
                encoding.pop()
            else:
                l1 = []
            if not self.right is None:
                encoding.append("1")
                l2 = self.right.encode_r(encoding)
                encoding.pop()
            else:
                l2 = []

            return l1 + l2


def build_tree_from_list(l, lookup=None):

    return btl(l, len(l)-1, 0, lookup)



def btl(l, n, use, lookup):

    if n >= 0:
        pair = l[n][use]
        t1 = btl(l, n-1, pair[0], lookup)
        t2 = None if len(pair) == 1 else btl(l, n-1, pair[1], lookup)
        return Tree(t1, t2)
    else:
        if not lookup is None:
            use = lookup[use]
        return Tree(value=use)


