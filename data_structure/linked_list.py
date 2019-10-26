
class LinkedListNode():
    def __init__(self, val, next=None):
        self.val = val
        self._next = next
    
    @property
    def next(self):
        return self._next

    @next.setter
    def next(self, node):
        self._next = node

    def __repr__(self):
        return f"{self.val} -> {self.next}" if self.next else f"{self.val}"


class LinedList():
    def __init__(self, list_):
        self.head = LinkedListNode(list_[0])
        current_node = self.head
        for data in list_[1:]:
            current_node.next = LinkedListNode(data)
            current_node = current_node.next
        
        self.current_node = self.head

    def __iter__(self):
        self.current_node = self.head
        return self

    def __next__(self):
        curr = self.current_node
        if not curr:
            raise StopIteration
        self.current_node = self.current_node.next
        return curr

    def find(self, data):
        for node in self:
            if data == node.val:
                return node
        return None

    def __repr__(self):
        return self.head.__repr__()


if __name__ == "__main__":

    #  s0 = LinkedListNode(1)
    #  s0.next = s1 = LinkedListNode(2)
    #  s1.next = s2 = LinkedListNode(3)
    #  s2.next = LinkedListNode(4)
    ll = LinedList([1,2,3,4])
    print(ll)
