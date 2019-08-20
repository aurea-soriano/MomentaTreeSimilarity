class Queue:

    def __init__(self):
        self.queue = list()
        
    def enqueue(self,data):
        if data not in self.queue:
            self.queue.insert(0,data)
            return True
        return False
    
    def dequeue(self):
        if len(self.queue)>0:
            return self.queue.pop()
    
    def empty(self):
        if len(self.queue)>0:
            return False
        return True
    
    def size(self):
        return len(self.queue)
    
    def printQueue(self):
        return self.queue
    
    def contains(self, object):
        if self.queue.__contains__(object):
            return True
        return False
    

if __name__ == "__main__":
    myQueue = Queue()
    print(myQueue.enqueue(5)) #prints True
    print(myQueue.enqueue(6)) #prints True
    print(myQueue.enqueue(9)) #prints True
    print(myQueue.enqueue(5)) #prints False
    print(myQueue.enqueue(3)) #prints True
    print(myQueue.contains(8)) #prints True
    print(myQueue.size())     #prints 4
    print(myQueue.dequeue())  #prints 5
    print(myQueue.dequeue())  #prints 6
    print(myQueue.dequeue())  #prints 9
    print(myQueue.dequeue())  #prints 3
    print(myQueue.size())     #prints 0
    print(myQueue.dequeue())  #prints Queue Empty!