from multiprocessing import Process, Queue, Pool

a = []
c = []

def run(i):
    b = 1
    return (b, i)

lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
pool = Pool(processes=4)
a,c  = zip(*pool.map(run, lst))

print(a)
print(c)
