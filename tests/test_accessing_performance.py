import time
from random import random

dictionary = {}
list_object = []

for i in range(int(1e7)):
    x = random()
    dictionary[x] = x
    list_object.append(x)

start = time.time()
for x in list_object:
    dictionary[x]

print(f"accessing dict: {time.time()-start}s")

start = time.time()
for i in range(len(list_object)):
    list_object[i]

print(f"accessing list: {time.time()-start}s")
