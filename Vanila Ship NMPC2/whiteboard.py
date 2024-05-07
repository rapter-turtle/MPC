import numpy as np
import math 

a = [1,2,3,4,5,6,7]
b = []

for i in range(9):
    b.append(a)

second_row = [sub_list[1] for sub_list in b]

print(second_row)
