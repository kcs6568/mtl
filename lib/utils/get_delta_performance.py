def get_mtl_performance(single, multi):
    assert len(single) == len(multi)
    T = len(single)  
    
    total = 0.
    for i in range(T):
        total += (multi[i] - single[i]) / single[i]

    delta_perf = total / T
    
    return delta_perf

s = [94.86, 83.61, 28.5, 87.45]
a = [94.9, 91.84, 13.16, 86.88]
b = [95.05, 91.46, 12.41, 87.91]

d1 = get_mtl_performance(s, a)
d2 = get_mtl_performance(s, b)

print(d1)
print(d2)