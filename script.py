from tools.functions import *

a = benchmark_different_sizes([2,4,8,16,32])
print(a)
saveResultsToCSV(a)
a = loadResultsFromCSV(a)
print(a)
# TODO Make a gui 