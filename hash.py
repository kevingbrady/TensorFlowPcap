import os
import hashlib
import sys
import cudf 

if __name__ == '__main__':

    #hash = hashlib.md5(open('preprocessedData_normalized_l2.csv','rb').read()).hexdigest()
    #print(hash)
    print('cudf' in sys.modules)
