import os
import hashlib

if __name__ == '__main__':

    hash = hashlib.md5(open('preprocessedData_normalized_l2.csv','rb').read()).hexdigest()
    print(hash)
