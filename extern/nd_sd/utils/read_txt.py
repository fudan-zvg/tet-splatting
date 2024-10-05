import pdb
import time

import pandas as pd

midas_writer = './laion_art_depth/midas_tokens.txt'

start = time.time()
data = pd.read_csv(midas_writer)
print(time.time() - start)

pdb.set_trace()
