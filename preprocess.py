import json
import os

import numpy as np


MEM_EMBEDDINGS = np.load('mem2embsNorm.npy')

with open('mem2captureLem.txt', 'r') as f:
    MEM_CAPTURES = list(map(lambda x: set(x.strip().split()), f.readlines()))

files = os.listdir('statsfiles')

if 'user_embs.npy' in files:
    USER_EMBEDDINGS = list(np.load('statsfiles/user_embs.npy'))
else:
    USER_EMBEDDINGS = []

if 'usrid2num.json' in files:
    with open('statsfiles/usrid2num.json', 'r') as f:
        USER_ID2NUM = json.loads(f.read())
else:
    USER_ID2NUM = {}

NUM_MEMS = len(MEM_CAPTURES)
NUM_TO_SHOW = 5

if 'memids.txt' in files:
    with open('statsfiles/memids.txt', 'r') as f:
        MEM_IDS = eval(f.read())
        if len(MEM_IDS) < NUM_MEMS:
            MEM_IDS.extend([-1 for _ in range(NUM_MEMS - len(MEM_IDS))])
else:
    MEM_IDS = [-1 for _ in range(NUM_MEMS)]

CURRENT_USER_DATA = {}

if 'memlikes.txt' in files:
    with open('statsfiles/memlikes.txt', 'r') as f:
        MEM_LIKES = eval(f.read())
        if len(MEM_LIKES) < NUM_MEMS:
            MEM_LIKES.extend([0 for _ in range(NUM_MEMS - len(MEM_LIKES))])
else:
    MEM_LIKES = [0 for _ in range(NUM_MEMS)]

if 'memdislikes.txt' in files:
    with open('statsfiles/memdislikes.txt', 'r') as f:
        MEM_DISLIKES = eval(f.read())
        if len(MEM_DISLIKES) < NUM_MEMS:
            MEM_DISLIKES.extend([-1 for _ in range(NUM_MEMS - len(MEM_DISLIKES))])
else:
    MEM_DISLIKES = [0 for _ in range(NUM_MEMS)]

EMB_SIZE = 768

if 'userlang.txt' in files:
    with open('statsfiles/userlang.txt', 'r') as f:
        USER_LANG = eval(f.read())
else:
    USER_LANG = []
