REPA0_RDB_VOC={
    'BACKBONE':{
        'NAME': 'A0',
        'PRETRAINED': True,
    },
    'NECK': {
        'NAME': 'RDB',
        'C3_CHANNEL': 96,
        'C4_CHANNEL': 192,
        'OUT_CHANNEL': 128,
        'DEPTH':3,
        'GROUPS':4,
        'USE_AT': False,
        },
    'HEAD':{
        'OUT_CHANNEL': 128,
        'DEPTH':1,
    }
}
REPB0_RDB_VOC={
    'BACKBONE':{
        'NAME': 'B0',
        'PRETRAINED': True,
    },
    'NECK': {
        'NAME': 'RDB',
        'C3_CHANNEL': 128,
        'C4_CHANNEL': 256,
        'OUT_CHANNEL': 128,
        'DEPTH':3,
        'GROUPS':4,
        'USE_AT': False,
        },
    'HEAD':{
        'OUT_CHANNEL': 128,
        'DEPTH':1,
    }
}


REPA0_RDB_COCO={
    'BACKBONE':{
        'NAME': 'A0',
        'PRETRAINED': True,
    },
    'NECK': {
        'NAME': 'RDB',
        'C3_CHANNEL': 96,
        'C4_CHANNEL': 192,
        'OUT_CHANNEL': 192,
        'DEPTH':3,
        'GROUPS':4,
        'USE_AT': True,
        },
    'HEAD':{
        'OUT_CHANNEL': 128,
        'DEPTH':4,
    }
}
