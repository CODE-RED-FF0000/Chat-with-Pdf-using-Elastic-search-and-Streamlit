indexMapping = {
    "properties":{
        "chunks":{
            "type":"text"
        },
        "chunksvector":{
            "type":"dense_vector",
            "dims": 768,
            "index":True,
            "similarity": "l2_norm"
        }

    }
}