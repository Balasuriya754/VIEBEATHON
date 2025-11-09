from OfflineRAG_Pro.rag_core.embedder import embed_texts
import numpy as np

q = "Explain smart parking system"
vec = embed_texts([q])

print("Type:", type(vec))
print("Shape/Structure:")
if isinstance(vec, np.ndarray):
    print(vec.shape)
else:
    import pprint; pprint.pprint(vec)
