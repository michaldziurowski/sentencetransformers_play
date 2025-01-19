from sentence_transformers import SentenceTransformer

#model = SentenceTransformer('OrlikB/KartonBERT-USE-base-v1')

# text_1 = 'Jestem wielkim fanem opakowań tekturowych'
# text_2 = 'Bardzo podobają mi się kartony'
#
# embeddings_1 = model.encode(text_1, normalize_embeddings=True)
# embeddings_2 = model.encode(text_2, normalize_embeddings=True)
#
# similarity = embeddings_1 @ embeddings_2.T
# print(similarity)

model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

task = "text-matching"
text_1 = 'Jestem wielkim fanem opakowań tekturowych'
text_2 = 'Bardzo podobają mi się kartony'
embeddings_1 = model.encode(
    [text_1],
    task=task,
    prompt_name=task,
)
embeddings_2= model.encode(
    [text_2],
    task=task,
    prompt_name=task,
)

similarity = embeddings_1 @ embeddings_2.T
print(similarity)
