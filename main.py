from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
import numpy as np
import time

load_dotenv()

# text_1 = "Jestem wielkim fanem opakowań tekturowych"
# text_2 = "Bardzo podobają mi się kartony"

text_1 = "szklana pułapka"
text_2 = "die hard"


def use_karton():
    model = SentenceTransformer("OrlikB/KartonBERT-USE-base-v1")

    embeddings_1 = model.encode(text_1, normalize_embeddings=True)
    embeddings_2 = model.encode(text_2, normalize_embeddings=True)

    similarity = embeddings_1 @ embeddings_2.T
    print(f"Karton similarity {similarity}")


def use_jina():
    model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

    task = "text-matching"
    text_1 = "Jestem wielkim fanem opakowań tekturowych"
    text_2 = "Bardzo podobają mi się kartony"
    embeddings_1 = model.encode(
        [text_1],
        task=task,
        prompt_name=task,
    )
    embeddings_2 = model.encode(
        [text_2],
        task=task,
        prompt_name=task,
    )

    similarity = embeddings_1 @ embeddings_2.T
    print(f"Jina similarity {similarity}")


def use_openai():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "API key not found. Please set the OPENAI_API_KEY environment variable in your .env file."
        )

    client = OpenAI(api_key=openai_api_key)
    res = client.embeddings.create(
        input=[text_1, text_2], model="text-embedding-3-large"
    )
    embeddings_1 = res.data[0].embedding
    embeddings_2 = res.data[1].embedding

    similarity = np.array(embeddings_1) @ np.array(embeddings_2)
    print(f"OpenAI similarity {similarity}")


start_time = time.time()
use_karton()
end_time = time.time()

print(f"Karton execution time: {end_time - start_time} seconds")

start_time = time.time()
use_jina()
end_time = time.time()

print(f"Jina execution time: {end_time - start_time} seconds")

start_time = time.time()
use_openai()
end_time = time.time()

print(f"OpenAI execution time: {end_time - start_time} seconds")
