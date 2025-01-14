from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
query = input( "Enter your query: ")
# Sample meme data which has text/caption, image link, and category
meme_data = [
    {"text": "When you realize it's Monday tomorrow", "image": "https://community.atlassian.com/t5/image/serverpage/image-id/263783i28EFE632226231DD/image-size/large?v=v2&px=999", "category": "weekend"},
    {"text": "Trying to stay productive at work", "image": "meme_2.jpg", "category": "work"},
    {"text": "When you finally finish a project", "image": "meme_3.jpg", "category": "achievement"},
    {"text": "When people say they will keep their new year resolution", "image":"https://blog.media.io/images/images2022/funny-text-memes-1.jpg","category":"new year"},
    {"text":"When corporate tells you to find a difference", "image":"https://cdn-useast1.kapwing.com/static/templates/theyre-the-same-picture-meme-template-full-f9cf8470.webp","category":"difference"}
]

# Loading BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encoding text using BERT
def encode_text(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1)  # Use the mean of token embeddings for a sentence vector

# Encoding meme descriptions
meme_descriptions = [meme["text"] for meme in meme_data]
meme_embeddings = [encode_text(desc) for desc in meme_descriptions]

# Function that will find the most similar meme
def find_most_similar_meme(query):
    query_embedding = encode_text(query)
    # Calculating cosine similarity
    similarities = [cosine_similarity(query_embedding.numpy(), meme_embedding.numpy()).item() for meme_embedding in meme_embeddings]
    # Returning the meme with high similarity
    return meme_data[similarities.index(max(similarities))]

most_similar_meme = find_most_similar_meme(query)
print(f"Most relevant meme: {most_similar_meme['image']}")
