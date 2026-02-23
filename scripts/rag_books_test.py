import json

from chatbots.backend.rag_books import search_books

result = search_books(
    query_text="A magical school adventure with friendship and mystery", top_k=3
)
print(json.dumps(result, indent=2, default=str))
