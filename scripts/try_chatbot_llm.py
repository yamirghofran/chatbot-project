import asyncio
from dotenv import load_dotenv
from bookdb.models.chatbot_llm import create_groq_client, rewrite_query, generate_response

load_dotenv()

QUERY = "An enemies to lovers romance set in a fantasy medieval world"

BOOKS = [
    {
        "book_id": "1",
        "description": (
            "TITLE: A Throne of Ash and Starlight\n"
            "AUTHOR: Seraphine Vale\n"
            "SHELVES: fantasy, romance, enemies-to-lovers, medieval, magic\n"
            "DESCRIPTION: When Lady Mira is captured by the ruthless Fae prince Caelen, "
            "she expects death. Instead she finds herself bound to him by an ancient curse "
            "that neither can break alone. As they navigate treacherous courts and older "
            "magic than either imagined, the line between hatred and desire blurs beyond recognition.\n"
        ),
    },
]

REVIEWS = [
    {
        "review_id": "101",
        "book_title": "A Throne of Ash and Starlight",
        "review": (
            "I devoured this in one sitting. Mira and Caelen have the most infuriating, "
            "electric dynamic — every scene crackles. The slow build is worth every page. "
            "My only gripe is the middle act drags slightly, but the payoff more than "
            "makes up for it. Absolutely staying on my all-time favourites shelf."
        ),
    },
    {
        "review_id": "102",
        "book_title": "The Iron Vow",
        "review": (
            "The fake-dating trope done exactly right. Soren and Elara bicker like they "
            "hate each other and somehow that made me root for them even harder. The "
            "political plot is surprisingly tight for a romance-forward book. Wished the "
            "ending had a bit more breathing room, but I'll be re-reading this one."
        ),
    },
]


async def main():
    client = create_groq_client()

    description, review = await rewrite_query(client, QUERY)
    print("REWRITTEN DESCRIPTION:\n", description)
    print("\nREWRITTEN REVIEW:\n", review)

    result = await generate_response(client, QUERY, BOOKS, REVIEWS)
    print("CHATBOT RESPONSE:\n", result["response"])
    print("\nReferenced book IDs:", result["referenced_book_ids"])
    print("Referenced review IDs:", result["referenced_review_ids"])

asyncio.run(main())
