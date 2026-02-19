import asyncio
from dotenv import load_dotenv
from bookdb.models.query_rewritter import create_groq_client, rewrite_query

load_dotenv()

async def main():
    client = create_groq_client()
    description, review = await rewrite_query(client, "An enemies to lovers romance set in a fantasy medieval world")
    print("DESCRIPTION:\n", description)
    print("\nREVIEW:\n", review)

asyncio.run(main())
