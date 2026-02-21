"""Update all users with realistic names and a consistent password hash.

This script updates ALL users in the database with random realistic names,
matching usernames, and the same password hash (default: "1234").
"""

from __future__ import annotations

import argparse
import random

import bcrypt
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from bookdb.db.session import SessionLocal


def normalize_database_url(url: str) -> str:
    """Normalize database URL to use psycopg driver."""
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql+psycopg2://"):
        return "postgresql+psycopg://" + url[len("postgresql+psycopg2://") :]
    return url


def get_session(database_url: str | None) -> Session:
    """Get a database session, using custom URL or default."""
    if database_url:
        normalized_url = normalize_database_url(database_url)
        engine = create_engine(normalized_url, future=True)
        SessionFactory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
        return SessionFactory()
    return SessionLocal()


# Sample realistic first names
FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
    "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
    "Kenneth", "Dorothy", "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa",
    "Edward", "Deborah", "Ronald", "Stephanie", "Timothy", "Rebecca", "Jason", "Sharon",
    "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob", "Kathleen", "Gary", "Amy",
    "Nicholas", "Angela", "Eric", "Shirley", "Jonathan", "Anna", "Stephen", "Brenda",
    "Larry", "Pamela", "Justin", "Emma", "Scott", "Nicole", "Brandon", "Helen",
    "Benjamin", "Samantha", "Samuel", "Katherine", "Raymond", "Christine", "Gregory", "Debra",
    "Frank", "Rachel", "Alexander", "Carolyn", "Patrick", "Janet", "Jack", "Catherine",
    "Dennis", "Maria", "Jerry", "Heather", "Tyler", "Diane", "Aaron", "Ruth",
    "Jose", "Julie", "Adam", "Olivia", "Henry", "Joyce", "Nathan", "Virginia",
    "Douglas", "Judith", "Zachary", "Megan", "Peter", "Andrea", "Kyle", "Cheryl",
    "Ethan", "Hannah", "Walter", "Jacqueline", "Noah", "Martha", "Jeremy", "Gloria",
    "Christian", "Teresa", "Keith", "Ann", "Roger", "Tiffany", "Terry", "Madison",
    "Austin", "Julia", "Sean", "Grace", "Gerald", "Jane", "Carl", "Kelly",
    "Dylan", "Sofia", "Jesse", "Alice", "Lawrence", "Kylie", "Jordan", "Christina",
    "Bryan", "Katherine", "Billy", "Isabella", "Joe", "Mia", "Bruce", "Charlotte",
    "Gabriel", "Audrey", "Logan", "Emily", "Albert", "Sophia", "Willie", "Avery",
    "Alan", "Lily", "Wayne", "Chloe", "Ralph", "Victoria", "Roy", "Penelope",
    "Eugene", "Harper", "Russell", "Aria", "Vincent", "Scarlett", "Philip", "Claire",
    "Bobby", "Eleanor", "Johnny", "Zoey", "Logan", "Stella", "Frank", "Ellie",
]

# Sample realistic last names
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker",
    "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris", "Morales", "Murphy",
    "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper", "Peterson", "Bailey",
    "Reed", "Kelly", "Howard", "Ramos", "Kim", "Cox", "Ward", "Richardson", "Watson",
    "Brooks", "Chavez", "Wood", "James", "Bennett", "Gray", "Mendoza", "Ruiz",
    "Hughes", "Li", "Alvarez", "Holt", "Sullivan", "Castillo", "Gordon", "Shaw",
    "Weaver", "Powell", "Ferguson", "Wagner", "Ryan", "Johnston", "Simpson", "Holmes",
    "Jacobs", "Grant", "Wallace", "Foster", "Murray", "Cole", "Payne", "Berry",
    "Butler", "Pearson", "Hoffman", "Hudson", "Harrison", "Hamilton", "Martin", "Gibson",
    "Olson", "Lynch", "Sandoval", "Rose", "Kennedy", "Weaver", "Fisher", "Richards",
    "Mason", "Stephens", "Ray", "Berry", "Roy", "Watkins", "Dixon", "Bishop",
    "Snyder", "Romero", "Sims", "Higgins", "Morrison", "Shaw", "Hopkins", "Perkins",
    "Cunningham", "Banks", "Perry", "Roberts", "Hansen", "Graham", "Lawson", "Chandler",
    "Payne", "Patrick", "Lindsey", "Boyd", "Griffin", "Dennis", "Oliver", "Mcdonald",
    "Reynolds", "Harrison", "Jenkins", "Perry", "Long", "Patterson", "Hughes", "Flores",
    "Washington", "Butler", "Simmons", "Foster", "Gonzales", "Bryant", "Alexander", "Russell",
    "Griffin", "Diaz", "Hayes", "Myers", "Ford", "Hamilton", "Graham", "Sullivan",
    "Wallace", "Woods", "Cole", "West", "Jordan", "Owens", "Reynolds", "Fisher",
    "Ellis", "Harrison", "Rogers", "Stevens", "Singh", "Patel", "Young", "Allen",
]


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def generate_names_for_users(user_ids: list[int]) -> list[tuple[str, str]]:
    """Generate (name, username) pairs using user_id to guarantee unique usernames."""
    results = []

    for user_id in user_ids:
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        full_name = f"{first_name} {last_name}"

        # Use user_id in username to guarantee uniqueness
        base = f"{first_name.lower()}_{last_name.lower()}"
        base = "".join(c for c in base if c.isalnum() or c == "_")
        username = f"{base}_{user_id}"

        results.append((full_name, username))

    return results


def update_users(
    session: Session,
    password_hash: str,
    batch_size: int = 5000,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, int]:
    """Update all users with realistic names and the given password hash."""
    stats = {
        "total_users": 0,
        "updated": 0,
    }

    # Count total users
    stats["total_users"] = session.execute(text("SELECT COUNT(*) FROM users")).scalar() or 0
    total_to_process = min(stats["total_users"], limit) if limit else stats["total_users"]

    print(f"Found {stats['total_users']} total users")
    if limit:
        print(f"Processing {total_to_process} users (limit applied)")
    if dry_run:
        print("DRY RUN - No changes will be made")

    offset = 0
    processed = 0

    while True:
        if limit and processed >= limit:
            break

        current_batch_size = min(batch_size, limit - processed) if limit else batch_size

        # Fetch batch of user IDs
        batch_result = session.execute(
            text("SELECT id FROM users ORDER BY id LIMIT :limit OFFSET :offset"),
            {"limit": current_batch_size, "offset": offset},
        ).fetchall()

        if not batch_result:
            break

        user_ids = [row[0] for row in batch_result]
        names_batch = generate_names_for_users(user_ids)

        if dry_run:
            for i, (user_id, (name, username)) in enumerate(zip(user_ids, names_batch)):
                print(f"  [{processed + i + 1}] User {user_id}: '{name}' ({username})")
            stats["updated"] += len(user_ids)
        else:
            update_data = [
                {
                    "user_id": uid,
                    "name": name,
                    "username": username,
                    "password_hash": password_hash,
                }
                for uid, (name, username) in zip(user_ids, names_batch)
            ]

            session.execute(
                text(
                    "UPDATE users SET name = :name, username = :username, password_hash = :password_hash "
                    "WHERE id = :user_id"
                ),
                update_data,
            )
            session.commit()
            stats["updated"] += len(user_ids)

        processed += len(user_ids)
        offset += len(user_ids)

        if not dry_run and processed % 10000 == 0:
            print(f"  Updated {processed}/{total_to_process} users...")

        if len(user_ids) < current_batch_size:
            break

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update all users with realistic names and a consistent password."
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Database connection URL (e.g., postgresql://user:pass@host:port/db).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Number of users to update per batch (default: 5000)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of users to update (default: all)",
    )
    parser.add_argument(
        "--password",
        type=str,
        default="1234",
        help="Password to set for all users (default: 1234)",
    )
    args = parser.parse_args()

    print(f"Hashing password '{args.password}'...")
    password_hash = hash_password(args.password)
    print(f"Password hash: {password_hash[:50]}...")

    if args.database_url:
        print(f"Connecting to: {args.database_url.split('@')[-1] if '@' in args.database_url else 'provided URL'}...")
    else:
        print("Connecting to database using default configuration...")

    session = get_session(args.database_url)
    try:
        stats = update_users(
            session,
            password_hash=password_hash,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            limit=args.limit,
        )

        print("\nSummary:")
        print(f"  Total users: {stats['total_users']}")
        print(f"  Updated: {stats['updated']}")

        if args.dry_run:
            print("\nThis was a dry run. Run without --dry-run to apply changes.")

    finally:
        session.close()


if __name__ == "__main__":
    main()
