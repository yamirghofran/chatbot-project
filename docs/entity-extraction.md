# Entity Extraction Feature

## Overview

This feature adds intelligent entity recognition to the book recommendation chatbot. When users mention specific book titles or author names in their queries, the system now:

1. **Extracts entities** from natural language using Groq LLM
2. **Finds matching books/authors** in the database with fuzzy matching
3. **Generates context-aware recommendations** based on the mentioned entities

## Example Usage

### Before Entity Extraction

**User Query:** "I really like Game of Thrones and want some fantasy novels like it"

**Previous Behavior:**
- LLM would create a fictional book description
- Recommendations would be based on this fictional description
- Not optimized for actual Game of Thrones fans

### After Entity Extraction

**User Query:** "I really like Game of Thrones and want some fantasy novels like it"

**New Behavior:**
1. Entity extraction identifies: "Game of Thrones"
2. Database lookup finds the actual book
3. LLM receives context about:
   ```
   TITLE: A Game of Thrones
   AUTHOR: George R.R. Martin
   GENRE: Fantasy, Epic, Adventure
   MATCH_SCORE: 0.97
   DESCRIPTION: The first book in the epic fantasy series...
   ```
4. Recommendations are based on similar books to actual Game of Thrones
5. **Result:** Better, more relevant fantasy recommendations

## Architecture

```
User Query
    ↓
[Entity Extraction] (Groq LLM)
    ↓
Book Titles: ["Game of Thrones"]
Author Names: []
    ↓
[Database Fuzzy Lookup] (pg_trgm)
    ↓
Found: Book(id=1, title="A Game of Thrones", score=0.97)
    ↓
[Context Generation]
    ↓
Context: "TITLE: A Game of Thrones\nAUTHOR: George R.R...."
    ↓
[Query Rewriter] (Groq LLM with context)
    ↓
Description: "A dark fantasy epic featuring political intrigue..."
    ↓
[Vector Search]
    ↓
Recommendations: The Name of the Wind, Mistborn, etc.
```

## Key Components

### 1. Entity Extraction Module

**File:** `apps/api/core/entity_extraction.py`

**Functions:**
- `extract_book_entities()` - Extracts titles/authors from query using LLM
- `find_books_by_title()` - Fuzzy search for books (pg_trgm)
- `find_authors_by_name()` - Fuzzy search for authors (pg_trgm)
- `get_book_context_string()` - Formats book metadata for LLM context
- `get_author_context_string()` - Formats author metadata for LLM context
- `resolve_entities()` - High-level function that orchestrates the flow

### 2. LLM Integration

**File:** `bookdb/models/chatbot_llm.py`

**Changes:**
- Added `BOOK_DESCRIPTION_WITH_CONTEXT_PROMPT` for entity-aware rewriting
- Updated `_rewrite_description()` to accept `entity_context` parameter
- Updated `_rewrite_description_sync()` to accept `entity_context` parameter
- Updated `rewrite_query()` and `rewrite_query_sync()` to pass entity context

### 3. Search Pipeline Integration

**File:** `apps/api/routers/books.py`

**Changes:**
- Modified `_run_chatbot_search_pipeline()` to:
  1. Call `resolve_entities()` before query rewriting
  2. Pass entity context to query rewriter
  3. Handle errors gracefully and fall back to original behavior

### 4. Configuration

**File:** `apps/api/core/config.py`

**New Settings:**
```python
# Entity extraction settings
ENTITY_EXTRACTION_ENABLED: bool = True  # Enable/disable feature
ENTITY_EXTRACTION_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"
ENTITY_SIMILARITY_THRESHOLD: float = 0.3  # Minimum score for fuzzy match
ENTITY_CONFIDENCE_THRESHOLD: float = 0.7  # Minimum LLM confidence
ENTITY_MAX_BOOKS_PER_QUERY: int = 2  # Max books to resolve
ENTITY_MAX_AUTHORS_PER_QUERY: int = 2  # Max authors to resolve
ENTITY_CACHE_TTL: int = 3600  # Cache duration (seconds)
```

### 5. Tests

**File:** `tests/test_api/test_entity_extraction.py`

**Coverage:**
- Unit tests for string similarity
- Unit tests for fuzzy lookup (books and authors)
- Unit tests for context generation
- Integration tests for entity extraction (requires LLM)
- Performance tests
- Error handling tests
- Edge cases and concurrency tests

## Fuzzy Matching

The system uses PostgreSQL's `pg_trgm` extension for sophisticated fuzzy matching:

### pg_trgm Similarity (Preferred)

```sql
SELECT title, similarity(lower(title), lower(:search_term)) as score
FROM books
WHERE similarity(lower(title), lower(:search_term)) >= :threshold
ORDER BY score DESC
```

**Advantages:**
- ✅ Database-indexed for performance
- ✅ Handles typos, partial matches, and reordering
- ✅ Sophisticated trigram-based algorithm

### Python Fallback

If `pg_trgm` is unavailable, the system falls back to Python's `difflib.SequenceMatcher`:

```python
score = difflib.SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
```

**Advantages:**
- ✅ No database dependencies
- ✅ Works with any database

**Disadvantages:**
- ⚠️ Less sophisticated than pg_trgm
- ⚠️ Fetches more candidates from database
- ⚠️ Slower for large datasets

## Caching

Entity lookups are cached to improve performance and reduce database load:

```python
_entity_lookup_cache: TTLCache = TTLCache(maxsize=1000, ttl=3600)
```

**Cache Key Format:**
- Books: `"book:{title.lower()}:{limit}:{similarity_threshold}"`
- Authors: `"author:{name.lower()}:{limit}:{similarity_threshold}"`

**Cache Management:**
- Automatic expiration after 1 hour (TTL)
- Maximum 1000 entries
- Manual cache clearing available via `clear_entity_cache()`

## Examples

### Example 1: Single Book

**Query:** "I love Harry Potter"

**Extraction:**
```python
{
    "book_titles": ["Harry Potter"],
    "author_names": [],
    "confidence": 0.95
}
```

**Match:**
```python
(
    Book(id=2, title="Harry Potter and the Sorcerer's Stone", ...),
    score=0.98
)
```

**Context:**
```
TITLE: Harry Potter and the Sorcerer's Stone
AUTHOR: J.K. Rowling
GENRE: Fantasy, Adventure, Young Adult
MATCH_SCORE: 0.98
DESCRIPTION: A young wizard's journey begins...
```

### Example 2: Multiple Books

**Query:** "I like Game of Thrones and Lord of the Rings"

**Extraction:**
```python
{
    "book_titles": ["Game of Thrones", "Lord of the Rings"],
    "author_names": [],
    "confidence": 0.92
}
```

**Matches:**
```python
[
    (Book(title="A Game of Thrones", ...), 0.97),
    (Book(title="The Lord of the Rings", ...), 0.95)
]
```

### Example 3: Author

**Query:** "I'm a big fan of Stephen King's novels"

**Extraction:**
```python
{
    "book_titles": [],
    "author_names": ["Stephen King"],
    "confidence": 0.88
}
```

**Match:**
```python
(Author(name="Stephen King", ...), 0.95)
```

**Context (includes author's books):**
```
AUTHOR: Stephen King (MATCH_SCORE: 0.95)
BOOKS BY THIS AUTHOR:
  - The Shining: A terrifying tale of isolation and madness...
  - It: A story about a killer clown...
  - The Stand: Post-apocalyptic survival...
```

### Example 4: No Entities

**Query:** "I want something romantic"

**Extraction:**
```python
{
    "book_titles": [],
    "author_names": [],
    "confidence": 0.2
}
```

**Behavior:** Falls back to original query rewriting (no context)

## Performance

### Expected Latency

- **Entity Extraction (LLM):** 0.5-1.5 seconds
- **Database Lookup (cached):** < 0.01 seconds
- **Database Lookup (uncached):** 0.1-0.3 seconds
- **Total Overhead:** ~1-2 seconds for entity-rich queries

### Optimization Tips

1. **Use caching:** Enable `ENTITY_CACHE_TTL` for frequently searched books
2. **Adjust thresholds:** Increase `ENTITY_SIMILARITY_THRESHOLD` to reduce false positives
3. **Limit entities:** Reduce `ENTITY_MAX_BOOKS_PER_QUERY` for faster lookups
4. **Disable for low-confidence:** Set `ENTITY_CONFIDENCE_THRESHOLD` to skip unnecessary extractions

## Configuration Examples

### Conservative (Fewer matches, higher precision)

```python
ENTITY_SIMILARITY_THRESHOLD = 0.5  # Only strong matches
ENTITY_CONFIDENCE_THRESHOLD = 0.8  # High LLM confidence
ENTITY_MAX_BOOKS_PER_QUERY = 1  # Only 1 book
```

### Aggressive (More matches, higher recall)

```python
ENTITY_SIMILARITY_THRESHOLD = 0.3  # Weaker matches
ENTITY_CONFIDENCE_THRESHOLD = 0.6  # Lower LLM confidence
ENTITY_MAX_BOOKS_PER_QUERY = 3  # Up to 3 books
```

### Disabled (Fallback to original behavior)

```python
ENTITY_EXTRACTION_ENABLED = False
```

## Testing

### Run All Tests

```bash
# Basic tests (no LLM required)
pytest tests/test_api/test_entity_extraction.py -v

# Include LLM tests (requires GROQ_API_KEY)
pytest tests/test_api/test_entity_extraction.py --run-llm-tests -v

# Run specific test category
pytest tests/test_api/test_entity_extraction.py::test_find_books_by_title_exact_match -v
```

### Test Coverage

- ✅ String similarity (5 tests)
- ✅ Book fuzzy lookup (8 tests)
- ✅ Author fuzzy lookup (4 tests)
- ✅ Context generation (4 tests)
- ✅ Entity extraction (5 tests, requires LLM)
- ✅ Entity resolution (3 tests, requires LLM)
- ✅ Cache management (2 tests)
- ✅ Performance (2 tests)
- ✅ Error handling (2 tests)
- ✅ Edge cases (4 tests)
- ✅ Concurrency (1 test)
- ✅ End-to-end (1 test, requires LLM)

**Total:** 41 tests

## Troubleshooting

### Issue: Entity extraction not working

**Check:**
1. `ENTITY_EXTRACTION_ENABLED` is `True` in configuration
2. `GROQ_API_KEY` is set in environment
3. Logs show "Entity extraction successful" messages

### Issue: Not finding books

**Check:**
1. Books exist in database
2. `ENTITY_SIMILARITY_THRESHOLD` is not too high
3. pg_trgm extension is installed in PostgreSQL

### Issue: Poor recommendations

**Check:**
1. Entity confidence is high enough (> 0.7)
2. Fuzzy match score is high enough (> 0.5)
3. Context is being passed to query rewriter (check logs)
4. Vector embeddings are working correctly

### Issue: Slow performance

**Solutions:**
1. Enable caching (default: `ENTITY_CACHE_TTL=3600`)
2. Reduce `ENTITY_MAX_BOOKS_PER_QUERY` and `ENTITY_MAX_AUTHORS_PER_QUERY`
3. Increase `ENTITY_SIMILARITY_THRESHOLD` to reduce false positives
4. Check database indexes (pg_trgm GIN indexes should exist)

## Future Enhancements

### Potential Improvements

1. **Series Recognition:** Detect when user mentions "Harry Potter books" (series)
2. **Genre Extraction:** Extract genres even without specific book titles
3. **Author Similarity:** Find similar authors when one is mentioned
4. **User History Integration:** Use user's past reads for entity context
5. **Feedback Loop:** Improve entity extraction based on user corrections

### Advanced Features

1. **Entity Linking:** Link book mentions to GoodReads/Wikipedia
2. **Knowledge Graph:** Use entity relationships for better recommendations
3. **Multi-turn Context:** Remember entities across conversation
4. **Entity Confidence Scoring:** More sophisticated confidence calculation

## Migration Notes

### Database Requirements

The feature requires the `pg_trgm` extension, which is already installed via migration:
`b7d4f91a2c10_add_performance_indexes.py`

**Verification:**
```sql
SELECT * FROM pg_extension WHERE extname = 'pg_trgm';
```

If not installed, run:
```sql
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

### No Code Breaking Changes

This feature is **fully backward compatible**:
- ✅ Works without any code changes to existing routes
- ✅ Graceful fallback if extraction fails
- ✅ Can be disabled via configuration
- ✅ No database schema changes required

## References

- [PostgreSQL pg_trgm Documentation](https://www.postgresql.org/docs/current/pgtrgm.html)
- [Groq API Documentation](https://console.groq.com/docs)
- [Entity Extraction Best Practices](https://nlp.stanford.edu/IR-book/html/58ner.html)

## Contributors

Feature implementation based on issues/requirements for improving book recommendation quality by recognizing user-mentioned books and authors.

## License

Same as parent project (see root LICENSE file).
