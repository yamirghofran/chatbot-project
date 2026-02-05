Basic EDA based on [Basic EDA Notebook](../notebooks/data/exploring_datasets.py)

### raw_book_id_map
- **Rows:** 2,360,650
- **Columns:** 2
- **Features:** book_id_csv, book_id
- **Data Types:**
  - book_id_csv: Int64
  - book_id: Int64

### raw_goodreads_book_authors
- **Rows:** 829,529
- **Columns:** 5
- **Features:** average_rating, author_id, text_reviews_count, name, ratings_count
- **Data Types:**
  - average_rating: String
  - author_id: String
  - text_reviews_count: String
  - name: String
  - ratings_count: String

### raw_goodreads_book_works
- **Rows:** 1,521,962
- **Columns:** 16
- **Features:** books_count, reviews_count, original_publication_month, default_description_language_code, text_reviews_count, best_book_id, original_publication_year, original_title, rating_dist, default_chaptering_book_id
- **Data Types:**
  - books_count: String
  - reviews_count: String
  - original_publication_month: String
  - default_description_language_code: String
  - text_reviews_count: String
  - best_book_id: String
  - original_publication_year: String
  - original_title: String

### raw_goodreads_interactions
- **Rows:** 228,648,342
- **Columns:** 5
- **Features:** user_id, book_id, is_read, rating, is_reviewed
- **Data Types:**
  - user_id: Int64
  - book_id: Int64
  - is_read: Int64
  - rating: Int64
  - is_reviewed: Int64

### raw_goodreads_interactions_dedup
- **Rows:** 228,648,342
- **Columns:** 10
- **Features:** user_id, book_id, review_id, is_read, rating, review_text_incomplete, date_added, date_updated, read_at, started_at
- **Data Types:**
  - user_id: String
  - book_id: String
  - review_id: String
  - is_read: Boolean
  - rating: Int64
  - review_text_incomplete: String
  - date_added: String
  - date_updated: String

### raw_goodreads_reviews_dedup
- **Rows:** 15,739,967
- **Columns:** 11
- **Features:** user_id, book_id, review_id, rating, review_text, date_added, date_updated, read_at, started_at, n_votes
- **Data Types:**
  - user_id: String
  - book_id: String
  - review_id: String
  - rating: Int64
  - review_text: String
  - date_added: String
  - date_updated: String
  - read_at: String

### raw_goodreads_reviews_spoiler
- **Rows:** 1,378,033
- **Columns:** 11
- **Features:** user_id, book_id, review_id, rating, review_text, date_added, date_updated, read_at, started_at, n_votes
- **Data Types:**
  - user_id: String
  - book_id: String
  - review_id: String
  - rating: Int64
  - review_text: String
  - date_added: String
  - date_updated: String
  - read_at: String

### raw_user_id_map
- **Rows:** 876,145
- **Columns:** 2
- **Features:** user_id_csv, user_id
- **Data Types:**
  - user_id_csv: Int64
  - user_id: String
