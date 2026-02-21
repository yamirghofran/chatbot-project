## Raw Datasets

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

## Data Processing Pipeline
Inside notebooks/data/processing you will find:

### Books Pipeline:
1. **books/1_clean_books.py:** Cleans the raw books dataset. Handles missing values in publication fields, casts columns to correct types, and prepares the books data for further processing.  
    **Input:** raw_goodreads_books.parquet  
    **Output:** 1_goodreads_books_cleaned.parquet

2. **books/2_standardize_book_ids.py:** Standardizes book IDs to integer CSV IDs using a mapping file. Drops rows with unmapped book IDs.  
    **Input:** 1_goodreads_books_cleaned.parquet, raw_book_id_map.parquet  
    **Output:** 2_goodreads_books_standardized.parquet

3. **books/3_aggregate_book_metrics.py:** Aggregates interaction metrics per book (num_interactions, num_read, num_ratings, num_reviews, avg_rating) using DuckDB and left-joins them onto the books dataset. Books with no interactions receive 0 for count columns.
    **Input:** 2_goodreads_books_standardized.parquet, 3_goodreads_interactions_reduced.parquet
    **Output:** 3_goodreads_books_with_metrics.parquet

---

### Interactions Pipeline:
1. **interactions/1_merge_interactions_book_editions.py:** Replaces book IDs in the interactions with canonical best book IDs, translating between Goodreads and CSV ID spaces using mapping files.  
    **Input:** raw_book_id_map.parquet, raw_user_id_map.parquet, best_book_id_map.json  
    **Output:** 1_goodreads_interactions_merged.parquet

2. **interactions/2_add_timestamps_to_merged.py:** Adds timestamps to the merged interactions, processing in chunks for memory efficiency.  
    **Input:** 1_goodreads_interactions_merged.parquet, 1_goodreads_interactions_dedup_merged.parquet  
    **Output:** 2_goodreads_interactions_merged_timestamps.parquet

3. **interactions/3_check_data_consistency.py:** Verifies that all book IDs in the interactions have a corresponding entry in 3_goodreads_books_with_metrics.parquet and drops unmatched rows.  
    **Input:** 2_goodreads_interactions_merged_timestamps.parquet, 3_goodreads_books_with_metrics.parquet  
    **Output:** Consistency-checked interactions (overwrites or filters previous file)

4. **interactions/4_interactions_cleaning.py:** Performs quality checks on the reduced interactions, such as timestamp range and logical consistency between fields, clips future timestamp values to current time.
    **Input:** 3_goodreads_interactions_reduced.parquet  
    **Output:** Cleaned and validated interactions (same file or new output)

---

### Reviews Pipeline:
1. **reviews/1_standardize_reviews.py:** Standardizes review files by mapping Goodreads book IDs and user IDs to CSV integer IDs, and replaces book IDs with canonical best book IDs.  
    **Input:** raw_book_id_map.parquet, raw_user_id_map.parquet, best_book_id_map.json  
    **Output:** 1_goodreads_reviews_dedup_merged.parquet

2. **reviews/2_check_data_consistency.py:** Verifies that all book IDs in the reviews dataset have a corresponding entry in 3_goodreads_books_with_metrics.parquet and drops unmatched rows.  
    **Input:** 1_goodreads_reviews_dedup_merged.parquet, 3_goodreads_books_with_metrics.parquet  
    **Output:** Consistency-checked reviews (overwrites or filters previous file)

3. **reviews/3_clean_dedup_reviews.py:** Cleans the reduced reviews data: drops unnecessary columns, parses dates to unix timestamps, and saves the cleaned result.  
    **Input:** 2_goodreads_reviews_dedup_reduced.parquet  
    **Output:** 3_goodreads_reviews_dedup_clean.parquet

4. **reviews/4_reduce_reviews.py:** Reduces reviews to at most k top-quality reviews per book, targeting around 2M total reviews. Reviews are scored by vote count, text length, and rating extremity; only non-empty review texts are kept.
    **Input:** 3_goodreads_reviews_dedup_clean.parquet, 3_goodreads_books_with_metrics.parquet
    **Output:** 4_goodreads_reviews_reduced.parquet
