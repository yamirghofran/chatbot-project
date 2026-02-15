## Database Schema

users (id, email, name, created_at)
authors (id, name, external_id, created_at)
books (id, title, description, image_url, publication_year, book_id, author_id, similar_books, created_at)
reviews (id, user_id, book_id, text, created_at)
ratings (user_id [PK], book_id [PK], rating 1-5)
lists (id, name, user_id, created_at)
book_authors junction (book_id, author_id)
list_books junction (list_id, book_id)


## Dataset → DB Column Mapping

### Books (Goodreads → DB)
| Dataset Column     | DB Column        | Notes                         |
|--------------------|------------------|-------------------------------|
| title              | title            |                               |
| original_title     | title            | book_works dataset variant    |
| description        | description      |                               |
| image_url          | image_url        |                               |
| publication_year   | publication_year |                               |
| book_id            | book_id          | external identifier           |
| similar_books      | similar_books    | JSON list of external book IDs|

**Not stored:** isbn, isbn13, asin, kindle_asin, text_reviews_count, series, country_code, language_code, popular_shelves, is_ebook, average_rating, format, link, publisher, num_pages, publication_day, publication_month, edition_information, url, ratings_count, work_id, title_without_series

### Authors (Goodreads → DB)
| Dataset Column | DB Column   | Notes               |
|----------------|-------------|----------------------|
| name           | name        |                      |
| author_id      | external_id | Goodreads author ID  |

**Not stored:** average_rating, ratings_count, text_reviews_count


## Dataset Columns (raw)

Books: isbn, text_reviews_count, series, country_code, language_code, popular_shelves, asin, is_ebook, average_rating, kindle_asin, similar_books, description, format, link, authors, publisher, num_pages, publication_day, isbn13, publication_month, edition_information, publication_year, url, image_url, book_id, ratings_count, work_id, title, title_without_series

Authors: average_rating, author_id, text_reviews_count, name, ratings_count
