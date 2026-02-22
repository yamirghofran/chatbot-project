from apps.api.core.serialize import resolve_cover_url


def test_resolve_cover_url_prefers_openlibrary_for_goodreads_nophoto():
    image_url = "https://s.gr-assets.com/assets/nophoto/book/111x148-bcc042.png"
    isbn13 = "9781400079278"

    result = resolve_cover_url(image_url, isbn13)

    assert result == "https://covers.openlibrary.org/b/isbn/9781400079278-L.jpg"


def test_resolve_cover_url_keeps_non_placeholder_image():
    image_url = "https://images.gr-assets.com/books/1327867963m/117833.jpg"
    isbn13 = "9781400079278"

    result = resolve_cover_url(image_url, isbn13)

    assert result == image_url


def test_resolve_cover_url_falls_back_to_original_when_no_isbn():
    image_url = "https://s.gr-assets.com/assets/nophoto/book/111x148-bcc042.png"

    result = resolve_cover_url(image_url, None)

    assert result == image_url
