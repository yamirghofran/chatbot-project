// Package client provides an HTTP client for the BookDB FastAPI backend.
package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

// APIClient wraps HTTP communication with the BookDB FastAPI backend.
type APIClient struct {
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client
}

// NewAPIClient creates a client targeting the given BookDB API base URL.
func NewAPIClient(baseURL, apiKey string) *APIClient {
	return &APIClient{
		BaseURL: baseURL,
		APIKey:  apiKey,
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// ── Response types ──────────────────────────────────────────────────────────
// These map 1:1 to the JSON shapes returned by the FastAPI routers.

// Book represents a serialised book returned by the API.
type Book struct {
	ID          string   `json:"id"`
	Title       string   `json:"title"`
	Author      string   `json:"author"`
	CoverURL    *string  `json:"coverUrl"`
	Description *string  `json:"description"`
	Tags        []string `json:"tags"`
}

// BookStats holds engagement metrics for a book.
type BookStats struct {
	AverageRating *float64 `json:"averageRating"`
	RatingCount   int      `json:"ratingCount"`
	CommentCount  int      `json:"commentCount"`
	ShellCount    int      `json:"shellCount"`
}

// BookDetail is the full book response including stats.
type BookDetail struct {
	Book
	Stats           *BookStats `json:"stats"`
	PublicationYear *int       `json:"publicationYear"`
	ISBN13          *string    `json:"isbn13"`
}

// SearchResult is the response from /books/search.
type SearchResult struct {
	DirectHit      *BookDetail  `json:"directHit"`
	KeywordResults []BookDetail `json:"keywordResults"`
	AINarrative    *string      `json:"aiNarrative"`
	AIBooks        []BookDetail `json:"aiBooks"`
}

// Review represents a single review.
type Review struct {
	ID          string `json:"id"`
	User        User   `json:"user"`
	Text        string `json:"text"`
	Likes       int    `json:"likes"`
	IsLikedByMe bool   `json:"isLikedByMe"`
	Timestamp   string `json:"timestamp"`
}

// ReviewListResponse wraps paginated reviews.
type ReviewListResponse struct {
	Items []Review `json:"items"`
	Total int      `json:"total"`
}

// User represents a user profile.
type User struct {
	ID        string  `json:"id"`
	Name      string  `json:"name"`
	Username  string  `json:"username"`
	AvatarURL *string `json:"avatarUrl"`
}

// UserRating is a single user rating entry.
type UserRating struct {
	Book    Book    `json:"book"`
	Rating  int     `json:"rating"`
	RatedAt *string `json:"ratedAt"`
}

// LoginRequest is the body for POST /auth/login.
type LoginRequest struct {
	Email    string `json:"email"`
	Password string `json:"password"`
}

// RegisterRequest is the body for POST /auth/register.
type RegisterRequest struct {
	Email    string `json:"email"`
	Name     string `json:"name"`
	Username string `json:"username"`
	Password string `json:"password"`
}

// TokenResponse is the response from login/register.
type TokenResponse struct {
	AccessToken string `json:"access_token"`
	TokenType   string `json:"token_type"`
}

// ── HTTP helpers ────────────────────────────────────────────────────────────

func (c *APIClient) doJSON(ctx context.Context, method, path string, result any) error {
	return c.doJSONBody(ctx, method, path, nil, result)
}

func (c *APIClient) doJSONBody(ctx context.Context, method, path string, body any, result any) error {
	fullURL := c.BaseURL + path

	var bodyReader io.Reader
	if body != nil {
		bodyBytes, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("encoding request body: %w", err)
		}
		bodyReader = bytes.NewReader(bodyBytes)
	}

	req, err := http.NewRequestWithContext(ctx, method, fullURL, bodyReader)
	if err != nil {
		return fmt.Errorf("creating request: %w", err)
	}
	if c.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.APIKey)
	}
	req.Header.Set("Accept", "application/json")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return fmt.Errorf("executing request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("API returned %d: %s", resp.StatusCode, string(respBody))
	}

	if result != nil {
		if err := json.NewDecoder(resp.Body).Decode(result); err != nil {
			return fmt.Errorf("decoding response: %w", err)
		}
	}
	return nil
}

// ── Public methods ──────────────────────────────────────────────────────────

// SearchBooks calls GET /books/search?q=...&limit=...
func (c *APIClient) SearchBooks(ctx context.Context, query string, limit int) (*SearchResult, error) {
	path := fmt.Sprintf("/books/search?q=%s&limit=%d", url.QueryEscape(query), limit)
	var result SearchResult
	if err := c.doJSON(ctx, http.MethodGet, path, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// GetBook calls GET /books/{id}
func (c *APIClient) GetBook(ctx context.Context, bookID int) (*BookDetail, error) {
	path := fmt.Sprintf("/books/%d", bookID)
	var result BookDetail
	if err := c.doJSON(ctx, http.MethodGet, path, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// GetRelatedBooks calls GET /books/{id}/related?limit=...
func (c *APIClient) GetRelatedBooks(ctx context.Context, bookID, limit int) ([]BookDetail, error) {
	path := fmt.Sprintf("/books/%d/related?limit=%d", bookID, limit)
	var result []BookDetail
	if err := c.doJSON(ctx, http.MethodGet, path, &result); err != nil {
		return nil, err
	}
	return result, nil
}

// GetBookReviews calls GET /books/{id}/reviews?limit=...&offset=...
func (c *APIClient) GetBookReviews(ctx context.Context, bookID, limit, offset int) (*ReviewListResponse, error) {
	path := fmt.Sprintf("/books/%d/reviews?limit=%d&offset=%d", bookID, limit, offset)
	var result ReviewListResponse
	if err := c.doJSON(ctx, http.MethodGet, path, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// GetRecommendations calls GET /discovery/recommendations?limit=...
func (c *APIClient) GetRecommendations(ctx context.Context, limit int) ([]BookDetail, error) {
	path := fmt.Sprintf("/discovery/recommendations?limit=%d", limit)
	var result []BookDetail
	if err := c.doJSON(ctx, http.MethodGet, path, &result); err != nil {
		return nil, err
	}
	return result, nil
}

// GetStaffPicks calls GET /discovery/staff-picks?limit=...
func (c *APIClient) GetStaffPicks(ctx context.Context, limit int) ([]BookDetail, error) {
	path := fmt.Sprintf("/discovery/staff-picks?limit=%d", limit)
	var result []BookDetail
	if err := c.doJSON(ctx, http.MethodGet, path, &result); err != nil {
		return nil, err
	}
	return result, nil
}

// GetUserProfile calls GET /user/{username}
func (c *APIClient) GetUserProfile(ctx context.Context, username string) (*User, error) {
	path := fmt.Sprintf("/user/%s", url.PathEscape(username))
	var result User
	if err := c.doJSON(ctx, http.MethodGet, path, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// GetUserRatings calls GET /user/{username}/ratings?limit=...&sort=...
func (c *APIClient) GetUserRatings(ctx context.Context, username string, limit int, sort string) ([]UserRating, error) {
	path := fmt.Sprintf("/user/%s/ratings?limit=%d&sort=%s", url.PathEscape(username), limit, sort)
	var result []UserRating
	if err := c.doJSON(ctx, http.MethodGet, path, &result); err != nil {
		return nil, err
	}
	return result, nil
}

// GetShell calls GET /me/shell and returns the authenticated user's shell books.
func (c *APIClient) GetShell(ctx context.Context) ([]Book, error) {
	var result []Book
	if err := c.doJSON(ctx, http.MethodGet, "/me/shell", &result); err != nil {
		return nil, err
	}
	return result, nil
}

// AddToShell calls POST /me/shell/{bookID} to add a book to the authenticated user's shell.
func (c *APIClient) AddToShell(ctx context.Context, bookID int) error {
	path := fmt.Sprintf("/me/shell/%d", bookID)
	return c.doJSON(ctx, http.MethodPost, path, nil)
}

// Login calls POST /auth/login with email + password and returns a JWT token.
func (c *APIClient) Login(ctx context.Context, email, password string) (*TokenResponse, error) {
	var result TokenResponse
	if err := c.doJSONBody(ctx, http.MethodPost, "/auth/login", LoginRequest{
		Email:    email,
		Password: password,
	}, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// Register calls POST /auth/register and returns a JWT token for the new account.
func (c *APIClient) Register(ctx context.Context, email, name, username, password string) (*TokenResponse, error) {
	var result TokenResponse
	if err := c.doJSONBody(ctx, http.MethodPost, "/auth/register", RegisterRequest{
		Email:    email,
		Name:     name,
		Username: username,
		Password: password,
	}, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// FormatBookDetail produces a human-readable summary of a BookDetail.
func FormatBookDetail(b BookDetail) string {
	text := fmt.Sprintf("**%s** by %s", b.Title, b.Author)
	if b.ID != "" {
		text += fmt.Sprintf(" [ID: %s]", b.ID)
	}
	if b.PublicationYear != nil {
		text += fmt.Sprintf(" (%d)", *b.PublicationYear)
	}
	if b.Stats != nil && b.Stats.RatingCount > 0 {
		text += fmt.Sprintf("\n⭐ %s/5 (%d ratings)", float64ToStr(b.Stats.AverageRating), b.Stats.RatingCount)
	}
	if len(b.Tags) > 0 {
		text += fmt.Sprintf("\nTags: %v", b.Tags)
	}
	if b.Description != nil && *b.Description != "" {
		desc := *b.Description
		if len(desc) > 300 {
			desc = desc[:300] + "..."
		}
		text += fmt.Sprintf("\n> %s", desc)
	}
	return text
}

// FormatBook produces a human-readable one-liner.
func FormatBook(b Book) string {
	return fmt.Sprintf("- **%s** by %s", b.Title, b.Author)
}

func float64ToStr(v *float64) string {
	if v == nil {
		return "N/A"
	}
	return strconv.FormatFloat(*v, 'f', 1, 64)
}
