export type Book = {
  id: string;
  title: string;
  author: string;
  coverUrl: string | null | undefined;
  description?: string;
  tags?: string[];
  publicationYear?: number | null;
  averageRating?: number | null;
  ratingCount?: number;
  commentCount?: number;
  shellCount?: number;
};

export type SearchBooksResponse = {
  directHit: Book | null;
  keywordResults: Book[];
  aiNarrative?: string | null;
  aiBooks: Book[];
};

export type User = {
  id: string;
  handle: string;
  displayName: string;
  avatarUrl?: string;
};

export type List = {
  id: string;
  name: string;
  description?: string;
  owner: User;
  books: Book[];
};

export type RatedBook = {
  book: Book;
  rating: number;
  ratedAt: string;
};

export type ActivityItem = {
  id: string;
  user: User;
  type: "rating" | "shell_add" | "list_add";
  book: Book;
  rating?: number;
  listName?: string;
  timestamp: string;
};

export type BookStats = {
  averageRating: number;
  ratingCount: number;
  commentCount: number;
  shellCount: number;
};

export type Review = {
  id: string;
  user: User;
  text: string;
  likes: number;
  isLikedByMe?: boolean;
  timestamp: string;
  replies?: Reply[];
};

export type Reply = {
  id: string;
  user: User;
  text: string;
  likes: number;
  isLikedByMe?: boolean;
  timestamp: string;
};

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

export type ChatSession = {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
};

export type ToolTrace = {
  tool: string;
  input: unknown;
  output: unknown;
  source?: string;
  data?: Record<string, unknown>;
};

export type ComparisonDimension = {
  name: string;
  values: string[];
};

export type ComparisonResult = {
  dimensions: ComparisonDimension[];
  verdict: string;
  bookIds: number[];
};

export type UserPreferences = {
  liked_genres?: string[];
  disliked_genres?: string[];
  max_pages?: number;
  standalone_only?: boolean;
  preferred_mood?: string;
  liked_books?: string[];
  disliked_books?: string[];
  other_constraints?: string[];
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "tool";
  content: string;
  toolName?: string;
  toolTrace?: ToolTrace;
  toolTraces?: ToolTrace[];
  comparison?: ComparisonResult;
  referencedBookIds?: number[];
  referencedBooks?: Book[];
  modelUsed?: string;
  timestamp: string;
};

export type ChatSessionDetail = {
  id: string;
  title: string;
  messages: ChatMessage[];
  preferences?: UserPreferences;
  createdAt: string;
};
