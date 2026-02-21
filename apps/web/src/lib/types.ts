export type Book = {
  id: string;
  title: string;
  author: string;
  coverUrl: string | null | undefined;
  description?: string;
  tags?: string[];
  averageRating?: number | null;
  ratingCount?: number;
  commentCount?: number;
  shellCount?: number;
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
