export type Book = {
  id: string;
  title: string;
  author: string;
  coverUrl: string;
  description?: string;
  tags?: string[];
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
  owner: User;
  books: Book[];
};

export type ActivityItem = {
  id: string;
  user: User;
  type: "rating" | "favourite" | "list_add";
  book: Book;
  rating?: number;
  listName?: string;
  timestamp: string;
};
