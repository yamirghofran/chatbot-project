import type { Book, List } from "./types";

export const homeStaffPicks: Book[] = [
  {
    id: "1268332",
    title: "Infinite Jest",
    author: "David Foster Wallace",
    coverUrl: "https://images.gr-assets.com/books/1446876799m/6759.jpg",
  },
  {
    id: "1278794",
    title: "Lolita",
    author: "Craig Raine, Vladimir Nabokov",
    coverUrl: "https://images.gr-assets.com/books/1377756377m/7604.jpg",
  },
  {
    id: "281296",
    title: "Pale Fire",
    author: "Vladimir Nabokov",
    coverUrl: "https://images.gr-assets.com/books/1388155863m/7805.jpg",
  },
];

// Fallback only. Real trending lists are loaded from /user/bookdb/lists.
export const homeTrendingLists: List[] = [];
